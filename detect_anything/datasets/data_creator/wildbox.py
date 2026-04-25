"""
WildBox (Omni3D-schema JSON) -> DetAny3D pickle converter.

Reads an Omni3D-style JSON produced by ovmono3d's
``tools/prepare_wildbox_dataset.py`` (i.e. ``WildBox_train.json`` /
``WildBox_val.json``) and writes a DetAny3D-style pickle that
``detect_anything.datasets.detany3d_dataset.DetAny3DDataset`` can load.

Why this is needed
------------------
DetAny3D's data loader expects a list of per-image dicts with this shape::

    {
        'K': np.ndarray (1, 3, 3),
        'img_path': <absolute path>,
        'depth_path': None,                      # WildBox has no metric depth
        'obj_list': [
            {
                '3d_bbox': [x, y, z, w, h, l, yaw],
                '2d_bbox_proj': [x1, y1, x2, y2],          # xyxy
                'rotation_pose': np.ndarray (3, 3),
                'instance_id': str,
                'label': int,                              # contiguous 0..N-1
                'image_id': int,
                'score': 1.0,
                'visibility': 1.0,
                'truncation': 0.0,
            }, ...
        ],
    }

The Omni3D-schema JSON uses absolute file paths and per-image intrinsics
already, so the conversion is mostly a re-indexing exercise plus one
subtle convention swap: dimensions ordering.

Convention swap (read carefully before changing)
------------------------------------------------
Omni3D stores ``dimensions = [W, H, L]`` with the axis assignment
``X = L, Y = H, Z = W`` (per ovmono3d/WILDBOX_EXPERIMENT.md §2.4).

DetAny3D's :func:`compute_3d_bbox_vertices` (in
``detect_anything/datasets/utils.py``) takes ``[w, h, l]`` and emits
``corners = [[+/-w/2, +/-h/2, +/-l/2]]`` -> X-extent is ``w``, Y-extent
is ``h``, Z-extent is ``l``.

So::

    DetAny3D.w = Omni3D.L = WildBox.dimensions[2]   # X-extent
    DetAny3D.h = Omni3D.H = WildBox.dimensions[1]   # Y-extent
    DetAny3D.l = Omni3D.W = WildBox.dimensions[0]   # Z-extent

i.e. ``[w, h, l] = reversed(WildBox.dimensions)``.

The rotation matrix ``R_cam`` is assumed compatible -- it rotates the
local cuboid frame to camera frame, and both conventions agree that the
local frame is camera-aligned at zero rotation. If post-training BEV/3D
AP looks anomalous on the smoke run, suspect a column permutation here
and re-validate with ``--dry-run --verify-projection``.

Yaw
---
``cfg.output_rotation_matrix: True`` is required for WildBox (top-down
drone shots have non-trivial pitch/roll, not just yaw). The 3d_bbox's
yaw value is consequently irrelevant for the loss -- DetAny3D reads
``rotation_pose`` directly when ``output_rotation_matrix`` is set. We
write yaw=0.0 as a placeholder.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("wildbox_converter")


def _wildbox_dims_to_da3d(dims_omni3d: List[float]) -> List[float]:
    """Reverse [W, H, L] -> [w, h, l]. See module docstring."""
    W, H, L = dims_omni3d
    return [L, H, W]


def _project_points(points_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project (N, 3) camera-frame points to (N, 2) image coords. K is (3, 3)."""
    pts_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    K_ext = np.hstack([K, np.zeros((3, 1))])
    p2 = K_ext @ pts_h.T
    return (p2[:2, :] / p2[2:3, :]).T


def _verify_projection(
    center_cam: np.ndarray,
    dims_da3d: List[float],
    R_cam: np.ndarray,
    K: np.ndarray,
    bbox2d_proj_gt: List[float],
    img_w: int,
    img_h: int,
    tol_px: float = 4.0,
) -> Tuple[bool, float]:
    """Project the cuboid corners and compare against the ground-truth
    bbox2D_proj. Returns ``(ok, max_corner_dist)``.

    Used as a sanity check that the dimension/rotation convention swap
    is correct. We clip the projected box to image bounds the same way
    ovmono3d's prep does (otherwise off-screen corners cause spurious
    mismatches when the GT happens to be border-clipped).
    """
    w, h, l = dims_da3d
    corners = np.array([
        [w / 2, h / 2, l / 2], [w / 2, -h / 2, l / 2],
        [-w / 2, -h / 2, l / 2], [-w / 2, h / 2, l / 2],
        [w / 2, h / 2, -l / 2], [w / 2, -h / 2, -l / 2],
        [-w / 2, -h / 2, -l / 2], [-w / 2, h / 2, -l / 2],
    ])
    corners_cam = (R_cam @ corners.T).T + center_cam
    pts_2d = _project_points(corners_cam, K)
    u_min = float(np.clip(pts_2d[:, 0].min(), 0, img_w - 1))
    v_min = float(np.clip(pts_2d[:, 1].min(), 0, img_h - 1))
    u_max = float(np.clip(pts_2d[:, 0].max(), 0, img_w - 1))
    v_max = float(np.clip(pts_2d[:, 1].max(), 0, img_h - 1))
    expected = np.array(bbox2d_proj_gt)
    actual = np.array([u_min, v_min, u_max, v_max])
    dist = float(np.max(np.abs(actual - expected)))
    return dist <= tol_px, dist


def convert(
    wildbox_json_path: Path,
    category_meta_path: Path,
    pkl_out_path: Path,
    path_remap: Optional[Tuple[str, str]] = None,
    skip_missing_paths: bool = False,
    verify_projection: bool = False,
    limit: int = 0,
) -> Dict[str, int]:
    """Convert one WildBox Omni3D-schema JSON to a DetAny3D pickle.

    Parameters
    ----------
    wildbox_json_path
        Path to ``WildBox_train.json`` or ``WildBox_val.json``.
    category_meta_path
        Path to ``category_meta_wildbox.json`` so we can map dataset_id
        -> contiguous_id consistently with the model head ordering.
    pkl_out_path
        Output ``.pkl``. Parent directory must exist.
    path_remap
        Optional ``(old_prefix, new_prefix)`` for rewriting absolute
        ``file_path`` values. Useful when JSON was authored on a
        different mount than the cluster the pickle will be loaded on.
    skip_missing_paths
        If True, drop images whose file_path doesn't exist on disk.
        Off by default -- on the WSL dev machine images won't exist
        but on the cluster they will.
    verify_projection
        If True, project a sample of cuboids and compare against
        ``bbox2D_proj`` to catch convention-swap bugs.
    limit
        If >0, stop after this many images. Useful for unit-testing.

    Returns
    -------
    Counts of {images_in, annotations_in, images_out, annotations_out,
    images_dropped_missing_path, annotations_dropped_filter}.
    """
    wildbox_json_path = Path(wildbox_json_path)
    category_meta_path = Path(category_meta_path)
    pkl_out_path = Path(pkl_out_path)
    pkl_out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(wildbox_json_path, "r") as f:
        data = json.load(f)
    with open(category_meta_path, "r") as f:
        meta = json.load(f)

    id_to_contig = {int(k): int(v)
                    for k, v in meta["thing_dataset_id_to_contiguous_id"].items()}
    thing_classes = meta["thing_classes"]
    n_classes = len(thing_classes)
    if max(id_to_contig.values()) >= n_classes:
        raise ValueError(f"category_meta inconsistent: contiguous ids "
                         f"reach {max(id_to_contig.values())} but only "
                         f"{n_classes} thing_classes")

    # Index annotations by image_id.
    anns_by_image: Dict[int, list] = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    images_in = len(data["images"])
    anns_in = len(data["annotations"])
    images_dropped_missing = 0
    anns_dropped_filter = 0
    proj_failures: List[Tuple[int, float]] = []  # (ann id, max-pixel dist)
    proj_total = 0

    pkl_records: List[Dict] = []

    for img_idx, img in enumerate(data["images"]):
        if limit and img_idx >= limit:
            break

        file_path = img["file_path"]
        if path_remap is not None and file_path.startswith(path_remap[0]):
            file_path = path_remap[1] + file_path[len(path_remap[0]):]
        if skip_missing_paths and not os.path.exists(file_path):
            images_dropped_missing += 1
            continue

        K_3x3 = np.array(img["K"], dtype=np.float32)  # (3, 3)
        K_with_batch = K_3x3[None, :, :]              # DetAny3D wants (1, 3, 3)

        obj_list = []
        for ann in anns_by_image.get(img["id"], []):
            ds_id = ann.get("dataset_id", ann.get("category_id"))
            if ds_id not in id_to_contig:
                anns_dropped_filter += 1
                continue
            contig_id = id_to_contig[ds_id]

            center_cam = np.array(ann["center_cam"], dtype=np.float64)
            dims_omni3d = list(ann["dimensions"])
            dims_da3d = _wildbox_dims_to_da3d(dims_omni3d)
            R_cam = np.array(ann["R_cam"], dtype=np.float64)

            # Drop degenerate cuboids defensively.
            if any(d <= 0 for d in dims_da3d) or center_cam[2] <= 0:
                anns_dropped_filter += 1
                continue

            if verify_projection and "bbox2D_proj" in ann:
                ok, dist = _verify_projection(
                    center_cam, dims_da3d, R_cam, K_3x3, ann["bbox2D_proj"],
                    img_w=int(img["width"]), img_h=int(img["height"]),
                )
                proj_total += 1
                if not ok:
                    proj_failures.append((ann["id"], dist))

            # 2D box: prefer SAM3-tight if present (xyxy), else bbox2D_proj.
            if ann.get("bbox2D_tight") is not None:
                bbox_xyxy = list(ann["bbox2D_tight"])
            elif ann.get("bbox2D_proj") is not None:
                bbox_xyxy = list(ann["bbox2D_proj"])
            else:
                # 'bbox' is xywh; convert.
                x, y, w, h = ann["bbox"]
                bbox_xyxy = [x, y, x + w, y + h]

            obj = {
                "3d_bbox": [
                    float(center_cam[0]), float(center_cam[1]), float(center_cam[2]),
                    float(dims_da3d[0]), float(dims_da3d[1]), float(dims_da3d[2]),
                    0.0,  # yaw placeholder; rotation_pose is authoritative
                ],
                "2d_bbox_proj": [float(v) for v in bbox_xyxy],
                "rotation_pose": R_cam.astype(np.float64),
                "instance_id": f"{ann['image_id']}_{ann['id']}",
                "label": contig_id,
                "image_id": int(ann["image_id"]),
                "score": 1.0,
                "visibility": float(ann.get("visibility", 1.0)),
                "truncation": float(ann.get("truncation", 0.0)),
            }
            obj_list.append(obj)

        if not obj_list:
            # DetAny3D can handle empty obj_list at val time but it
            # contributes nothing -- drop to keep the pickle dense.
            anns_dropped_filter += 0
            continue

        pkl_records.append({
            "K": K_with_batch.astype(np.float32),
            "img_path": file_path,
            "depth_path": None,                       # synthetic-scale GT, no metric depth
            "obj_list": obj_list,
        })

    with open(pkl_out_path, "wb") as f:
        pickle.dump(pkl_records, f)

    logger.info(f"wrote {len(pkl_records)} samples ({sum(len(r['obj_list']) for r in pkl_records)} obj) -> {pkl_out_path}")
    if verify_projection:
        n_fail = len(proj_failures)
        worst = max((d for _, d in proj_failures), default=0.0)
        logger.info(f"projection verify: {proj_total - n_fail}/{proj_total} ok "
                    f"(tol=4 px), worst_dist={worst:.2f} px")
        if n_fail > proj_total * 0.05:
            logger.warning(f"more than 5% of cuboids failed projection check "
                           f"-- the dimension/rotation convention swap may be wrong")

    return {
        "images_in": images_in,
        "annotations_in": anns_in,
        "images_out": len(pkl_records),
        "annotations_out": sum(len(r["obj_list"]) for r in pkl_records),
        "images_dropped_missing_path": images_dropped_missing,
        "annotations_dropped_filter": anns_dropped_filter,
        "projection_failures": len(proj_failures),
    }


def _parse_remap(arg: Optional[str]) -> Optional[Tuple[str, str]]:
    if not arg:
        return None
    if "=" not in arg:
        raise argparse.ArgumentTypeError("--path-remap must be OLD=NEW")
    old, new = arg.split("=", 1)
    return (old, new)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wildbox-json", type=Path, required=True,
                        help="Path to WildBox_{train,val}.json (Omni3D schema).")
    parser.add_argument("--category-meta", type=Path,
                        default=Path("data/category_meta_wildbox.json"),
                        help="Path to category_meta_wildbox.json.")
    parser.add_argument("--out-pkl", type=Path, required=True,
                        help="Output pickle path.")
    parser.add_argument("--path-remap", type=str, default=None,
                        help="OLD=NEW prefix rewrite for image file_path "
                             "(useful when JSON was generated on a different mount).")
    parser.add_argument("--skip-missing-paths", action="store_true",
                        help="Drop images whose file_path doesn't exist. "
                             "Off by default; on the cluster all paths should exist.")
    parser.add_argument("--verify-projection", action="store_true",
                        help="Project cuboids and compare against bbox2D_proj. "
                             "Sanity check for the dim/rotation convention swap.")
    parser.add_argument("--limit", type=int, default=0,
                        help="If >0, only convert the first N images (for testing).")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    counts = convert(
        wildbox_json_path=args.wildbox_json,
        category_meta_path=args.category_meta,
        pkl_out_path=args.out_pkl,
        path_remap=_parse_remap(args.path_remap),
        skip_missing_paths=args.skip_missing_paths,
        verify_projection=args.verify_projection,
        limit=args.limit,
    )
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
