"""
DetAny3D eval JSON -> ovmono3d-compatible instances_predictions.pth.

DetAny3D's :func:`validate_one_epoch` (train.py) writes one big JSON
per dataset:

    [
        {
            "image_id": int,
            "bbox": [x, y, w, h],      # xywh, original-image pixels
            "category_id": int,        # contiguous (matches our category_meta)
            "score": float,
            "depth": float,
            "bbox3D": [[x,y,z]*8],     # 8 corners (DetAny3D's reordered convention)
            "center_cam": [x,y,z],
            "center_2D": [cx, cy],
            "pose": [[3x3]],           # rotation matrix (camera-from-local)
            "dimensions": [w, h, l],   # DetAny3D ordering: [X-extent, Y-extent, Z-extent]
            "area": float,
            "yaw": float
        },
        ...
    ]

ovmono3d's eval stack (bev_ap_eval.py, class_agnostic_eval.py,
omni3d_evaluation.py) consumes a detectron2-format
``instances_predictions.pth`` -- a list of one dict per image, each with::

    {
        "image_id": int,
        "instances": [
            {
                "image_id": int,
                "image_width": int,
                "image_height": int,
                "category_id": int,             # contiguous 0..N-1
                "bbox": [x, y, w, h],           # xywh
                "score": float,
                "score_3d": float,              # alias of score
                "center_cam": [x, y, z],
                "dimensions": [W, H, L],        # OMNI3D ordering -- swap from DetAny3D's [w,h,l]
                "pose": [[3x3]] or [9 floats],
                "depth": float,
                "bbox3D_cam": [[x,y,z]*8]
            },
            ...
        ]
    }

The two key normalizations:
  1. Reverse ``dimensions`` (DetAny3D ``[w,h,l]`` -> Omni3D ``[L,H,W] == [W,H,L]``
     interpreted as Omni3D's storage order). See
     ``detect_anything/datasets/data_creator/wildbox.py`` for the original swap.
  2. Group per-instance dicts by ``image_id`` and add the per-image
     ``image_width``/``image_height`` from the GT JSON.

The category_id mapping is already consistent (we used the same
category_meta during training and conversion), so no remap needed --
both ovmono3d and DetAny3D end up with contiguous ids in the same
ascending-dataset-id order.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

logger = logging.getLogger("wildbox_export")


def _da3d_dims_to_omni3d(dims_da3d: List[float]) -> List[float]:
    """Reverse [w, h, l] (DetAny3D) -> [W, H, L] (Omni3D)."""
    w, h, l = dims_da3d
    return [l, h, w]


def export(
    da3d_json_path: Path,
    gt_json_path: Path,
    out_pth_path: Path,
    score_threshold: float = 0.0,
) -> Dict[str, int]:
    """Convert one DetAny3D eval-JSON to instances_predictions.pth."""
    da3d_json_path = Path(da3d_json_path)
    gt_json_path = Path(gt_json_path)
    out_pth_path = Path(out_pth_path)
    out_pth_path.parent.mkdir(parents=True, exist_ok=True)

    with open(da3d_json_path, "r") as f:
        da3d_preds = json.load(f)
    with open(gt_json_path, "r") as f:
        gt = json.load(f)

    image_meta = {
        int(img["id"]): (int(img["width"]), int(img["height"]))
        for img in gt["images"]
    }
    image_ids_in_gt = set(image_meta.keys())

    by_image: Dict[int, List[Dict]] = {}
    n_skipped_score = 0
    n_skipped_unknown_image = 0

    for p in da3d_preds:
        if p.get("score", 0.0) < score_threshold:
            n_skipped_score += 1
            continue
        image_id = int(p["image_id"])
        if image_id not in image_ids_in_gt:
            n_skipped_unknown_image += 1
            continue
        w_im, h_im = image_meta[image_id]

        # Pose: DetAny3D writes a 3x3 list-of-lists. ovmono3d's evaluators
        # accept either 3x3 or 9-flat -- pass 3x3 unchanged.
        pose = p["pose"]

        dims_omni3d = _da3d_dims_to_omni3d(p["dimensions"])

        inst = {
            "image_id": image_id,
            "image_width": w_im,
            "image_height": h_im,
            "category_id": int(p["category_id"]),
            "bbox": [float(v) for v in p["bbox"]],         # xywh
            "score": float(p["score"]),
            "score_3d": float(p["score"]),
            "center_cam": [float(v) for v in p["center_cam"]],
            "dimensions": [float(v) for v in dims_omni3d],
            "pose": pose,
            "depth": float(p["depth"]),
            "bbox3D_cam": p["bbox3D"],
            "yaw": float(p.get("yaw", 0.0)),
        }
        by_image.setdefault(image_id, []).append(inst)

    # detectron2 format expects one entry per image (even if no preds).
    pred_list = []
    for image_id in sorted(image_ids_in_gt):
        pred_list.append({
            "image_id": image_id,
            "instances": by_image.get(image_id, []),
        })

    torch.save(pred_list, out_pth_path)
    logger.info(f"wrote {sum(len(p['instances']) for p in pred_list)} instances "
                f"across {len(pred_list)} images -> {out_pth_path}")

    return {
        "predictions_in": len(da3d_preds),
        "instances_out": sum(len(p["instances"]) for p in pred_list),
        "images_out": len(pred_list),
        "skipped_score_threshold": n_skipped_score,
        "skipped_unknown_image_id": n_skipped_unknown_image,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--da3d-json", type=Path, required=True,
                        help="DetAny3D eval JSON "
                             "(e.g. exps/wildbox_smoke/<ts>/wildbox_<...>.json).")
    parser.add_argument("--gt-json", type=Path, required=True,
                        help="ovmono3d WildBox_val.json (Omni3D schema). "
                             "Used for image_id -> (width, height) lookup and "
                             "to ensure all val images get an entry in the .pth.")
    parser.add_argument("--out-pth", type=Path, required=True,
                        help="Output instances_predictions.pth path. Convention: "
                             "<ovmono3d_run_dir>/inference/iter_final/WildBox_val/"
                             "instances_predictions.pth")
    parser.add_argument("--score-threshold", type=float, default=0.0,
                        help="Drop predictions below this score "
                             "(DetAny3D default scoring is loose).")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    counts = export(
        da3d_json_path=args.da3d_json,
        gt_json_path=args.gt_json,
        out_pth_path=args.out_pth,
        score_threshold=args.score_threshold,
    )
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
