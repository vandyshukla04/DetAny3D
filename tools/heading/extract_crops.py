"""Cut the animal crops for the motion-labelled frames.  [LOCAL / CPU -- run where the data is]

    python -m tools.heading.extract_crops \
        --labels data/heading/labels.npz --out data/heading/crops.npz

WHY THIS IS A SEPARATE STEP
---------------------------
`papersubdata` lives on the local disk; DINOv3 and its GPU live only on the cluster. Rather
than sync the dataset, we ship the one thing the GPU needs: the crops, JPEG-encoded inside a
single .npz. One `scp`, no directory tree to keep in sync.

SQUARE-PAD, DO NOT STRETCH  (correctness, not cosmetics)
--------------------------------------------------------
Squashing a wide crop into a square is not a similarity transform -- it CHANGES ANGLES. We
pad to a square first (angle-preserving) and only then resize, so the 2D positions we store
alongside the crop are consistent with the pixels the network sees.

THE CROP BOX IS `bbox_2d`, RESCALED BY A *MEASURED* RATIO
---------------------------------------------------------
`bbox_2d` is the detector's own tight 2D box -- better than a projected 3D hull, which
inherits the box fit's slop. But it lives in VGGT's 518-space while the frames are 1920x1080,
and reading it as full-res is what cut every crop from the background last time. So
`papersub.Segment.scale` *measures* the ratio (by projecting the 3D centres with the verified
full-res K) rather than hardcoding 3.707, and raises if a segment disagrees.

WHAT TRAVELS WITH EACH CROP
---------------------------
  * the 4 candidate FRONT faces, with their centres projected INTO CROP COORDINATES
    -> the probe asks: does the DINOv3 head-part land nearest the true one?  (chance 25%)
  * each candidate's allocentric angle, and the TRUE front face from motion
    -> the learned baseline predicts alpha and is scored on the SAME 4-way question.
Both methods are therefore compared on identical crops with an identical metric.
"""
from __future__ import annotations

import argparse
import io
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.heading.cropset import load_npz

from tools.heading.conventions import face_centers_world
from tools.heading.papersub import load_segment

__all__ = ["square_crop", "crop_transform"]


def crop_transform(bbox, pad: float):
    """The square crop's origin and side, so a full-res pixel maps into crop coords.

    Returns (ox, oy, side): a full-res point (u, v) lands at ((u-ox)/side, (v-oy)/side) in
    [0,1]^2. Kept next to `square_crop` because the two MUST agree -- deriving the mapping
    separately is exactly how they drift apart.
    """
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float64)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + pad)
    return cx - side / 2.0, cy - side / 2.0, side


def square_crop(img: np.ndarray, bbox, pad: float) -> np.ndarray | None:
    """Crop the animal and pad to a SQUARE (angle-preserving; see module docstring)."""
    h, w = img.shape[:2]
    ox, oy, side = crop_transform(bbox, pad)
    if side < 8:
        return None

    sx1, sy1 = int(round(ox)), int(round(oy))
    sx2, sy2 = int(round(ox + side)), int(round(oy + side))

    out = np.zeros((sy2 - sy1, sx2 - sx1, 3), dtype=img.dtype)   # zero-pad past the edges
    ix1, iy1 = max(sx1, 0), max(sy1, 0)
    ix2, iy2 = min(sx2, w), min(sy2, h)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    out[iy1 - sy1: iy2 - sy1, ix1 - sx1: ix2 - sx1] = img[iy1:iy2, ix1:ix2]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, required=True, help="from tools.heading.autolabel")
    ap.add_argument("--root", type=Path, default=Path("/mnt/d/3DBOX/papersubdata"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--pad", type=float, default=0.15)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every Nth label (adjacent frames are near-duplicates)")
    ap.add_argument("--min-body-px", type=float, default=24.0,
                    help="drop animals smaller than this on screen: DINOv3 gives a 14x14 patch "
                         "grid over the crop, so a tiny animal has no parts left to find")
    args = ap.parse_args()

    from PIL import Image

    from tools.heading.conventions import FACE_AXIS

    d = load_npz(args.labels)
    n_all = len(d["seg"])
    sel = np.arange(0, n_all, max(1, args.stride))
    print(f"{n_all} labels -> {len(sel)} after stride {args.stride}")

    # Group by segment so each tracking_summary.json is parsed once and each frame decoded once.
    by_seg: dict[str, list[int]] = defaultdict(list)
    for j in sel:
        by_seg[str(d["seg"][j])].append(int(j))

    jpegs: list[bytes] = []
    rec: list[dict] = []
    n_small = n_bad = 0

    for si, (seg_key, idxs) in enumerate(sorted(by_seg.items()), 1):
        seg = load_segment(args.root / seg_key)
        _ = seg.scale                                  # measure + assert the 518 ratio, once
        idxs.sort(key=lambda j: int(d["frame"][j]))
        last_fidx, img = None, None

        for j in idxs:
            tr = seg.tracks[str(d["track"][j])]
            i, fidx = int(d["i"][j]), int(d["frame"][j])

            box = seg.crop_box(tr, i)                 # bbox_2d x measured scale
            body_px = float(max(box[2] - box[0], box[3] - box[1]))
            if body_px < args.min_body_px:
                n_small += 1
                continue

            if fidx != last_fidx:
                fp = seg.frame_path(fidx)
                if not fp.is_file():
                    n_bad += 1
                    continue
                img = np.asarray(Image.open(fp).convert("RGB"))
                last_fidx = fidx

            crop = square_crop(img, box, args.pad)
            if crop is None:
                n_bad += 1
                continue

            # --- map the 4 candidate front faces into CROP coordinates ---
            faces = seg.horizontal_face_dirs(tr, i)
            fids = sorted(faces)
            if len(fids) != 4:
                n_bad += 1
                continue
            centres = face_centers_world(tr.centers[i], tr.dims[i], tr.rotations[i])
            uv = seg.cameras[fidx].project(np.stack([centres[f] for f in fids]))
            if not np.isfinite(uv).all():
                n_bad += 1
                continue
            ox, oy, side = crop_transform(box, args.pad)
            uv = (uv - np.array([ox, oy])) / side              # -> [0,1] within the square crop

            # GEOMETRY'S PROPOSAL: which slot sits on the longest horizontal box axis. Appearance
            # is measurably bad at choosing the axis (letting it try dropped the 4-way to 39% while
            # the head/tail cue itself was 83.7%), so we hand it the axis and ask only for the sign.
            up = seg.up_at(tr, i)
            ext = {}
            for k, f in enumerate(fids):
                c = FACE_AXIS[f]
                a = tr.rotations[i][:, c] - np.dot(tr.rotations[i][:, c], up) * up
                ext[k] = float(tr.dims[i][c] * np.linalg.norm(a))
            geo_axis = int(max(ext, key=ext.get))

            # the drone, as seen from the animal: azimuth in the ground basis, and elevation above
            # the animal's horizontal plane (the aerial-oblique geometry, made explicit)
            to_cam = seg.cameras[fidx].center - tr.centers[i]
            h_cam = to_cam - np.dot(to_cam, up) * up
            cam_az = seg.azimuth_of(h_cam) if np.linalg.norm(h_cam) > 1e-9 else 0.0
            cam_elev = float(np.arctan2(np.dot(to_cam, up), np.linalg.norm(h_cam) + 1e-12))

            buf = io.BytesIO()
            Image.fromarray(crop).resize((args.size, args.size), Image.BICUBIC).save(
                buf, format="JPEG", quality=args.quality)
            jpegs.append(buf.getvalue())
            rec.append({
                "seg": seg_key, "video": str(d["video"][j]), "species": str(d["species"][j]),
                "track": f"{seg_key}::{tr.tid}", "frame": fidx,
                "face_ids": np.array(fids, dtype=np.int8),
                "face_uv": uv.astype(np.float32),                          # (4, 2), crop coords
                "face_alpha": np.array([seg.alpha_of(tr, i, faces[f]) for f in fids],
                                       dtype=np.float32),
                # WORLD azimuth of each candidate, in the segment's fixed ground basis. This is
                # what the tracklet decoder smooths over: alpha is view-relative, so a STANDING
                # animal's alpha drifts as the drone moves, and smoothing it would be smoothing the
                # camera. World azimuth is the quantity that is actually temporally coherent.
                "face_az": np.array([seg.azimuth_of(faces[f]) for f in fids], dtype=np.float32),
                "az": float(seg.azimuth_of(d["heading"][j])),
                # WHERE THE CAMERA IS, relative to the animal, in the segment's ground basis.
                # cam_az is recoverable from face_az - face_alpha, but cam_elev is not -- and
                # without it the world panel of the paper figure would have to invent the drone's
                # height. Two floats; store them rather than fake them.
                "cam_az": float(cam_az), "cam_elev": float(cam_elev),
                "front_face": int(d["front_face"][j]),
                "alpha": float(seg.alpha_of(tr, i, d["heading"][j])),
                "body_px": body_px,
                "geo_axis": geo_axis,
                # The EXACT square this crop was cut from, in full-res pixels, plus the frame it
                # came from. The cluster reads these to cut the SAM instance mask identically --
                # nothing is recomputed there, so the mask cannot drift out of alignment with the
                # pixels the network sees.
                "crop_box": np.array([ox, oy, side], dtype=np.float32),
                "box2d": box.astype(np.float32),          # for the mask->track join assert
                "image_name": seg.cameras[fidx].image_name,
            })
        if si % 25 == 0:
            print(f"  {si}/{len(by_seg)} segments, {len(rec)} crops", flush=True)

    if not rec:
        print("no crops produced")
        return 1

    # index (0..3) of the TRUE front face within face_ids -- the 4-way classification target
    y_face = np.array([int(np.where(r["face_ids"] == r["front_face"])[0][0]) for r in rec],
                      dtype=np.int8)
    alpha = np.array([r["alpha"] for r in rec], dtype=np.float32)

    out = dict(
        jpeg=np.array(jpegs, dtype=object),
        y_face=y_face,                                             # 4-way target (chance 25%)
        Y=np.stack([np.cos(alpha), np.sin(alpha)], axis=1).astype(np.float32),   # allocentric
        face_uv=np.stack([r["face_uv"] for r in rec]),
        face_alpha=np.stack([r["face_alpha"] for r in rec]),
        face_az=np.stack([r["face_az"] for r in rec]),               # WORLD azimuth per candidate
        az=np.array([r["az"] for r in rec], dtype=np.float32),       # the TRUE world azimuth
        cam_az=np.array([r["cam_az"] for r in rec], dtype=np.float32),
        cam_elev=np.array([r["cam_elev"] for r in rec], dtype=np.float32),
        face_ids=np.stack([r["face_ids"] for r in rec]),
        geo_axis=np.array([r["geo_axis"] for r in rec], dtype=np.int8),
        species=np.array([r["species"] for r in rec]),
        video=np.array([r["video"] for r in rec]),
        seg=np.array([r["seg"] for r in rec]),
        track=np.array([r["track"] for r in rec]),
        frame=np.array([r["frame"] for r in rec], dtype=np.int32),
        body_px=np.array([r["body_px"] for r in rec], dtype=np.float32),
        crop_box=np.stack([r["crop_box"] for r in rec]),          # (ox, oy, side), full-res px
        box2d=np.stack([r["box2d"] for r in rec]),                # the animal's own 2D box
        image_name=np.array([r["image_name"] for r in rec]),      # joins to obj_<tid>/<stem>.png
        seg_name=np.array([r["seg"].split("/")[-1] for r in rec]),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)

    mb = args.out.stat().st_size / 1e6
    print(f"\nwrote {len(rec)} crops -> {args.out}  ({mb:.0f} MB)")
    print(f"  dropped: {n_small} too small (<{args.min_body_px:.0f}px), {n_bad} unusable")
    bp = np.array([r["body_px"] for r in rec])
    print(f"  on-screen size (px): median {np.median(bp):.0f}  p10 {np.percentile(bp, 10):.0f}  "
          f"p90 {np.percentile(bp, 90):.0f}")
    for sp in sorted({r["species"] for r in rec}):
        m = [r["body_px"] for r in rec if r["species"] == sp]
        print(f"    {sp:>9s}: {len(m):6d} crops, median {np.median(m):5.0f} px")
    print(f"  videos: {len(set(r['video'] for r in rec))}")
    print(f"\nscp this ONE file to the cluster; nothing else is needed for DINOv3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
