"""Cut the labelled animal crops.  [LOCAL / CPU -- run where the data actually is]

    python -m tools.heading.extract_crops \
        --manifest data/heading/manifest.json --out data/heading/crops.npz

WHY THIS IS A SEPARATE STEP
---------------------------
The human face annotations (and the frames they refer to) live ONLY on the local disk --
there are no `semantic_faces/` annotations anywhere on the cluster. But DINOv3, which is
the only part that needs a GPU, lives ONLY on the cluster. Rather than sync a multi-TB
dataset, we ship the one thing the GPU actually needs: the crops.

Crops are stored JPEG-ENCODED inside a single .npz (~200MB for 11k crops, versus ~1.7GB
raw), so it is one `scp` and no directory tree to keep in sync.

SQUARE-PAD, DO NOT STRETCH  (correctness, not cosmetics)
--------------------------------------------------------
The target is an ANGLE in image space. Squashing a wide crop into a square is not a
similarity transform -- it *changes angles* -- so training on stretched crops against
unstretched angles would teach the head a systematic lie. We pad to a square first, which
is angle-preserving, and only then resize.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np


def square_crop(img: np.ndarray, bbox: list[float], pad: float) -> np.ndarray | None:
    """Crop the animal and pad to a SQUARE (angle-preserving; see module docstring)."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + pad)
    if side < 8:
        return None

    sx1, sy1 = int(round(cx - side / 2)), int(round(cy - side / 2))
    sx2, sy2 = int(round(cx + side / 2)), int(round(cy + side / 2))

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
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--pad", type=float, default=0.15)
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()

    from PIL import Image

    samples = json.loads(args.manifest.read_text())
    print(f"{len(samples)} samples -> cropping at {args.size}px")

    jpegs: list[bytes] = []
    keep: list[dict] = []
    last_path, last_img = None, None

    for n, s in enumerate(samples):
        fp = str(Path(s["segment"]) / s["image_name"])
        if fp != last_path:                              # frames come in order; cache one
            try:
                last_img = np.asarray(Image.open(fp).convert("RGB"))
            except OSError as e:
                print(f"  skip unreadable {fp}: {e}")
                last_path = None
                continue
            last_path = fp

        crop = square_crop(last_img, s["bbox_2d"], args.pad)
        if crop is None:
            continue

        buf = io.BytesIO()
        Image.fromarray(crop).resize((args.size, args.size), Image.BICUBIC).save(
            buf, format="JPEG", quality=args.quality
        )
        jpegs.append(buf.getvalue())
        keep.append(s)
        if n % 1000 == 0:
            print(f"  {n}/{len(samples)}", flush=True)

    if not keep:
        print("no crops produced")
        return 1

    ang = np.radians([s["heading_deg"] for s in keep]).astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        jpeg=np.array(jpegs, dtype=object),
        Y=np.stack([np.cos(ang), np.sin(ang)], axis=1),          # unit-circle target
        species=np.array([s["species"] for s in keep]),
        track=np.array([f'{s["segment"]}::{s["track_id"]}' for s in keep]),
        frame=np.array([s["frame_index"] for s in keep]),
        body_px=np.array([s["body_px"] for s in keep], dtype=np.float32),
    )
    mb = args.out.stat().st_size / 1e6
    print(f"\nwrote {len(keep)} crops -> {args.out}  ({mb:.0f} MB)")
    print(f"scp this one file to the cluster; nothing else is needed for DINOv3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
