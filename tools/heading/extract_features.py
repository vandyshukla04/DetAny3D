"""Extract FROZEN DINOv3 features for every heading sample.  [GPU / cluster]

    python -m tools.heading.extract_features \
        --manifest data/heading/manifest.json \
        --out      data/heading/features.npz \
        --device   cuda

Reads the manifest (see build_manifest.py), crops each animal, runs DINOv3 with NO
gradients, and stores one feature vector per sample alongside its heading target.

WHY SQUARE-PAD RATHER THAN PLAIN RESIZE  (this is a correctness issue, not cosmetics)
------------------------------------------------------------------------------------
The target is an ANGLE in image space. A non-uniform resize (stretching a wide crop into
a square) is not a similarity transform: it *changes angles*. Training on stretched crops
against unstretched angles would teach the head a systematic lie. So we pad the crop to a
square first and only then resize -- which is a similarity transform, and leaves the
angle exactly as measured.

DINOv3 is frozen: we never backprop into it. Only the small head (train_head.py) learns.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"


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
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--size", type=int, default=224, help="square crop side fed to DINOv3")
    ap.add_argument("--pad", type=float, default=0.15)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    samples = json.loads(args.manifest.read_text())
    print(f"{len(samples)} samples; loading {args.model} (frozen) on {args.device}")
    proc = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(args.device).eval()

    feats: list[np.ndarray] = []
    keep: list[dict] = []
    batch_imgs: list[Image.Image] = []
    batch_meta: list[dict] = []

    def flush():
        if not batch_imgs:
            return
        inputs = proc(images=batch_imgs, return_tensors="pt").to(args.device)
        with torch.no_grad():
            out = model(**inputs).last_hidden_state          # (B, 1+reg+N, D)
        cls = out[:, 0]                                      # global descriptor
        patches = out[:, 1:].mean(dim=1)                     # mean-pooled dense context
        f = torch.cat([cls, patches], dim=-1).float().cpu().numpy()
        feats.append(f)
        keep.extend(batch_meta)
        batch_imgs.clear()
        batch_meta.clear()

    cache: dict[str, np.ndarray] = {}
    for n, s in enumerate(samples):
        fp = str(Path(s["segment"]) / s["image_name"])
        img = cache.get(fp)
        if img is None:
            cache.clear()                                    # frames are visited in order
            img = np.asarray(Image.open(fp).convert("RGB"))
            cache[fp] = img
        crop = square_crop(img, s["bbox_2d"], args.pad)
        if crop is None:
            continue
        batch_imgs.append(Image.fromarray(crop).resize((args.size, args.size), Image.BICUBIC))
        batch_meta.append(s)
        if len(batch_imgs) >= args.batch:
            flush()
        if n % 500 == 0:
            print(f"  {n}/{len(samples)}", flush=True)
    flush()

    X = np.concatenate(feats).astype(np.float32)
    ang = np.radians([s["heading_deg"] for s in keep]).astype(np.float32)
    Y = np.stack([np.cos(ang), np.sin(ang)], axis=1)         # unit-circle target

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, X=X, Y=Y,
        species=np.array([s["species"] for s in keep]),
        track=np.array([f'{s["segment"]}::{s["track_id"]}' for s in keep]),
        frame=np.array([s["frame_index"] for s in keep]),
        body_px=np.array([s["body_px"] for s in keep], dtype=np.float32),
    )
    print(f"\nwrote {X.shape[0]} x {X.shape[1]} features -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
