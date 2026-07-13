"""Frozen-DINOv3 features for the labelled crops.  [GPU / cluster]

    python -m tools.heading.extract_features \
        --crops data/heading/crops.npz \
        --out   data/heading/features.npz --device cuda

Takes the crops produced LOCALLY by `extract_crops.py` (the human annotations and frames
exist only on the local disk; DINOv3 exists only on the cluster -- so we ship the crops,
not the dataset) and emits one feature vector per crop, plus the heading target.

DINOv3 is FROZEN: no gradients, no fine-tuning. Only the small head (train_head.py) is
trained. That is what makes 11k labels enough.

Features = [CLS ; mean-pooled patch tokens]. CLS carries the crop-level pose/orientation;
the pooled patch tokens add dense context. Both come free from one forward pass.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    d = np.load(args.crops, allow_pickle=True)
    jpegs = d["jpeg"]
    print(f"{len(jpegs)} crops; loading {args.model} (FROZEN) on {args.device}")

    proc = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(args.device).eval()

    feats: list[np.ndarray] = []
    for i in range(0, len(jpegs), args.batch):
        imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in jpegs[i: i + args.batch]]
        inputs = proc(images=imgs, return_tensors="pt").to(args.device)
        with torch.no_grad():
            out = model(**inputs).last_hidden_state       # (B, 1 + registers + patches, D)
        cls = out[:, 0]
        patch = out[:, 1:].mean(dim=1)
        feats.append(torch.cat([cls, patch], dim=-1).float().cpu().numpy())
        if i % (args.batch * 20) == 0:
            print(f"  {i}/{len(jpegs)}", flush=True)

    X = np.concatenate(feats).astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, X=X, Y=d["Y"], species=d["species"], track=d["track"],
        frame=d["frame"], body_px=d["body_px"],
    )
    print(f"\nwrote {X.shape[0]} x {X.shape[1]} features -> {args.out}")
    print("next:  python -m tools.heading.train_head --features", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
