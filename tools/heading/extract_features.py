"""Frozen-DINOv3 features for the motion-labelled crops.  [GPU / cluster]

    python -m tools.heading.extract_features \
        --crops data/heading/crops.npz \
        --out   data/heading/features.npz \
        --patches data/heading/patches.npz --device cuda

Takes the crops produced LOCALLY by `extract_crops.py` (papersubdata lives on the local disk;
DINOv3 and its GPU live only on the cluster -- so we ship the crops, not the dataset).

DINOv3 is FROZEN: no gradients, no fine-tuning, ever.

TWO OUTPUTS -- AND THE SECOND ONE IS THE POINT
----------------------------------------------
  features.npz : [CLS ; mean-pooled patches]  -- one vector per crop. Feeds the learned
                 baseline (train_head.py).
  patches.npz  : the FULL PATCH GRID (gh x gw x D, fp16) for a balanced subset. Feeds the
                 part probe (parts.py).

We used to mean-pool the patch tokens and throw the grid away. But a mean-pool destroys
exactly the thing we now need: WHERE on the animal each feature lives. DINOv3's headline
property is dense semantic part correspondence -- the head of one zebra matches the head of
another -- and that signal only exists per-patch. Keeping the grid is the whole experiment.

Register tokens: DINOv3 puts [CLS, registers..., patches] in `last_hidden_state`, and the
register count varies by checkpoint. So we take the LAST gh*gw tokens rather than assuming an
offset -- guessing it silently shifts the grid by one row and quietly ruins the geometry.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def balanced_subset(species: np.ndarray, video: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Up to `n` indices, spread evenly over species and then over videos within a species.

    A head-on sample of the first `n` crops would be ~all rhino (they walk the most), and the
    probe would tell us nothing about zebras.
    """
    rng = np.random.default_rng(seed)
    sps = sorted(set(species.tolist()))
    per_sp = max(1, n // len(sps))
    picks: list[int] = []
    for s in sps:
        idx = np.where(species == s)[0]
        vids = sorted(set(video[idx].tolist()))
        per_v = max(1, per_sp // len(vids))
        got: list[int] = []
        for v in vids:
            cand = idx[video[idx] == v]
            got.extend(rng.permutation(cand)[:per_v].tolist())
        if len(got) < per_sp:                      # top up from whatever is left
            rest = np.setdiff1d(idx, np.array(got, dtype=int))
            got.extend(rng.permutation(rest)[: per_sp - len(got)].tolist())
        picks.extend(got[:per_sp])
    return np.sort(np.array(picks, dtype=int))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--patches", type=Path, default=None,
                    help="also write the full patch grid for a balanced subset (the part probe)")
    ap.add_argument("--patch-max", type=int, default=6000,
                    help="how many crops keep their full grid (~0.4 MB each in fp16)")
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
    ps = int(model.config.patch_size)

    want_patches = set()
    if args.patches:
        sub = balanced_subset(d["species"], d["video"], args.patch_max)
        want_patches = set(sub.tolist())
        print(f"keeping the full patch grid for {len(sub)} crops "
              f"({dict(zip(*np.unique(d['species'][sub], return_counts=True)))})")

    feats: list[np.ndarray] = []
    grids: list[np.ndarray] = []
    grid_idx: list[int] = []
    gh = gw = None

    for i in range(0, len(jpegs), args.batch):
        imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in jpegs[i: i + args.batch]]
        inputs = proc(images=imgs, return_tensors="pt").to(args.device)
        with torch.no_grad():
            out = model(**inputs).last_hidden_state          # (B, 1 + registers + gh*gw, D)

        H, W = inputs["pixel_values"].shape[-2:]
        gh, gw = H // ps, W // ps
        n = gh * gw
        patch = out[:, -n:]                                  # the LAST n tokens are the grid
        feats.append(torch.cat([out[:, 0], patch.mean(1)], dim=-1).float().cpu().numpy())

        if want_patches:
            for b in range(len(imgs)):
                j = i + b
                if j in want_patches:
                    grids.append(patch[b].reshape(gh, gw, -1).half().cpu().numpy())
                    grid_idx.append(j)

        if i % (args.batch * 20) == 0:
            print(f"  {i}/{len(jpegs)}", flush=True)

    X = np.concatenate(feats).astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, X=X,
        Y=d["Y"], y_face=d["y_face"], face_uv=d["face_uv"], face_alpha=d["face_alpha"],
        species=d["species"], video=d["video"], seg=d["seg"], track=d["track"],
        frame=d["frame"], body_px=d["body_px"],
    )
    print(f"\nwrote {X.shape[0]} x {X.shape[1]} pooled features -> {args.out}")

    if args.patches and grids:
        gi = np.array(grid_idx, dtype=np.int32)
        args.patches.parent.mkdir(parents=True, exist_ok=True)
        np.savez(                                            # NOT compressed: fp16, already dense
            args.patches,
            P=np.stack(grids),                               # (n, gh, gw, D) fp16
            idx=gi,
            grid=np.array([gh, gw], dtype=np.int32),
            y_face=d["y_face"][gi], face_uv=d["face_uv"][gi], face_alpha=d["face_alpha"][gi],
            species=d["species"][gi], video=d["video"][gi], track=d["track"][gi],
            body_px=d["body_px"][gi],
        )
        mb = args.patches.stat().st_size / 1e6
        print(f"wrote {len(grids)} patch grids ({gh}x{gw}) -> {args.patches}  ({mb:.0f} MB)")

    print("\nnext:  parts.py (the probe)  and  train_head.py (the baseline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
