"""Score every re-ID crop with the template -> per-detection heading + DINOv3 feature.  [CLUSTER / GPU]

    python -m tools.heading.predict_headings \
        --crops data/heading/crops.npz --reid-crops data/heading/crops_reid.npz \
        --out-table data/heading/headings_v1v2.npz \
        --out-feats data/heading/dinov3_feats.npz --device cuda --dtype fp16

ONE DINOv3 PASS, TWO OUTPUTS
----------------------------
The template (the mean rump->head profile per species, built once from the WALKING training crops of
`crops.npz` -- nothing trained) is applied to every detection in `crops_reid.npz`. For each crop the same
forward gives:
  * the per-detection PREDICTED HEADING -- geometry proposes the axis (`geo_axis`), appearance disposes
    the sign (`choose`), then `viewpoint_of` reads out the flank / face-rear tag. Emitted keyed by
    (video, seg, track, frame, image_name) so the local re-ID side can join it. The heading is emitted as
    `head_face_id` (0..5) + `pred_az` (world azimuth); the local bridge reconstructs the world vector from
    papersubdata (`track_render._face_dir`), which it has and the cluster does not.
  * the DINOv3 re-ID FEATURE -- the foreground-mean patch descriptor, L2-normalised. Same grid, no second
    forward. This is Phase 2's DINOv3 backbone.

`crops_reid.npz` has `y_face = -1` everywhere (no labels); prediction never reads it. Detections that
abstain (end-on / no template) are emitted with `valid = False` and the geometric-axis end as a fallback.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tools.heading.cropset import CropSet
from tools.heading.descriptors import DEFAULT_MODEL, Config, DenseExtractor, foreground
from tools.heading.split import video_split
from tools.heading.template import Accumulator, choose
from tools.heading.viewpoint import viewpoint_of


def _pool(g: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """Foreground-mean patch descriptor (L2-normalised). g is (gh,gw,D), per-patch L2-normalised."""
    D = g.shape[-1]
    w = fg.reshape(-1).astype(np.float32)
    feat = (g.reshape(-1, D) * w[:, None]).sum(0) / max(float(w.sum()), 1.0)
    n = float(np.linalg.norm(feat))
    return (feat / n).astype(np.float32) if n > 1e-9 else feat.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", type=Path, required=True, help="training crops -> the template")
    ap.add_argument("--reid-crops", type=Path, required=True, help="crops_reid.npz to score")
    ap.add_argument("--out-table", type=Path, required=True)
    ap.add_argument("--out-feats", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--facet", default="token", choices=["token", "key"])
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--fit", type=int, default=1500, help="template fit-set size")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = Config(args.layer, args.facet, args.size, args.bins)

    train = CropSet(args.crops)
    te, _ = video_split(train.species, train.video, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    idx_fit = np.sort(rng.permutation(np.where(~te)[0])[: args.fit])
    train.prefetch(idx_fit)

    reid = CropSet(args.reid_crops)
    d = reid.d
    idx = np.arange(len(reid))
    reid.prefetch(idx)

    keys = ["video", "seg", "seg_name", "track", "frame", "image_name"]
    rec = {k: [] for k in ["head_face_id", "alpha", "margin", "flank", "flank_w",
                           "end", "end_w", "pred_az", "valid"]}
    feats = np.zeros((len(reid), 0), dtype=np.float32)

    with DenseExtractor(args.model, args.device, args.dtype) as ex:
        # ---- template: mean rump->head profile of WALKING training crops. Nothing trained. ----
        acc = Accumulator(cfg)
        for b in range(0, len(idx_fit), args.batch):
            items = train.batch(idx_fit[b: b + args.batch])
            for g, it in zip(ex.grid(np.stack([i.image for i in items]), cfg), items):
                acc.add(g, it)
        tmpl = acc.build()
        print(f"template (nothing trained): {tmpl.n_fitted}")

        # ---- score every re-ID crop; pool the DINOv3 feature from the same grid ----
        feat_dim = None
        print(f"scoring {len(reid)} re-ID crops ...")
        for b in range(0, len(idx), args.batch):
            items = reid.batch(idx[b: b + args.batch])
            G = ex.grid(np.stack([it.image for it in items]), cfg)
            for k, (g, it) in enumerate(zip(G, items)):
                gi = b + k
                fg = foreground(g, it.instance)
                pooled = _pool(g, fg)
                if feat_dim is None:
                    feat_dim = pooled.shape[0]
                    feats = np.zeros((len(reid), feat_dim), dtype=np.float32)
                feats[gi] = pooled

                if it.species in tmpl.templates:
                    s = tmpl.score_faces(g, fg, it.face_uv, it.face_ids, it.species)
                    slot, margin = choose(s, it.face_ids, axis=int(it.geo_axis))
                else:
                    slot, margin = -1, 0.0
                valid = slot >= 0
                if not valid:
                    slot = int(it.geo_axis)                # fallback: the geometric-axis end
                head_id = int(it.face_ids[slot])
                alpha = float(d["face_alpha"][gi, slot])
                vp = viewpoint_of(alpha)
                rec["head_face_id"].append(head_id)
                rec["alpha"].append(alpha)
                rec["margin"].append(float(margin))
                rec["flank"].append(vp["flank"])
                rec["flank_w"].append(float(vp["flank_strength"]))
                rec["end"].append(vp["end"])
                rec["end_w"].append(float(vp["end_strength"]))
                rec["pred_az"].append(float(d["face_az"][gi, slot]))
                rec["valid"].append(bool(valid))
            print(f"    {min(b+args.batch, len(idx))}/{len(idx)}", end="\r", flush=True)
        print(" " * 40, end="\r")

    # ---- write the per-detection heading table + the DINOv3 features (row-aligned) ----
    table = {k: np.asarray(d[k]) for k in keys}
    table.update({k: np.asarray(v) for k, v in rec.items()})
    args.out_table.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_table, **table)
    np.savez_compressed(args.out_feats, feats=feats, **{k: np.asarray(d[k]) for k in keys})

    nv = int(np.sum(rec["valid"]))
    print(f"\nwrote {len(reid)} detections -> {args.out_table}  ({100*nv/len(reid):.1f}% valid)")
    print(f"wrote DINOv3 features {feats.shape} -> {args.out_feats}")
    for v in np.unique(table["video"]):
        m = table["video"] == v
        print(f"    {v}: {int(m.sum())} detections, "
              f"flank L/R "
              f"{int(np.sum((table['flank'] == 'LEFT') & m))}/{int(np.sum((table['flank'] == 'RIGHT') & m))}")
    print("scp both back to LOCAL; join to the re-ID tracks (reid_v1v2_data.py).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
