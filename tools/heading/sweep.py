"""X1: how should we use DINOv3?  Sweep layer x facet x resolution, on the ONE task that matters.

    python -m tools.heading.sweep --crops data/heading/crops.npz \
        --out data/heading/desc --device cuda

TWO NUMBERS, NOT ONE -- and the gap between them is the point
-------------------------------------------------------------
  2-WAY  head vs tail ALONG THE TRUE AXIS       (chance 50%)  <- isolates the APPEARANCE cue
  4-WAY  which of the 4 horizontal faces is front (chance 25%) <- end-to-end, comparable to the
                                                                 79.8% supervised MLP baseline

We have never separated these, and they demand opposite fixes. The 4-way number folds together
"is the heading along the animal at all" (an AXIS question, geometry's job) with "which end is the
head" (a SIGN question, appearance's job). The box's longest-horizontal-axis is itself wrong 13%
of the time (zebra: 26%), so a perfect appearance cue would still cap the 4-way at ~87%.

If 2-way is high and 4-way lags, the appearance cue works and the AXIS is the bottleneck.
If 2-way is at chance, DINOv3 does not carry head/tail here and no decoder saves it.

Scoring is free: on WALKING crops, motion already told us which face is the head.
Held out BY VIDEO (shared with train_head.py via split.py, so every number is comparable).
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np

from tools.heading.descriptors import DEFAULT_MODEL, Config, DenseExtractor, foreground
from tools.heading.split import video_split
from tools.heading.template import AxisTemplate, opposite_slot


def evaluate(ex: DenseExtractor, crops, idx, tmpl: AxisTemplate, batch: int = 32):
    """-> per-species dict of (n, correct_4way, correct_2way)."""
    from PIL import Image

    jpeg, face_uv, y_face = crops["jpeg"], crops["face_uv"], crops["y_face"]
    face_ids, species = crops["face_ids"], crops["species"]
    stat: dict[str, np.ndarray] = {}

    for b in range(0, len(idx), batch):
        ii = [int(k) for k in idx[b: b + batch]]
        imgs = np.stack([np.asarray(Image.open(io.BytesIO(jpeg[i])).convert("RGB")) for i in ii])
        G = ex.grid(imgs, cfg=tmpl.cfg)

        for g, i in zip(G, ii):
            sp = str(species[i])
            if sp not in tmpl.templates:
                continue
            fg = foreground(g)
            s = tmpl.score_faces(g, fg, face_uv[i], face_ids[i], sp)
            if np.isnan(s).all():
                continue
            h = int(y_face[i])
            t = opposite_slot(face_ids[i], h)

            # 4-way: the best of all four hypotheses.
            pred4 = int(np.nanargmax(s))
            # 2-way: restricted to the TRUE axis -- head vs tail only. This is the appearance cue
            # with the axis question removed.
            ok2 = int(np.nan_to_num(s[h], nan=-np.inf) >= np.nan_to_num(s[t], nan=-np.inf))

            r = stat.setdefault(sp, np.zeros(3))
            r += (1, int(pred4 == h), ok2)
    return stat


def _report(stat, label, mlp=None):
    tot = sum(r[0] for r in stat.values())
    if not tot:
        return None
    a4 = sum(r[1] for r in stat.values()) / tot
    a2 = sum(r[2] for r in stat.values()) / tot
    per = "  ".join(f"{s[:5]}:{100*r[2]/r[0]:4.0f}/{100*r[1]/r[0]:<4.0f}" for s, r in sorted(stat.items()))
    extra = f"   [MLP 4-way {mlp}]" if mlp else ""
    print(f"{label:<22s} {100*a2:6.1f}%  {100*a4:6.1f}%   {per}{extra}")
    return a2, a4


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--subset", type=int, default=1500, help="crops per split, per config")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    crops = np.load(args.crops, allow_pickle=True)
    sp, vid = crops["species"], crops["video"]
    te, test_v = video_split(sp, vid, seed=args.seed)
    print(f"{len(sp)} crops | held out {len(test_v)} videos ({te.sum()} crops)")

    rng = np.random.default_rng(args.seed)
    idx_fit = np.sort(rng.permutation(np.where(~te)[0])[: args.subset])
    idx_te = np.sort(rng.permutation(np.where(te)[0])[: args.subset])
    print(f"fit the template on {len(idx_fit)} crops, score on {len(idx_te)} held-out\n")

    with DenseExtractor(args.model, args.device) as ex:
        print(f"{ex.n_layers} layers, {ex.dim}-d, patch {ex.patch}\n")

        # Coarse layer sweep first: part semantics usually peak mid-network, not at the end.
        L = ex.n_layers
        configs = [Config(l, f, s, args.bins)
                   for l in sorted({L // 2, (2 * L) // 3, (3 * L) // 4, L})
                   for f in ("key", "token")
                   for s in (448, 224)]

        print(f"{'config':<22s} {'2-WAY':>6s}  {'4-WAY':>6s}   per-species (2way/4way)")
        print(f"{'chance':<22s} {50.0:6.1f}%  {25.0:6.1f}%")
        rows = []
        for cfg in configs:
            tmpl = AxisTemplate.fit(ex, crops, idx_fit, cfg, args.batch)
            if not tmpl.templates:
                continue
            stat = evaluate(ex, crops, idx_te, tmpl, args.batch)
            got = _report(stat, str(cfg))
            if got:
                rows.append((str(cfg), *got))

    if not rows:
        print("nothing scored")
        return 1

    rows.sort(key=lambda r: -r[1])                     # rank by the 2-WAY (the appearance cue)
    best = rows[0]
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "sweep.npz", config=np.array([r[0] for r in rows]),
             two_way=np.array([r[1] for r in rows]), four_way=np.array([r[2] for r in rows]))

    print(f"\n  BEST (by 2-way): {best[0]}   2-way {100*best[1]:.1f}%   4-way {100*best[2]:.1f}%")
    print(f"  supervised MLP baseline (4-way): 79.8%")
    print(f"  retracted k-means strawman     : 16.6%")
    print()
    if best[1] > 0.75:
        print("  => DINOv3 DOES carry head/tail. Take it to the tracklet decoder (X2), where")
        print("     per-frame evidence accumulates across viewpoints and errors decorrelate.")
    elif best[1] > 0.58:
        print("  => a real but weak per-frame cue. This is EXACTLY the regime where track-level")
        print("     accumulation pays: a 65% frame cue becomes a near-certain track answer.")
    else:
        print("  => at chance. DINOv3 does not carry head/tail here; say so and stop.")
    print(f"\n  wrote {args.out}/sweep.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
