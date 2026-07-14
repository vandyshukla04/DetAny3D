"""X1: how should we use DINOv3?  Sweep layer x facet x resolution, on the ONE task that matters.

    python -m tools.heading.sweep --crops data/heading/crops.npz \
        --out data/heading/desc --device cuda --all-layers

WHAT IS REPORTED, AND WHY IT IS FOUR NUMBERS AND NOT ONE
--------------------------------------------------------
The heading question factorises into an AXIS ("which way does the animal lie?") and a SIGN
("which end is the head?"). Geometry is good at the first; DINOv3 is good at the second. Reporting
only the end-to-end number hides which half is failing -- and they need opposite fixes.

  2-WAY   head vs tail, ALONG THE TRUE AXIS         chance 50%   <- the appearance cue, isolated
  APP-4   appearance freely picks among all 4 faces chance 25%   <- lets appearance choose the axis
  GEO+APP geometry picks the axis, appearance the sign           <- geometry proposes, appearance disposes
  PRIOR   all 4, with a soft bonus on the geometric axis         <- the principled middle ground

Measured before this change: 2-WAY 83.7% but APP-4 only 39.4% -- appearance was overriding
geometry at the one thing it is BAD at. The box's longest horizontal axis is right 87% of the time
(zebra 74%), so GEO+APP should land near 0.87 x 0.837 ~= 73%, and PRIOR should beat both.

Scoring is free: on WALKING crops, motion already told us which face is the head.
Held out BY VIDEO, using the same split as train_head.py -- so every number here is comparable to
the 79.8% supervised MLP baseline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tools.heading.cropset import CropSet
from tools.heading.descriptors import DEFAULT_MODEL, Config, DenseExtractor, foreground
from tools.heading.split import video_split
from tools.heading.template import Accumulator, AxisTemplate, opposite_slot

MODES = ("2way", "app4", "geo", "prior")


def score_one(tmpl: AxisTemplate, g, it, stat, prior: float) -> None:
    """Score one crop under one template, into `stat` (all four modes)."""
    if it.species not in tmpl.templates:
        return
    fg = foreground(g, it.instance)
    s = tmpl.score_faces(g, fg, it.face_uv, it.face_ids, it.species)
    if np.isnan(s).all():
        return

    h, t = it.y_face, opposite_slot(it.face_ids, it.y_face)
    sv = np.where(np.isnan(s), -np.inf, s)

    r = stat.setdefault(it.species, {m: np.zeros(2) for m in MODES})
    r["2way"] += (int(sv[h] >= sv[t]), 1)            # appearance cue, axis question REMOVED
    r["app4"] += (int(np.argmax(sv) == h), 1)        # appearance also picks the axis (the 39%)
    if it.geo_axis >= 0:
        p_geo, _ = tmpl.predict_face(g, fg, it.face_uv, it.face_ids, it.species, axis=it.geo_axis)
        p_pri, _ = tmpl.predict_face(g, fg, it.face_uv, it.face_ids, it.species,
                                     axis=it.geo_axis, axis_prior=prior)
        r["geo"] += (int(p_geo == h), 1)
        r["prior"] += (int(p_pri == h), 1)


def _row(label, stat):
    out = {}
    for m in MODES:
        c = sum(r[m][0] for r in stat.values())
        n = sum(r[m][1] for r in stat.values())
        out[m] = c / n if n else float("nan")
    per = "  ".join(f"{s[:5]}:{100*r['2way'][0]/max(r['2way'][1],1):3.0f}"
                    f"/{100*r['prior'][0]/max(r['prior'][1],1):<3.0f}"
                    for s, r in sorted(stat.items()))
    print(f"{label:<20s} {100*out['2way']:6.1f} {100*out['app4']:6.1f} {100*out['geo']:6.1f} "
          f"{100*out['prior']:6.1f}   {per}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--subset", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=64, help="A40 has headroom; raise this")
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--prior", type=float, default=0.05, help="bonus on the geometric axis")
    ap.add_argument("--all-layers", action="store_true",
                    help="every layer, not a 4-point sample (the A40 can afford it)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    crops = CropSet(args.crops)
    te, test_v = video_split(crops.species, crops.video, seed=args.seed)
    print(f"{len(crops)} crops | held out {len(test_v)} videos ({te.sum()} crops) | "
          f"SAM masks: {'available' if crops.can_mask else 'NOT FOUND -- zebra stays at chance'}")

    rng = np.random.default_rng(args.seed)
    idx_fit = np.sort(rng.permutation(np.where(~te)[0])[: args.subset])
    idx_te = np.sort(rng.permutation(np.where(te)[0])[: args.subset])
    print(f"fit on {len(idx_fit)}, score on {len(idx_te)} held-out\n")

    rows = []
    with DenseExtractor(args.model, args.device) as ex:
        L = ex.n_layers
        layers = list(range(1, L + 1)) if args.all_layers else sorted(
            {L // 2, (2 * L) // 3, (3 * L) // 4, L})
        facets = ("key", "token")
        sizes = (448, 224)
        print(f"{ex.n_layers} layers, {ex.dim}-d, patch {ex.patch}")
        print(f"{len(layers)*len(facets)*len(sizes)} configs, but only "
              f"{len(facets)*len(sizes)*2} model passes: `output_hidden_states` already returns "
              f"EVERY layer from one forward, so a layer sweep is free.\n")

        print(f"{'config':<20s} {'2WAY':>6s} {'APP4':>6s} {'GEO':>6s} {'PRIOR':>6s}   "
              f"per-species (2way/prior)")
        print(f"{'chance':<20s} {50.0:6.1f} {25.0:6.1f} {50.0:6.1f} {25.0:6.1f}")

        for facet in facets:
            for size in sizes:
                # ---- ONE pass over the fit set: templates for EVERY layer at once ----
                accs = {l: Accumulator(Config(l, facet, size, args.bins)) for l in layers}
                for b in range(0, len(idx_fit), args.batch):
                    items = crops.batch(idx_fit[b: b + args.batch])
                    G = ex.grids(np.stack([it.image for it in items]), facet, size, layers)
                    for l in layers:
                        for g, it in zip(G[l], items):
                            accs[l].add(g, it)
                tmpls = {l: a.build() for l, a in accs.items()}

                # ---- ONE pass over the held-out set: score EVERY layer at once ----
                stats = {l: {} for l in layers}
                for b in range(0, len(idx_te), args.batch):
                    items = crops.batch(idx_te[b: b + args.batch])
                    G = ex.grids(np.stack([it.image for it in items]), facet, size, layers)
                    for l in layers:
                        if not tmpls[l].templates:
                            continue
                        for g, it in zip(G[l], items):
                            score_one(tmpls[l], g, it, stats[l], args.prior)

                for l in layers:
                    if stats[l]:
                        rows.append((str(tmpls[l].cfg), _row(str(tmpls[l].cfg), stats[l])))

    if not rows:
        print("nothing scored")
        return 1

    rows.sort(key=lambda r: -r[1]["2way"])
    best_cue = rows[0]
    best_e2e = max(rows, key=lambda r: max(r[1]["geo"], r[1]["prior"], r[1]["app4"]))

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "sweep.npz", config=np.array([r[0] for r in rows]),
             **{m: np.array([r[1][m] for r in rows]) for m in MODES})

    print(f"\n  BEST APPEARANCE CUE (2-way): {best_cue[0]}  {100*best_cue[1]['2way']:.1f}%  "
          f"(chance 50%)")
    print(f"  BEST END-TO-END           : {best_e2e[0]}  "
          f"app4 {100*best_e2e[1]['app4']:.1f}  geo {100*best_e2e[1]['geo']:.1f}  "
          f"prior {100*best_e2e[1]['prior']:.1f}   (chance 25%)")
    print(f"  supervised MLP baseline   : 79.8%")
    print(f"\n  wrote {args.out}/sweep.npz")
    print("  Next: tracklet.py -- a per-frame cue at this level becomes near-certain once")
    print("  evidence accumulates across a track in WORLD space (errors decorrelate).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
