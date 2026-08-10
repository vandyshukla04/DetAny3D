"""Appendix D, CORRECTED: choose the DINOv3 configuration on a VALIDATION split of the TEMPLATE
videos -- never on the 20 evaluation videos.  [GPU]

    python -m tools.heading.sweep_val --crops data/heading/crops.npz \
        --out data/heading/REPORT_v3/sweep_val --device cuda --dtype fp32

WHAT WAS WRONG WITH `sweep.py`
------------------------------
`sweep.py:102-108` fits the template on the TEMPLATE videos and then scores every configuration on
`te` -- the 20 EVALUATION videos. The configuration reported in the paper was therefore selected
using the same videos the headline numbers are measured on. That is a selection effect, and it is
the one criticism in review that needs a real experiment rather than a wording change.

WHAT THIS DOES INSTEAD
----------------------
  1. Carve a VALIDATION set out of the template videos (`split.template_val_split`), by VIDEO,
     species-stratified, with the rule fixed in advance.
  2. Build each configuration's species templates from the FIT videos ONLY. The validation videos
     contribute nothing -- otherwise the observations being scored would partly determine the thing
     they are scored against.
  3. Score every configuration on the validation videos.
  4. Select by ONE predefined criterion: highest overall (crop-weighted) 2-way accuracy.
     Ties break to the LOWER resolution, then the LOWER layer.

The evaluation videos never participate. After selection, the winner's final templates are rebuilt
from ALL template videos by `experiments.py --fit 1500`, which is the protocol the method actually
uses; this script does not touch that.

WHY 2-WAY IS THE CRITERION
--------------------------
Layer / facet / resolution is being chosen for the FRONT-vs-BACK appearance decision. 2-way asks
exactly that question with the unsigned axis supplied, so it does not confound the descriptor choice
with the separate geometric-axis module. `app4`, `geo` and `prior` are reported as diagnostics.

Overall 2-way is CROP-weighted (as in `sweep.py`), so a species with many validation crops dominates
it. The macro (per-species mean) is printed alongside so that dependence is visible rather than
hidden -- but the selection uses the crop-weighted number, as specified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.heading.cropset import CropSet
from tools.heading.descriptors import DEFAULT_MODEL, Config, DenseExtractor
from tools.heading.split import split_fingerprint, template_val_split, video_split
from tools.heading.sweep import MODES, score_one
from tools.heading.template import Accumulator

# The eight configurations in Appendix Table 5, grouped by (facet, size) so that one forward pass
# serves every layer in the group -- the hooks capture all requested layers at once.
GROUPS = {
    ("key", 448): [12, 16, 18, 24],
    ("token", 448): [12, 24],
    ("token", 224): [12, 24],
}


def summarise(stat: dict) -> dict:
    """stat = {species: {mode: [correct, n]}} -> overall (crop-weighted) + macro + per-species."""
    out = {}
    for m in MODES:
        c = sum(r[m][0] for r in stat.values())
        n = sum(r[m][1] for r in stat.values())
        out[m] = c / n if n else float("nan")
    per = {s: (r["2way"][0] / r["2way"][1] if r["2way"][1] else float("nan"))
           for s, r in stat.items()}
    out["per_species_2way"] = per
    out["n"] = int(sum(r["2way"][1] for r in stat.values()))
    vals = [v for v in per.values() if np.isfinite(v)]
    out["macro_2way"] = float(np.mean(vals)) if vals else float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp32", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--prior", type=float, default=0.05)
    ap.add_argument("--fit", type=int, default=1500,
                    help="template fit-set size -- matches experiments.py, so the sweep builds "
                         "templates the same way the final method does")
    ap.add_argument("--n-val-videos", type=int, default=10)
    ap.add_argument("--mask-workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    crops = CropSet(args.crops)
    sp, vid = crops.species, crops.video

    # ---- 1. the EVALUATION split (untouched) -- these videos take no part in what follows ----
    te, eval_v = video_split(sp, vid, seed=args.seed)
    tmpl_idx = np.where(~te)[0]
    print(f"{len(crops)} crops | {len(eval_v)} EVALUATION videos ({te.sum()} crops) -- "
          f"EXCLUDED from this script entirely")

    # ---- 2. the FIT / VALIDATION split, inside the template videos ----
    val_local, fit_v, val_v = template_val_split(sp[tmpl_idx], vid[tmpl_idx],
                                                 n_val=args.n_val_videos, seed=args.seed)
    idx_val = tmpl_idx[val_local]
    idx_fit_all = tmpl_idx[~val_local]
    rng = np.random.default_rng(args.seed)
    idx_fit = np.sort(rng.permutation(idx_fit_all)[: args.fit])

    print(f"template videos: {len(fit_v) + len(val_v)}  ({len(tmpl_idx)} crops)")
    print(f"  FIT        {len(fit_v):>3} videos  {len(idx_fit_all):>5} crops  "
          f"(template built from {len(idx_fit)})")
    print(f"  VALIDATION {len(val_v):>3} videos  {len(idx_val):>5} crops\n")
    print(f"{'species':>9} {'fit vid':>8} {'fit crops':>10} {'val vid':>8} {'val crops':>10}")
    split_counts = {}
    for s in sorted(set(sp.tolist())):
        f = idx_fit_all[sp[idx_fit_all] == s]
        v = idx_val[sp[idx_val] == s]
        split_counts[s] = {"fit_videos": len(set(vid[f].tolist())), "fit_crops": int(len(f)),
                           "val_videos": len(set(vid[v].tolist())), "val_crops": int(len(v))}
        print(f"{s:>9} {split_counts[s]['fit_videos']:>8} {len(f):>10} "
              f"{split_counts[s]['val_videos']:>8} {len(v):>10}")
    print(f"\nvalidation videos: {val_v}\n")

    crops.prefetch(np.concatenate([idx_fit, idx_val]), workers=args.mask_workers)

    # ---- 3. one fit pass + one score pass per (facet, size); layers come free from the hooks ----
    rows = []
    with DenseExtractor(args.model, args.device, args.dtype) as ex:
        print(f"{'configuration':<18s} {'2-way':>7s} {'App-4':>7s} {'Geo':>7s} {'Prior':>7s} "
              f"{'macro':>7s}   {'Ele':>5s} {'Gir':>5s} {'Rhi':>5s} {'Zeb':>5s}")
        for (facet, size), layers in GROUPS.items():
            accs = {l: Accumulator(Config(l, facet, size, args.bins)) for l in layers}
            for b in range(0, len(idx_fit), args.batch):
                items = crops.batch(idx_fit[b: b + args.batch])
                G = ex.grids(np.stack([it.image for it in items]), facet, size, layers)
                for l in layers:
                    for g, it in zip(G[l], items):
                        accs[l].add(g, it)
                print(f"    [{facet}/{size}] fit {min(b+args.batch, len(idx_fit))}/{len(idx_fit)}",
                      end="\r", flush=True)
            tmpls = {l: a.build() for l, a in accs.items()}

            stats = {l: {} for l in layers}
            for b in range(0, len(idx_val), args.batch):
                items = crops.batch(idx_val[b: b + args.batch])
                G = ex.grids(np.stack([it.image for it in items]), facet, size, layers)
                for l in layers:
                    if not tmpls[l].templates:
                        continue
                    for g, it in zip(G[l], items):
                        score_one(tmpls[l], g, it, stats[l], args.prior)
                print(f"    [{facet}/{size}] score {min(b+args.batch, len(idx_val))}"
                      f"/{len(idx_val)}", end="\r", flush=True)
            print(" " * 60, end="\r")

            for l in layers:
                if not stats[l]:
                    continue
                r = summarise(stats[l])
                name = f"L{l}/{facet}/{size}"
                r.update(config=name, layer=l, facet=facet, size=size)
                rows.append(r)
                ps = r["per_species_2way"]
                g = lambda s: (f"{100*ps[s]:5.1f}" if s in ps and np.isfinite(ps[s]) else "    -")
                print(f"{name:<18s} {100*r['2way']:7.1f} {100*r['app4']:7.1f} "
                      f"{100*r['geo']:7.1f} {100*r['prior']:7.1f} {100*r['macro_2way']:7.1f}   "
                      f"{g('elephant')} {g('giraffe')} {g('rhino')} {g('zebra')}", flush=True)

    if not rows:
        print("nothing scored")
        return 1

    # ---- 4. THE PREDEFINED SELECTION RULE ----
    # highest crop-weighted validation 2-way; ties -> lower resolution, then lower layer.
    winner = min(rows, key=lambda r: (-r["2way"], r["size"], r["layer"]))
    print(f"\n  SELECTED: {winner['config']}")
    print(f"  criterion: highest overall (crop-weighted) validation 2-way accuracy "
          f"= {100*winner['2way']:.1f}%  (n={winner['n']})")
    print(f"  ties broken to lower resolution, then lower layer. Macro 2-way "
          f"{100*winner['macro_2way']:.1f}% is a diagnostic, not the criterion.")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "sweep_val.json").write_text(json.dumps({
        "selected": winner["config"],
        "criterion": "highest crop-weighted validation 2-way; ties -> lower size, then lower layer",
        "rows": rows,
        "split": {
            "evaluation": split_fingerprint(eval_v, seed=args.seed),
            "fit_videos": fit_v,
            "validation_videos": val_v,
            "counts": split_counts,
            "n_fit_crops": int(len(idx_fit_all)), "n_fit_used": int(len(idx_fit)),
            "n_val_crops": int(len(idx_val)),
        },
        "config": {"dtype": args.dtype, "bins": args.bins, "prior": args.prior,
                   "fit": args.fit, "seed": args.seed},
    }, indent=1))
    print(f"\n  wrote {args.out}/sweep_val.json")
    print("  Next: rebuild the winner's templates from ALL template videos and evaluate once on the "
          "20 evaluation videos (experiments.py). If the winner is L24/token/224, that run already "
          "exists as REPORT_v3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
