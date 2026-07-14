"""Count WildBox, exactly. Writes `dataset.json`, which the report then quotes.  [LOCAL]

    python -m tools.heading.dataset_stats --root /mnt/d/3DBOX/papersubdata \
        --out tools/heading/dataset.json

The cluster does not have papersubdata, so the report cannot recount it at write time. This script
runs where the data is, and its output is COMMITTED -- so `final_report.py` quotes a measured
artefact rather than a number somebody remembered.

"animal images" = one animal in one frame = a (track, frame) pair. It is the quantity the method
actually consumes, and it is ~3.4x the frame count because the frames contain herds.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

SPECIES = {"elep": "elephant", "rhin": "rhino", "zebr": "zebra", "gira": "giraffe"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("/mnt/d/3DBOX/papersubdata"))
    ap.add_argument("--out", type=Path, default=Path("tools/heading/dataset.json"))
    args = ap.parse_args()

    per = defaultdict(lambda: defaultdict(int))
    vids = defaultdict(set)

    for g in sorted(args.root.iterdir()):
        if not g.is_dir() or g.name[:4] not in SPECIES:
            continue
        sp = SPECIES[g.name[:4]]
        for ts in sorted(g.rglob("seg*/tracking_summary.json")):
            seg = ts.parent
            vids[sp].add(seg.parent.name)
            per[sp]["segments"] += 1
            per[sp]["frames"] += len(list(seg.glob("frame_*.jpg")))
            try:
                d = json.loads(ts.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            tk = d.get("tracks", {})
            per[sp]["tracks"] += len(tk)
            per[sp]["animal_images"] += sum(len(t["frames"]) for t in tk.values())

    out = {sp: {"videos": len(vids[sp]), **dict(d)} for sp, d in per.items()}
    tot = defaultdict(int)
    for d in out.values():
        for k, v in d.items():
            tot[k] += v
    out["TOTAL"] = dict(tot)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))

    print(f"{'species':>9s} {'videos':>7s} {'segments':>9s} {'frames':>8s} {'tracks':>7s} "
          f"{'animal images':>14s}")
    for sp in sorted(k for k in out if k != "TOTAL"):
        d = out[sp]
        print(f"{sp:>9s} {d['videos']:7d} {d['segments']:9d} {d['frames']:8d} {d['tracks']:7d} "
              f"{d['animal_images']:14d}")
    t = out["TOTAL"]
    print(f"{'TOTAL':>9s} {t['videos']:7d} {t['segments']:9d} {t['frames']:8d} {t['tracks']:7d} "
          f"{t['animal_images']:14d}")
    print(f"\n  {t['animal_images']/t['frames']:.1f} animals per frame -- the frames contain HERDS, "
          f"which is why an instance mask is necessary rather than a nicety.")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
