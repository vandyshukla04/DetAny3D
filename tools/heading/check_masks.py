"""Why is a SAM mask missing? Break it down, per species.  [CLUSTER]

    python -m tools.heading.check_masks --crops data/heading/crops.npz

`CropSet.prefetch` reports one number ("masks ready: 61%"), which is not actionable. A mask can be
absent for five completely different reasons, and they demand completely different responses:

  no_video    the video is not in the archive index          -> a coverage gap
  no_seg      the segment has no sam3_masks/ at all          -> SAM was never run there
  no_obj      no obj_<track_id>/ dir for this track          -> ** THE JOIN IS WRONG **
  no_frame    obj_<tid>/ exists but not this frame           -> a frame-stride mismatch
  rejected    the mask exists but its centroid is NOT on     -> ** THE JOIN IS WRONG **
              this animal (it belongs to a neighbour)

The last two are the dangerous ones. `no_obj` and `rejected` both mean `obj_<N>` is not track `N`,
and if we had "fixed" that by dropping the check, we would have been profiling the NEIGHBOURING
zebra -- the exact bug the masks were introduced to eliminate -- while the numbers looked fine.

The per-species split matters just as much: zebra is the herd species and the only one that needs
the masks. A 61% overall hit rate is fine if the misses are giraffes and fatal if they are zebras.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from tools.heading.masks import build_index, check_join


def classify(d, i) -> tuple[str, float]:
    from PIL import Image

    video = str(d["video"][i])
    seg = str(d["seg_name"][i])
    tid = str(d["track"][i]).split("::")[-1]
    stem = Path(str(d["image_name"][i])).stem

    vdir = build_index().get(video)
    if vdir is None:
        return "no_video", 0.0
    sdir = vdir / seg / "sam3_masks" / "masks"
    if not sdir.is_dir():
        return "no_seg", 0.0
    odir = sdir / f"obj_{tid}"
    if not odir.is_dir():
        return "no_obj", 0.0
    p = odir / f"{stem}.png"
    if not p.is_file():
        return "no_frame", 0.0

    m = np.asarray(Image.open(p)) > 127
    ok, off = check_join(m, d["box2d"][i])
    return ("ok" if ok else "rejected"), off


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1500, help="sample this many crops")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from concurrent.futures import ThreadPoolExecutor

    d = np.load(args.crops, allow_pickle=True)
    idx = build_index()
    print(f"{len(d['jpeg'])} crops | {len(idx)} videos in the archive\n")

    rng = np.random.default_rng(args.seed)
    sample = rng.permutation(len(d["jpeg"]))[: args.n]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        res = list(pool.map(lambda i: classify(d, int(i)), sample))

    per = defaultdict(Counter)
    offs = defaultdict(list)
    for i, (why, off) in zip(sample, res):
        sp = str(d["species"][i])
        per[sp][why] += 1
        if why in ("ok", "rejected"):
            offs[why].append(off)

    reasons = ["ok", "rejected", "no_obj", "no_frame", "no_seg", "no_video"]
    print(f"{'species':>9s} {'n':>6s} " + " ".join(f"{r:>9s}" for r in reasons))
    tot = Counter()
    for sp in sorted(per):
        c = per[sp]
        n = sum(c.values())
        tot.update(c)
        print(f"{sp:>9s} {n:6d} " +
              " ".join(f"{100*c[r]/n:8.0f}%" for r in reasons))
    N = sum(tot.values())
    print(f"{'ALL':>9s} {N:6d} " + " ".join(f"{100*tot[r]/N:8.0f}%" for r in reasons))

    print()
    if offs["ok"]:
        o = np.array(offs["ok"])
        print(f"  accepted: centroid offset median {np.median(o):.2f}  p90 "
              f"{np.percentile(o, 90):.2f}   (1.0 = the box edge)")
    if offs["rejected"]:
        o = np.array(offs["rejected"])
        print(f"  REJECTED: centroid offset median {np.median(o):.2f}  p90 "
              f"{np.percentile(o, 90):.2f}   (limit 1.6)")

    print("\n=== what to do ===")
    if tot["no_obj"] or tot["rejected"]:
        print("  no_obj / rejected are the DANGEROUS ones: obj_<N> is not track <N>, so the mask")
        print("  belongs to a DIFFERENT animal. Do NOT relax the check -- that would silently")
        print("  profile the neighbouring zebra, which is the exact bug the masks are here to fix.")
        print("  Fix the JOIN (find the real track<->obj mapping), or drop those crops.")
    if tot["no_frame"]:
        print("  no_frame: SAM ran on a different frame stride than the boxes. Match the strides,")
        print("  or accept the loss -- but check it is not concentrated in one species.")
    if tot["no_seg"] or tot["no_video"]:
        print("  no_seg / no_video: SAM was simply never run there. A coverage gap, not a bug.")
        print("  Fine to fall back to the appearance mask, UNLESS it is mostly zebra -- zebra is")
        print("  the herd species and the only one that actually needs an instance mask.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
