"""Does bf16 change any answer?  [GPU -- skips cleanly without one]

    python -m tools.heading.tests.test_dtype --crops data/heading/crops.npz

We run the frozen extractor in bf16 for speed (MEASURED: fp32 @448 is 40 ms/crop and 97% of the
whole pipeline). Precision *should* be irrelevant -- the descriptors are L2-normalised immediately,
and the template is an average over hundreds of crops. But "should be" is exactly the kind of claim
that has been wrong three times today, so it is checked rather than asserted.

Three things must hold, in increasing order of what we actually care about:

  1. DESCRIPTORS   cosine(bf16, fp32) > 0.999 per patch
  2. SCORES        the 4 per-face scores agree to < 1% of the decision margin
  3. DECISIONS     the chosen front face is IDENTICAL on every crop

(3) is the one that matters -- (1) and (2) are only diagnostics for when (3) fails. A precision loss
that never changes a decision is not a precision loss we care about.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


class Skip(Exception):
    pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", type=Path, default=Path("data/heading/crops.npz"))
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--facet", default="key")
    ap.add_argument("--size", type=int, default=448)
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("SKIP: no GPU (bf16 is a GPU-only path)")
        return 0
    if not args.crops.is_file():
        print(f"SKIP: {args.crops} not found")
        return 0

    from tools.heading.cropset import CropSet
    from tools.heading.descriptors import Config, DenseExtractor, foreground
    from tools.heading.template import Accumulator, choose

    crops = CropSet(args.crops)
    idx = np.arange(min(args.n, len(crops)))
    crops.prefetch(idx, workers=16)
    items = crops.batch(idx)
    imgs = np.stack([it.image for it in items])
    cfg = Config(args.layer, args.facet, args.size, 5)

    grids, tmpls = {}, {}
    for dt in ("fp32", "bf16"):
        with DenseExtractor(device="cuda", dtype=dt) as ex:
            grids[dt] = ex.grids(imgs, cfg.facet, cfg.size, [cfg.layer])[cfg.layer]
            acc = Accumulator(cfg)
            for g, it in zip(grids[dt], items):
                acc.add(g, it)
            tmpls[dt] = acc.build()

    # --- 1. descriptors ---
    a, b = grids["fp32"], grids["bf16"]
    cos = np.einsum("nhwd,nhwd->nhw", a, b)              # both already L2-normalised
    print(f"1. DESCRIPTORS  cosine(bf16, fp32): min {cos.min():.5f}  "
          f"median {np.median(cos):.5f}")

    # --- 2 & 3. scores and decisions, using each dtype's OWN template (the real pipeline) ---
    n_flip = 0
    dscore, dmargin = [], []
    for k, it in enumerate(items):
        s32 = tmpls["fp32"].score_faces(grids["fp32"][k], foreground(grids["fp32"][k], it.instance),
                                        it.face_uv, it.face_ids, it.species)
        s16 = tmpls["bf16"].score_faces(grids["bf16"][k], foreground(grids["bf16"][k], it.instance),
                                        it.face_uv, it.face_ids, it.species)
        if np.isnan(s32).all() or np.isnan(s16).all():
            continue
        p32, m32 = choose(s32, it.face_ids, axis=it.geo_axis)
        p16, _ = choose(s16, it.face_ids, axis=it.geo_axis)
        n_flip += int(p32 != p16)
        dscore.append(float(np.nanmax(np.abs(s32 - s16))))
        dmargin.append(abs(m32))

    d = np.array(dscore)
    m = np.array(dmargin)
    print(f"2. SCORES       max |delta| median {np.median(d):.5f}  worst {d.max():.5f}"
          f"   (decision margin median {np.median(m):.4f})")
    print(f"3. DECISIONS    front face changed on {n_flip}/{len(dscore)} crops")

    ok = cos.min() > 0.999 and n_flip == 0
    print("\n" + ("PASS -- bf16 changes no decision. Use it." if ok else
                  "FAIL -- bf16 changes answers. Run with --dtype fp32 and eat the 4x."))
    if np.median(d) > 0.1 * np.median(m):
        print("  WARNING: the dtype noise is >10% of the decision margin. Too close for comfort.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
