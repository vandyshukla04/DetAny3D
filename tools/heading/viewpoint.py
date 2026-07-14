"""THE END GOAL: which part of the animal are we looking at?  -> the re-ID viewpoint tag.

    python -m tools.heading.viewpoint --crops data/heading/crops.npz \
        --out data/heading/view --layer 24 --facet token --size 224

WHY THIS IS THE POINT
---------------------
Wildlife re-ID is SIDE-DEPENDENT: a zebra's left flank carries a different stripe pattern from its
right. Matching a left-flank query against a right-flank gallery entry is not a hard match -- it is
a GUARANTEED MISS. Everything else in this pipeline exists to produce one tag per detection:

    "we are looking at this animal's LEFT flank, 0.85 broadside, slightly from behind"

IT ALL COMES OUT OF ONE ANGLE
-----------------------------
alpha is the animal's heading measured against the VIEWING RAY (papersub.allocentric_basis: r =
horizontalised camera->animal ray, s = up x r). With  h = cos(a).r + sin(a).s  and
left = up x h = cos(a).s - sin(a).r, and the camera lying in the -r direction from the animal:

    dot(left, -r) =  sin(alpha)      ->  sin > 0 : we see its LEFT flank
    dot(h,    -r) = -cos(alpha)      ->  cos < 0 : we see its FACE (it walks toward us)

So:
    flank      = LEFT if sin(alpha) > 0 else RIGHT      strength |sin(alpha)|
    end        = FACE if cos(alpha) < 0 else REAR       strength |cos(alpha)|
    |sin| -> 0 = head-on / tail-on: the flank genuinely IS NOT VISIBLE, and saying so is the
                 honest answer, not a failure.

No extra geometry, no extra model. Once the head end is chosen, alpha is read straight out of
crops.npz (`face_alpha`), and the viewpoint follows.

THE LABEL-FREE ERROR DETECTOR
-----------------------------
Physics gives us a check that needs no ground truth: an animal CANNOT swap which flank faces the
camera while standing broadside. The flank may only flip when it passes through head-on or tail-on,
i.e. exactly when |sin(alpha)| -> 0. So:

    a flank switch at LOW  |sin| = CORRECT   (the animal genuinely turned through end-on)
    a flank switch at HIGH |sin| = IMPOSSIBLE -> that is an error, and we can count it with no labels

That is a free error estimate on every unlabelled track in the dataset.
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.heading.cropset import CropSet
from tools.heading.descriptors import DEFAULT_MODEL, Config, DenseExtractor, foreground
from tools.heading.split import video_split
from tools.heading.template import Accumulator, choose, opposite_slot

__all__ = ["viewpoint_of"]

EDGE_ON = 0.35          # |sin alpha| below this: the flank is edge-on and a switch is expected


def viewpoint_of(alpha: float) -> dict:
    """The whole re-ID tag, from one angle."""
    s, c = math.sin(alpha), math.cos(alpha)
    return {
        "flank": "LEFT" if s > 0 else "RIGHT",
        "flank_strength": abs(s),          # 1 = fully broadside, 0 = head-on/tail-on
        "end": "FACE" if c < 0 else "REAR",
        "end_strength": abs(c),
        "usable": abs(s) >= EDGE_ON,       # is the flank actually visible enough to re-ID from?
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--facet", default="token", choices=["token", "key"])
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--no-geo-axis", action="store_true",
                    help="let appearance pick the axis too (it is measurably worse at it)")
    ap.add_argument("--per-species", type=int, default=10)
    ap.add_argument("--cell", type=int, default=190)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from PIL import Image, ImageDraw

    crops = CropSet(args.crops)
    d = crops.d
    te, _ = video_split(crops.species, crops.video, seed=args.seed)
    cfg = Config(args.layer, args.facet, args.size, args.bins)

    rng = np.random.default_rng(args.seed)
    idx_fit = np.sort(rng.permutation(np.where(~te)[0])[:1500])
    idx_te = np.where(te)[0]
    crops.prefetch(np.concatenate([idx_fit, idx_te]))

    # ---- template from the WALKING crops of the training videos. Nothing is trained. ----
    with DenseExtractor(args.model, args.device, args.dtype) as ex:
        acc = Accumulator(cfg)
        for b in range(0, len(idx_fit), args.batch):
            items = crops.batch(idx_fit[b: b + args.batch])
            for g, it in zip(ex.grid(np.stack([i.image for i in items]), cfg), items):
                acc.add(g, it)
        tmpl = acc.build()
        print(f"template: {tmpl.n_fitted}")

        pred = np.full(len(idx_te), -1, dtype=int)
        marg = np.zeros(len(idx_te))
        for b in range(0, len(idx_te), args.batch):
            items = crops.batch(idx_te[b: b + args.batch])
            for k, (g, it) in enumerate(zip(ex.grid(np.stack([i.image for i in items]), cfg),
                                            items)):
                if it.species not in tmpl.templates:
                    continue
                s = tmpl.score_faces(g, foreground(g, it.instance), it.face_uv,
                                     it.face_ids, it.species)
                p, m = choose(s, it.face_ids,
                              axis=None if args.no_geo_axis else it.geo_axis)
                pred[b + k], marg[b + k] = p, m
            print(f"  {min(b+args.batch, len(idx_te))}/{len(idx_te)}", end="\r", flush=True)
        print()

    ok = pred >= 0
    j = idx_te[ok]
    p = pred[ok]
    alpha_pred = d["face_alpha"][j, p]                       # the viewpoint, from the chosen head
    alpha_true = d["face_alpha"][j, d["y_face"][j]]
    sp = d["species"][j]

    vp_p = [viewpoint_of(float(a)) for a in alpha_pred]
    vp_t = [viewpoint_of(float(a)) for a in alpha_true]

    # ---- 1. THE NUMBER RE-ID ACTUALLY CONSUMES ----
    fl_p = np.array([v["flank"] for v in vp_p])
    fl_t = np.array([v["flank"] for v in vp_t])
    strong = np.array([v["flank_strength"] for v in vp_t]) >= EDGE_ON

    print(f"\n=== FLANK: which side of the animal are we looking at?  (the re-ID tag) ===")
    print(f"{'species':>9s} {'n':>6s} {'flank acc':>10s} {'...where the flank is VISIBLE':>32s}")
    for s in sorted(set(sp.tolist())):
        m = sp == s
        a_all = float((fl_p[m] == fl_t[m]).mean())
        m2 = m & strong
        a_vis = float((fl_p[m2] == fl_t[m2]).mean()) if m2.any() else float("nan")
        print(f"{s:>9s} {m.sum():6d} {100*a_all:9.1f}% {100*a_vis:31.1f}%")
    print(f"{'ALL':>9s} {len(sp):6d} {100*(fl_p == fl_t).mean():9.1f}% "
          f"{100*(fl_p[strong] == fl_t[strong]).mean():31.1f}%")
    print(f"  chance: 50%.  'VISIBLE' = |sin alpha| >= {EDGE_ON}, i.e. the animal is not head-on --")
    print(f"  which is the only regime where a flank tag is meaningful for re-ID anyway.")

    # ---- 2. VIEWPOINT COVERAGE: does the dataset ever SHOW us both sides? ----
    print(f"\n=== VIEWPOINT COVERAGE (per track) -- can re-ID even match this animal? ===")
    by_tr = defaultdict(list)
    for k in range(len(j)):
        by_tr[str(d["track"][j[k]])].append(k)
    both = one = none = 0
    for tid, ks in by_tr.items():
        v = [vp_p[k] for k in ks if vp_p[k]["usable"]]
        sides = {x["flank"] for x in v}
        if len(sides) == 2:
            both += 1
        elif len(sides) == 1:
            one += 1
        else:
            none += 1
    n_tr = len(by_tr)
    print(f"  tracks showing BOTH flanks : {both:4d} ({100*both/n_tr:.0f}%)  <- can self-match")
    print(f"  tracks showing ONE flank   : {one:4d} ({100*one/n_tr:.0f}%)  <- a side-consistent "
          f"gallery is ESSENTIAL here")
    print(f"  tracks never broadside     : {none:4d} ({100*none/n_tr:.0f}%)  <- no usable flank at all")

    # ---- 3. THE LABEL-FREE PHYSICS CHECK ----
    print(f"\n=== PHYSICS CHECK (needs NO labels): a flank may only switch when edge-on ===")
    bad = tot_sw = 0
    for tid, ks in by_tr.items():
        ks = sorted(ks, key=lambda k: int(d["frame"][j[k]]))
        seq = [(vp_p[k]["flank"], vp_p[k]["flank_strength"]) for k in ks]
        sw = [q for q in range(1, len(seq)) if seq[q][0] != seq[q - 1][0]]
        tot_sw += len(sw)
        bad += sum(1 for q in sw if seq[q][1] > EDGE_ON and seq[q - 1][1] > EDGE_ON)
    print(f"  flank switches: {tot_sw}   of which IMPOSSIBLE (switched while broadside): {bad}")
    print(f"  An animal cannot swap which side faces you while standing side-on. So {bad} is a")
    print(f"  LABEL-FREE error count -- it works on every unlabelled track in the dataset.")

    # ---- 4. THE PICTURE: arrow + what we see ----
    C = args.cell
    species = sorted(set(sp.tolist()))
    sheet = Image.new("RGB", (args.per_species * C, len(species) * (C + 34)), (14, 14, 16))
    draw = ImageDraw.Draw(sheet)

    for r, s in enumerate(species):
        pool = np.where(sp == s)[0]
        pick = rng.permutation(pool)[: args.per_species]
        for c, k in enumerate(pick):
            it = crops.get(int(j[k]))
            x0, y0 = c * C, r * (C + 34)
            sheet.paste(Image.fromarray(it.image).resize((C, C)), (x0, y0))

            uv = it.face_uv * C
            cx, cy = uv.mean(axis=0)                       # box centre
            hx, hy = uv[int(p[k])]                         # the PREDICTED head end
            # the heading ARROW: from the animal's centre toward its head
            draw.line([(x0 + cx, y0 + cy), (x0 + hx, y0 + hy)], fill=(255, 60, 60), width=4)
            ang = math.atan2(hy - cy, hx - cx)
            L = 0.3 * math.hypot(hx - cx, hy - cy)
            for sgn in (+1, -1):
                a = ang + sgn * math.radians(150)
                draw.line([(x0 + hx, y0 + hy),
                           (x0 + hx + L * math.cos(a), y0 + hy + L * math.sin(a))],
                          fill=(255, 60, 60), width=4)
            # the TRUE head, for reference
            th = uv[int(d["y_face"][j[k]])]
            draw.ellipse([x0 + th[0] - 4, y0 + th[1] - 4, x0 + th[0] + 4, y0 + th[1] + 4],
                         fill=(40, 255, 90))
            if int(p[k]) != int(d["y_face"][j[k]]):
                draw.rectangle([x0, y0, x0 + C - 1, y0 + C - 1], outline=(255, 40, 40), width=3)

            v = vp_p[k]
            col = (90, 230, 255) if v["usable"] else (200, 170, 60)
            draw.text((x0 + 4, y0 + C + 3),
                      f"{v['flank']} {v['flank_strength']:.2f}  {v['end']} {v['end_strength']:.2f}",
                      fill=col)
            draw.text((x0 + 4, y0 + C + 18),
                      f"{s[:8]}" + ("" if v["usable"] else "   (head-on: no flank)"),
                      fill=(150, 150, 155))

    args.out.mkdir(parents=True, exist_ok=True)
    q = args.out / f"viewpoint_L{args.layer}_{args.facet}_{args.size}.jpg"
    sheet.save(q, quality=92)
    print(f"\nwrote {q}")
    print("  RED ARROW = the predicted HEADING (centre -> head).  GREEN dot = the true head.")
    print("  The caption is the answer to 'what are we looking at': which flank, how broadside,")
    print("  and whether we see its face or its rear. THAT is the tag re-ID consumes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
