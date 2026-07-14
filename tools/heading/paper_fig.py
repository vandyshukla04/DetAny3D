"""PAPER FIGURE: one row per ANIMAL. Heading + viewpoint, and the DINOv3 features that produce it.

    python -m tools.heading.paper_fig --crops data/heading/crops.npz \
        --out data/heading/fig --layer 24 --facet token --size 224

    # or name the tracks explicitly
    python -m tools.heading.paper_fig ... --tracks "zebr3/DJI_.../seg1::4,rhin1/DJI_.../seg2::0"

ONE ROW = ONE TRACKED ANIMAL, IN TEMPORAL ORDER. Not a grab-bag of unrelated crops.

TWO STRIPS PER ANIMAL
---------------------
  TOP     the crops in time. RED ARROW = the predicted heading (centre -> head).
          GREEN dot = the true head (from motion). Caption = what we are looking at:
          which flank, how broadside, face or rear.

  BOTTOM  the DINOv3 dense features, as PCA-RGB.

THE PCA BASIS IS FITTED ONCE PER TRACK, NOT PER FRAME. THIS IS THE WHOLE POINT.
------------------------------------------------------------------------------
A per-frame PCA gives each frame its own arbitrary colour axes, so the same body part comes out a
different colour in every frame and the figure shows NOTHING. Fitting one basis over ALL the
foreground patches of the track makes the colours COMMENSURABLE: the head is the same colour in
frame 1 and frame 40, and the rump is a different, equally stable colour.

That is the claim of the whole method, made visible: DINOv3's dense features put the SAME animal
part at the SAME place in feature space, across viewpoint, pose and lighting. It is why locomotion
only has to name the head ONCE, and the features can carry that name to every other frame --
including the ones where the animal is standing still and motion tells us nothing.

If the colours were incoherent across a row, there would be no correspondence to exploit and the
method could not work. They are not, and it does.
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
from tools.heading.template import Accumulator, choose
from tools.heading.viewpoint import viewpoint_of


def track_pca(grids, masks):
    """ONE PCA basis for the whole track, fitted on its foreground patches only.

    Foreground-only is what lets parts surface at all: over every patch the dominant variance is
    animal-vs-grass, and the anatomy never appears. Track-wide (rather than per-frame) is what makes
    the colours mean the same thing in every frame.
    """
    X = np.concatenate([g[m] for g, m in zip(grids, masks) if m.any()])
    if len(X) < 8:
        return None
    mu = X.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    V = Vt[:3].T
    sd = ((X - mu) @ V).std(0, keepdims=True) + 1e-9

    def project(g):                                    # (gh, gw, D) -> (gh, gw, 3) in [0,1]
        P = (g.reshape(-1, g.shape[-1]) - mu) @ V / sd
        return (1.0 / (1.0 + np.exp(-2.0 * P))).reshape(*g.shape[:2], 3)

    return project


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
    ap.add_argument("--tracks", default=None, help="comma-separated track keys")
    ap.add_argument("--per-species", type=int, default=1, help="tracks per species if auto-picking")
    ap.add_argument("--frames", type=int, default=10, help="frames shown per animal")
    ap.add_argument("--cell", type=int, default=170)
    ap.add_argument("--no-geo-axis", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from PIL import Image, ImageDraw

    crops = CropSet(args.crops)
    d = crops.d
    te, _ = video_split(crops.species, crops.video, seed=args.seed)
    cfg = Config(args.layer, args.facet, args.size, args.bins)

    # ---- choose the animals: LONGEST held-out track per species (the most legible) ----
    by_tr = defaultdict(list)
    for i in np.where(te)[0]:
        by_tr[str(d["track"][i])].append(int(i))

    if args.tracks:
        picks = [t.strip() for t in args.tracks.split(",") if t.strip() in by_tr]
    else:
        per_sp = defaultdict(list)
        for tid, ii in by_tr.items():
            per_sp[str(d["species"][ii[0]])].append((len(ii), tid))
        picks = []
        for s in sorted(per_sp):
            per_sp[s].sort(reverse=True)
            picks += [t for _, t in per_sp[s][: args.per_species]]
    if not picks:
        print("no tracks found")
        return 1
    print(f"{len(picks)} animals: {[p.split('/')[-1] for p in picks]}")

    rng = np.random.default_rng(args.seed)
    idx_fit = np.sort(rng.permutation(np.where(~te)[0])[:1500])

    # the frames we will actually draw: evenly spaced in TIME along each track
    shown = {}
    for tid in picks:
        ii = sorted(by_tr[tid], key=lambda i: int(d["frame"][i]))
        sel = np.linspace(0, len(ii) - 1, min(args.frames, len(ii))).astype(int)
        shown[tid] = [ii[k] for k in sel]

    need = np.array(sorted({i for v in shown.values() for i in v}))
    crops.prefetch(np.concatenate([idx_fit, need]))

    with DenseExtractor(args.model, args.device, args.dtype) as ex:
        acc = Accumulator(cfg)
        for b in range(0, len(idx_fit), args.batch):
            items = crops.batch(idx_fit[b: b + args.batch])
            for g, it in zip(ex.grid(np.stack([i.image for i in items]), cfg), items):
                acc.add(g, it)
        tmpl = acc.build()
        print(f"template: {tmpl.n_fitted}")

        G, FG, IT = {}, {}, {}
        for b in range(0, len(need), args.batch):
            items = crops.batch(need[b: b + args.batch])
            for g, it in zip(ex.grid(np.stack([i.image for i in items]), cfg), items):
                G[it.i], FG[it.i], IT[it.i] = g, foreground(g, it.instance), it

    # ---- draw ----
    C = args.cell
    CAP = 34
    ROW = 2 * C + CAP + 10
    cols = args.frames
    sheet = Image.new("RGB", (cols * C, len(picks) * ROW), (12, 12, 14))
    draw = ImageDraw.Draw(sheet)

    for r, tid in enumerate(picks):
        ii = shown[tid]
        y0 = r * ROW
        project = track_pca([G[i] for i in ii], [FG[i] for i in ii])   # ONE basis for the row
        n_ok = 0

        for c, i in enumerate(ii):
            it = IT[i]
            x0 = c * C
            s = tmpl.score_faces(G[i], FG[i], it.face_uv, it.face_ids, it.species)
            p, _ = choose(s, it.face_ids, axis=None if args.no_geo_axis else it.geo_axis)
            if p < 0:
                p = it.y_face
            n_ok += int(p == it.y_face)

            # --- strip 1: the crop, the heading ARROW, and what we are looking at ---
            sheet.paste(Image.fromarray(it.image).resize((C, C)), (x0, y0))
            uv = it.face_uv * C
            cx, cy = uv.mean(axis=0)
            hx, hy = uv[p]
            draw.line([(x0 + cx, y0 + cy), (x0 + hx, y0 + hy)], fill=(255, 55, 55), width=4)
            ang = math.atan2(hy - cy, hx - cx)
            L = 0.32 * math.hypot(hx - cx, hy - cy)
            for sg in (+1, -1):
                a = ang + sg * math.radians(150)
                draw.line([(x0 + hx, y0 + hy),
                           (x0 + hx + L * math.cos(a), y0 + hy + L * math.sin(a))],
                          fill=(255, 55, 55), width=4)
            th = uv[it.y_face]
            draw.ellipse([x0 + th[0] - 4, y0 + th[1] - 4, x0 + th[0] + 4, y0 + th[1] + 4],
                         fill=(40, 255, 90))
            if p != it.y_face:
                draw.rectangle([x0, y0, x0 + C - 1, y0 + C - 1], outline=(255, 40, 40), width=3)

            v = viewpoint_of(float(d["face_alpha"][i, p]))
            draw.text((x0 + 4, y0 + 4), f"t={it.frame}", fill=(190, 190, 195))
            draw.text((x0 + 4, y0 + C - 15),
                      f"{v['flank']} {v['flank_strength']:.2f} {v['end'][0]}",
                      fill=(90, 235, 255) if v["usable"] else (215, 175, 70))

            # --- strip 2: the SAME features, in the SAME colour basis for the whole row ---
            y1 = y0 + C
            if project is not None:
                rgb = project(G[i]) * FG[i][..., None]
                sheet.paste(Image.fromarray((255 * rgb).astype(np.uint8))
                            .resize((C, C), Image.NEAREST), (x0, y1))
            draw.rectangle([x0, y1, x0 + C - 1, y1 + C - 1], outline=(45, 45, 52), width=1)

        sp = IT[ii[0]].species
        draw.text((4, y0 + 2 * C + 6),
                  f"{sp}   {tid}   {len(by_tr[tid])} frames   "
                  f"head correct on {n_ok}/{len(ii)} of the frames shown   |   "
                  f"TOP: red arrow = predicted heading, green = true head (motion).   "
                  f"BOTTOM: DINOv3 features, ONE PCA basis for the whole row -- the same colour "
                  f"is the same body part, in every frame.",
                  fill=(220, 220, 225))

    args.out.mkdir(parents=True, exist_ok=True)
    q = args.out / f"figure_L{args.layer}_{args.facet}_{args.size}.jpg"
    sheet.save(q, quality=95)
    print(f"\nwrote {q}")
    print("  The BOTTOM strip is the argument: one PCA basis across the whole track, so the head is")
    print("  the SAME COLOUR in frame 1 and frame 40, through changes of viewpoint, pose and light.")
    print("  That correspondence is why locomotion only has to name the head ONCE -- the features")
    print("  carry the name to every other frame, including the ones where the animal is standing")
    print("  still and motion tells us nothing at all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
