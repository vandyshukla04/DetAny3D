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


def mask_outline(mask: np.ndarray, size: int, thickness: int = 2) -> np.ndarray:
    """The BOUNDARY of a crop-space mask, as a (size, size) bool -- not the filled mask.

    Drawn white on the crop, it shows WHICH animal is being tracked when the frame contains a herd,
    without hiding the animal under a colour overlay. Pure numpy: cv2/scipy are not guaranteed in
    the cluster env.

    boundary = foreground patches that touch a background patch (4-neighbour), then dilated to
    `thickness` px so the line reads at figure scale.
    """
    from PIL import Image

    m = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((size, size),
                                                                       Image.NEAREST)) > 127
    if not m.any():
        return np.zeros((size, size), bool)

    inner = (m & np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
    edge = m & ~inner                                     # 1-px boundary
    for _ in range(max(thickness - 1, 0)):               # dilate to `thickness`
        edge = (edge | np.roll(edge, 1, 0) | np.roll(edge, -1, 0)
                | np.roll(edge, 1, 1) | np.roll(edge, -1, 1))
    return edge


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
    ap.add_argument("--n-tracks", type=int, default=24,
                    help="total animals, balanced across species. One FIGURE each.")
    ap.add_argument("--outline", type=int, default=2,
                    help="white SAM-mask boundary thickness (px). 0 = off. Shows WHICH animal is "
                         "tracked in a herd, without covering it.")
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
        # Balanced across species, longest tracks first (they are the most legible), and drawn from
        # DIFFERENT videos where possible -- 16 tracks from one flight would not be 16 samples.
        per_sp = defaultdict(list)
        for tid, ii in by_tr.items():
            if len(ii) >= args.frames:
                per_sp[str(d["species"][ii[0]])].append((len(ii), str(d["video"][ii[0]]), tid))
        for s in per_sp:
            per_sp[s].sort(reverse=True)

        picks, seen_vid = [], defaultdict(set)
        for rnd in range(2):                        # pass 1: one per video. pass 2: fill the rest.
            for s in sorted(per_sp):
                for n, vid, tid in per_sp[s]:
                    if len(picks) >= args.n_tracks:
                        break
                    if tid in picks:
                        continue
                    if rnd == 0 and vid in seen_vid[s]:
                        continue
                    seen_vid[s].add(vid)
                    picks.append(tid)
            if len(picks) >= args.n_tracks:
                break
        picks = picks[: args.n_tracks]

    if not picks:
        print("no tracks found")
        return 1
    print(f"{len(picks)} animals, one FIGURE each")

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

    # ---- draw: ONE FIGURE PER ANIMAL. Two rows + the visibility strip. ----
    C = args.cell
    STRIP = 20                       # the ONE piece of text kept on the figure (see below)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []

    for r, tid in enumerate(picks, 1):
        ii = shown[tid]
        project = track_pca([G[i] for i in ii], [FG[i] for i in ii])   # ONE basis for the figure
        cols = len(ii)
        fig = Image.new("RGB", (cols * C, 2 * C + STRIP), (12, 12, 14))
        draw = ImageDraw.Draw(fig)
        n_ok = 0
        tags = []

        for c, i in enumerate(ii):
            it = IT[i]
            x0 = c * C
            s = tmpl.score_faces(G[i], FG[i], it.face_uv, it.face_ids, it.species)
            p, _ = choose(s, it.face_ids, axis=None if args.no_geo_axis else it.geo_axis)
            if p < 0:
                p = it.y_face
            n_ok += int(p == it.y_face)

            # --- ROW 1: the crop + the predicted heading ARROW ---
            fig.paste(Image.fromarray(it.image).resize((C, C)), (x0, 0))
            uv = it.face_uv * C
            cx, cy = uv.mean(axis=0)
            hx, hy = uv[p]
            draw.line([(x0 + cx, cy), (x0 + hx, hy)], fill=(255, 55, 55), width=4)
            ang = math.atan2(hy - cy, hx - cx)
            L = 0.32 * math.hypot(hx - cx, hy - cy)
            for sg in (+1, -1):
                a = ang + sg * math.radians(150)
                draw.line([(x0 + hx, hy), (x0 + hx + L * math.cos(a), hy + L * math.sin(a))],
                          fill=(255, 55, 55), width=4)
            th = uv[it.y_face]
            draw.ellipse([x0 + th[0] - 4, th[1] - 4, x0 + th[0] + 4, th[1] + 4], fill=(40, 255, 90))

            # --- the WHITE SAM boundary: which animal is being tracked, when the frame is a herd ---
            if it.instance is not None:
                eb = mask_outline(it.instance, C, thickness=args.outline)
                if eb.any():
                    ov = np.zeros((C, C, 4), np.uint8)
                    ov[eb] = (255, 255, 255, 255)         # outline only; the animal is NOT covered
                    fig.paste(Image.fromarray(ov, "RGBA"), (x0, 0), Image.fromarray(ov, "RGBA"))

            # --- ROW 2: the SAME features, in the SAME colour basis for the whole figure ---
            if project is not None:
                rgb = project(G[i]) * FG[i][..., None]
                fig.paste(Image.fromarray((255 * rgb).astype(np.uint8))
                          .resize((C, C), Image.NEAREST), (x0, C + STRIP), )

            # --- THE VISIBILITY STRIP, under row 1. The one text kept on the figure. ---
            # NOT a hard LEFT/RIGHT label: the pair of WEIGHTS. |sin a| is how much flank we see,
            # |cos a| is how much of the front or rear. They are the two components of one unit
            # vector, so an animal that is 0.95 broadside is 0.31 rear -- and a head-on animal reads
            # L0.08 / F0.99, which is the honest statement that NO FLANK IS VISIBLE. That is the
            # quantity re-ID consumes, and it is why a binary side label would throw away the part
            # that matters.
            alpha = float(d["face_alpha"][i, p])
            v = viewpoint_of(alpha)
            tags.append(f"{v['flank'][0]}{v['flank_strength']:.2f}")
            col = (95, 235, 255) if v["usable"] else (225, 175, 70)
            draw.rectangle([x0, C, x0 + C - 1, C + STRIP - 1], fill=(20, 20, 24))
            draw.text((x0 + 5, C + 4),
                      f"{v['flank']} {v['flank_strength']:.2f}   "
                      f"{v['end']} {v['end_strength']:.2f}", fill=col)


        sp = IT[ii[0]].species
        vid = str(d["video"][ii[0]])
        name = f"fig{r:02d}_{sp}_{vid}_{tid.split('::')[-1]}.jpg"
        fig.save(args.out / name, quality=96)
        manifest.append((name, sp, vid, tid, len(by_tr[tid]), len(ii), n_ok, " ".join(tags)))

    # a manifest, so the LaTeX caption can be written from fact rather than memory
    lines = ["file,species,video,track,track_frames,frames_shown,head_correct,viewpoint_tags"]
    for m in manifest:
        lines.append(",".join(str(x) for x in m))
    (args.out / "figures.csv").write_text("\n".join(lines))

    print(f"\nwrote {len(manifest)} figures -> {args.out}/")
    for m in manifest:
        print(f"  {m[0]:<52s} {m[6]}/{m[5]} heads correct")
    print(f"\n  {args.out}/figures.csv  -- species, video, track, accuracy, viewpoint tags per figure")
    print("\n  NO TEXT is drawn on the figures. Suggested caption:")
    print("    TOP: frames of one tracked animal, in temporal order. The red arrow is the predicted")
    print("    3D heading (box centre -> head); the green dot is the head derived from the animal's")
    print("    own locomotion. BOTTOM: the corresponding DINOv3 dense features, projected to RGB")
    print("    through a SINGLE PCA basis fitted over the whole track -- so the same colour denotes")
    print("    the same body part in every frame, across changes of viewpoint, pose and lighting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
