"""16 single-row figures: WHERE IS THE CAMERA, relative to the animal's heading?  [GPU / cluster]

    python -m tools.heading.cam_fig --crops data/heading/crops.npz \
        --out data/heading/camfig --layer 24 --facet token --size 224 --n-tracks 16

ONE FIGURE PER TRACKED ANIMAL. One row: the frames in temporal order.
Each cell is a 3D view of TWO vectors and nothing else:

    the RED ARROW   the animal's predicted heading, lying in the ground plane
    the CAMERA      the drone, at its TRUE azimuth and TRUE elevation above the animal

plus a dashed sight-line from the drone to the animal, and a drop-line to the ground so the height
reads as height. That is the whole diagram. No animal is drawn, no wedge, no grid -- the question is
only where the camera sits with respect to the heading, and everything else was clutter.

WHY THIS IS THE FIGURE THAT EXPLAINS THE METHOD
-----------------------------------------------
The angle between those two vectors IS the result. Call it alpha (the allocentric angle):

    the drone off to the animal's SIDE      -> we see a flank      -> |sin alpha| ~ 1
    the drone BEHIND or AHEAD of the animal -> we see rump or face -> |sin alpha| ~ 0

So the picture is not decoration: sweeping the camera around the arrow sweeps the viewpoint tag
through LEFT -> REAR -> RIGHT -> FACE, and a re-ID system can only match two sightings of an animal
when this diagram looks similar in both.

EVERY ELEMENT IS DATA
---------------------
`cam_az` and `cam_elev` are stored at crop time from the segment's own camera poses. Nothing here is
stylised into existence. Measured across the dataset: the drone sits a median of 18 degrees above the
animal (p10 9, p90 30) -- AERIAL-OBLIQUE, not top-down. That obliquity is a precondition of the task:
from directly overhead an animal presents its back and no flank at all, and the viewpoint question
would not arise.
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

VIEW_AZ, VIEW_EL = 0.62, 0.40          # the fixed vantage the diagram itself is drawn from


def draw_cell(draw, x0, y0, C, az_h, az_c, elev):
    """Camera + heading. Two vectors, one ground plane, nothing else."""
    R = 0.33 * C
    cx, cy = x0 + C / 2, y0 + C / 2 + 0.10 * C

    def proj(X, Y, Z):
        sx = -math.sin(VIEW_AZ) * X + math.cos(VIEW_AZ) * Y
        sy = (-math.cos(VIEW_AZ) * math.sin(VIEW_EL) * X
              - math.sin(VIEW_AZ) * math.sin(VIEW_EL) * Y + math.cos(VIEW_EL) * Z)
        return cx + R * sx, cy - R * sy

    # the ground plane: a single ellipse. A horizon, not a grid.
    n = 72
    draw.polygon([proj(math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n), 0)
                  for k in range(n)], fill=(26, 27, 32), outline=(56, 58, 68))

    O = proj(0, 0, 0)

    # --- the drone: TRUE azimuth, TRUE elevation ---
    D = 1.05
    Cx, Cy, Cz = (D * math.cos(elev) * math.cos(az_c),
                  D * math.cos(elev) * math.sin(az_c),
                  D * math.sin(elev))
    P = proj(Cx, Cy, Cz)
    F = proj(Cx, Cy, 0)

    draw.line([F, P], fill=(70, 72, 84), width=1)                       # drop-line: height reads
    draw.ellipse([F[0] - 2, F[1] - 2, F[0] + 2, F[1] + 2], fill=(70, 72, 84))
    for t in np.linspace(0, 1, 16):                                     # dashed sight-line
        if int(t * 16) % 2:
            continue
        a = proj(Cx * (1 - t), Cy * (1 - t), Cz * (1 - t))
        b = proj(Cx * (1 - t - 0.06), Cy * (1 - t - 0.06), Cz * (1 - t - 0.06))
        draw.line([a, b], fill=(155, 158, 175), width=1)

    # --- the heading: a red arrow in the ground plane ---
    hx, hy = math.cos(az_h), math.sin(az_h)
    A = proj(0.78 * hx, 0.78 * hy, 0)
    draw.line([O, A], fill=(255, 60, 60), width=5)
    th = math.atan2(A[1] - O[1], A[0] - O[0])
    L = 0.15 * C
    draw.polygon([A,
                  (A[0] + L * math.cos(th + math.radians(150)),
                   A[1] + L * math.sin(th + math.radians(150))),
                  (A[0] + L * math.cos(th - math.radians(150)),
                   A[1] + L * math.sin(th - math.radians(150)))], fill=(255, 60, 60))
    draw.ellipse([O[0] - 4, O[1] - 4, O[0] + 4, O[1] + 4], fill=(240, 240, 245))   # the animal

    # --- the camera body, drawn last so it sits on top ---
    draw.rectangle([P[0] - 6, P[1] - 5, P[0] + 6, P[1] + 5],
                   fill=(245, 246, 250), outline=(110, 112, 125))
    draw.polygon([(P[0] + 6, P[1] - 4), (P[0] + 13, P[1] - 7),
                  (P[0] + 13, P[1] + 7), (P[0] + 6, P[1] + 4)], fill=(245, 246, 250))


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
    ap.add_argument("--n-tracks", type=int, default=16)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--cell", type=int, default=200)
    ap.add_argument("--strip", action="store_true", help="also print the viewpoint weights")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from PIL import Image, ImageDraw

    crops = CropSet(args.crops)
    d = crops.d
    for k in ("cam_az", "cam_elev"):
        if k not in d:
            raise SystemExit(
                f"crops.npz has no `{k}` -- re-cut the crops. The camera is drawn at its TRUE "
                f"azimuth and elevation; without them the figure would have to invent the drone's "
                f"position, and a figure that invents its own evidence is worse than no figure.")

    te, _ = video_split(crops.species, crops.video, seed=args.seed)
    cfg = Config(args.layer, args.facet, args.size, args.bins)

    by_tr = defaultdict(list)
    for i in np.where(te)[0]:
        by_tr[str(d["track"][i])].append(int(i))

    per_sp = defaultdict(list)
    for tid, ii in by_tr.items():
        if len(ii) >= args.frames:
            per_sp[str(d["species"][ii[0]])].append((len(ii), str(d["video"][ii[0]]), tid))
    for s in per_sp:
        per_sp[s].sort(reverse=True)
    picks, seen = [], defaultdict(set)
    for rnd in range(2):                              # one per video first, then fill
        for s in sorted(per_sp):
            for n, vid, tid in per_sp[s]:
                if len(picks) >= args.n_tracks:
                    break
                if tid in picks or (rnd == 0 and vid in seen[s]):
                    continue
                seen[s].add(vid)
                picks.append(tid)
        if len(picks) >= args.n_tracks:
            break
    picks = picks[: args.n_tracks]
    print(f"{len(picks)} animals, one single-row figure each")

    shown = {}
    for tid in picks:
        ii = sorted(by_tr[tid], key=lambda i: int(d["frame"][i]))
        sel = np.linspace(0, len(ii) - 1, min(args.frames, len(ii))).astype(int)
        shown[tid] = [ii[k] for k in sel]

    rng = np.random.default_rng(args.seed)
    idx_fit = np.sort(rng.permutation(np.where(~te)[0])[:1500])
    need = np.array(sorted({i for v in shown.values() for i in v}))
    crops.prefetch(np.concatenate([idx_fit, need]))

    with DenseExtractor(args.model, args.device, args.dtype) as ex:
        acc = Accumulator(cfg)
        for b in range(0, len(idx_fit), args.batch):
            items = crops.batch(idx_fit[b: b + args.batch])
            for g, it in zip(ex.grid(np.stack([i.image for i in items]), cfg), items):
                acc.add(g, it)
        tmpl = acc.build()

        P = {}
        for b in range(0, len(need), args.batch):
            items = crops.batch(need[b: b + args.batch])
            for g, it in zip(ex.grid(np.stack([i.image for i in items]), cfg), items):
                s = tmpl.score_faces(g, foreground(g, it.instance), it.face_uv,
                                     it.face_ids, it.species)
                p, _ = choose(s, it.face_ids, axis=it.geo_axis)
                P[it.i] = it.y_face if p < 0 else p

    C = args.cell
    STRIP = 20 if args.strip else 0
    args.out.mkdir(parents=True, exist_ok=True)

    for r, tid in enumerate(picks, 1):
        ii = shown[tid]
        fig = Image.new("RGB", (len(ii) * C, C + STRIP), (12, 12, 14))
        draw = ImageDraw.Draw(fig)
        for c, i in enumerate(ii):
            p = P[i]
            draw_cell(draw, c * C, 0, C,
                      az_h=float(d["face_az"][i, p]),
                      az_c=float(d["cam_az"][i]),
                      elev=float(d["cam_elev"][i]))
            if STRIP:
                v = viewpoint_of(float(d["face_alpha"][i, p]))
                draw.rectangle([c * C, C, c * C + C - 1, C + STRIP - 1], fill=(20, 20, 24))
                draw.text((c * C + 5, C + 4),
                          f"{v['flank']} {v['flank_strength']:.2f}   "
                          f"{v['end']} {v['end_strength']:.2f}",
                          fill=(95, 235, 255) if v["usable"] else (225, 175, 70))
        sp = str(d["species"][ii[0]])
        vid = str(d["video"][ii[0]])
        fig.save(args.out / f"cam{r:02d}_{sp}_{vid}_{tid.split('::')[-1]}.jpg", quality=96)

    print(f"\nwrote {len(picks)} figures -> {args.out}/")
    print("  Each cell: the RED ARROW is the animal's heading in the ground plane; the CAMERA is the")
    print("  drone at its TRUE azimuth and elevation, with a dashed sight-line down to the animal.")
    print("  The ANGLE BETWEEN THEM is the result: camera to the SIDE of the arrow -> we see a flank;")
    print("  camera BEHIND or AHEAD of it -> we see the rump or the face, and no flank at all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
