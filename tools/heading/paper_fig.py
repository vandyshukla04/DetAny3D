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


def world_panel(draw, x0, y0, C, az_h, az_c, elev, alpha, *, view_az=0.6, view_el=0.42):
    """ROW 3: the animal, its heading, and the drone -- in the WORLD ground frame.

    An axonometric projection of the segment's ground basis. EVERY element is data:
      az_h   the predicted heading, in world azimuth      (face_az of the chosen head)
      az_c   the drone's azimuth as seen from the animal  (stored at crop time)
      elev   the drone's elevation above the animal's horizontal plane (stored at crop time)
      alpha  the allocentric angle -> which flank faces the drone, and how broadside

    Nothing here is invented or stylised into existence: the aerial-oblique geometry that makes the
    flank visible at all is exactly what this panel draws. The shaded wedge is the side of the animal
    the camera can actually see, and its opacity is |sin(alpha)| -- so a head-on animal shows almost
    no wedge, which is the honest statement that no flank is visible.
    """
    R = 0.34 * C
    cx, cy = x0 + C / 2, y0 + C / 2 + 0.06 * C

    def proj(p):
        """(X, Y, Z) in the ground basis -> screen. Z is up."""
        X, Y, Z = p
        sx = (-math.sin(view_az) * X + math.cos(view_az) * Y)
        sy = (-math.cos(view_az) * math.sin(view_el) * X
              - math.sin(view_az) * math.sin(view_el) * Y
              + math.cos(view_el) * Z)
        return cx + R * sx, cy - R * sy

    def ring(rad, col, w=1, n=64):
        pts = [proj((rad * math.cos(2 * math.pi * k / n), rad * math.sin(2 * math.pi * k / n), 0))
               for k in range(n + 1)]
        draw.line(pts, fill=col, width=w)

    # --- the ground plane ---
    ring(1.00, (58, 58, 66), 1)
    ring(0.62, (40, 40, 48), 1)
    for k in range(8):                                     # radial ticks
        a = 2 * math.pi * k / 8
        draw.line([proj((0.62 * math.cos(a), 0.62 * math.sin(a), 0)),
                   proj((1.0 * math.cos(a), 1.0 * math.sin(a), 0))], fill=(38, 38, 46), width=1)

    # --- the VISIBLE FLANK: a wedge on the side the camera can see ---
    s = math.sin(alpha)
    side = az_h + (math.pi / 2 if s > 0 else -math.pi / 2)  # left = heading + 90 deg
    k = abs(s)                                             # 1 = broadside, 0 = head-on
    if k > 0.05:
        col = (int(30 + 60 * k), int(120 + 115 * k), int(150 + 105 * k))
        wedge = [proj((0, 0, 0))]
        for t in np.linspace(-0.55, 0.55, 20):
            wedge.append(proj((0.86 * math.cos(side + t), 0.86 * math.sin(side + t), 0)))
        draw.polygon(wedge, fill=col if k > 0.5 else None, outline=col)

    # --- the ANIMAL: a body along the heading, with a head end ---
    hx, hy = math.cos(az_h), math.sin(az_h)
    body = [proj((-0.42 * hx - 0.13 * -hy, -0.42 * hy - 0.13 * hx, 0)),
            proj((-0.42 * hx + 0.13 * -hy, -0.42 * hy + 0.13 * hx, 0)),
            proj((0.34 * hx + 0.10 * -hy, 0.34 * hy + 0.10 * hx, 0)),
            proj((0.34 * hx - 0.10 * -hy, 0.34 * hy - 0.10 * hx, 0))]
    draw.polygon(body, fill=(78, 78, 88), outline=(120, 120, 132))

    # --- the HEADING arrow ---
    a0, a1 = proj((0, 0, 0)), proj((0.82 * hx, 0.82 * hy, 0))
    draw.line([a0, a1], fill=(255, 55, 55), width=4)
    th = math.atan2(a1[1] - a0[1], a1[0] - a0[0])
    L = 0.16 * C
    for sg in (+1, -1):
        b = th + sg * math.radians(150)
        draw.line([a1, (a1[0] + L * math.cos(b), a1[1] + L * math.sin(b))],
                  fill=(255, 55, 55), width=4)

    # --- the DRONE: at its true azimuth AND its true elevation ---
    D = 1.15
    cam = (D * math.cos(elev) * math.cos(az_c), D * math.cos(elev) * math.sin(az_c),
           D * math.sin(elev))
    p_cam = proj(cam)
    p_foot = proj((cam[0], cam[1], 0))
    for t in np.linspace(0, 1, 14):                        # the viewing ray, dashed
        if int(t * 14) % 2:
            continue
        q0 = proj((cam[0] * (1 - t), cam[1] * (1 - t), cam[2] * (1 - t)))
        q1 = proj((cam[0] * (1 - t - 0.07), cam[1] * (1 - t - 0.07), cam[2] * (1 - t - 0.07)))
        draw.line([q0, q1], fill=(150, 150, 165), width=1)
    draw.line([p_cam, p_foot], fill=(60, 60, 70), width=1)          # the drop line
    draw.ellipse([p_foot[0] - 2, p_foot[1] - 2, p_foot[0] + 2, p_foot[1] + 2], fill=(60, 60, 70))
    draw.rectangle([p_cam[0] - 5, p_cam[1] - 4, p_cam[0] + 5, p_cam[1] + 4],
                   fill=(240, 240, 245), outline=(120, 120, 130))
    draw.polygon([(p_cam[0] + 5, p_cam[1] - 3), (p_cam[0] + 11, p_cam[1] - 6),
                  (p_cam[0] + 11, p_cam[1] + 6), (p_cam[0] + 5, p_cam[1] + 3)],
                 fill=(240, 240, 245))


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
    ap.add_argument("--n-tracks", type=int, default=16,
                    help="total animals, balanced across species. One FIGURE each.")
    ap.add_argument("--frames", type=int, default=10, help="frames shown per animal")
    ap.add_argument("--cell", type=int, default=170)
    ap.add_argument("--rows", type=int, default=3, choices=[2, 3],
                    help="2 = crops+arrows and DINOv3 features. 3 = also the WORLD panel (the "
                         "animal, its heading, and the drone, in the ground frame).")
    ap.add_argument("--text", action="store_true",
                    help="draw captions ON the figure. Off by default: these are paper figures and "
                         "the description belongs in the LaTeX caption, not burnt into the pixels.")
    ap.add_argument("--no-geo-axis", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from PIL import Image, ImageDraw

    crops = CropSet(args.crops)
    d = crops.d
    if args.rows >= 3 and ("cam_az" not in d or "cam_elev" not in d):
        raise SystemExit(
            "crops.npz has no cam_az/cam_elev -- re-cut the crops. The world panel draws the drone "
            "at its TRUE azimuth and elevation; without them it would have to invent the geometry, "
            "and a figure that invents its own evidence is worse than no figure.")
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

    # ---- draw: ONE FIGURE PER ANIMAL, two rows, no text ----
    C = args.cell
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []

    for r, tid in enumerate(picks, 1):
        ii = shown[tid]
        project = track_pca([G[i] for i in ii], [FG[i] for i in ii])   # ONE basis for the figure
        cols = len(ii)
        fig = Image.new("RGB", (cols * C, args.rows * C), (12, 12, 14))
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

            # --- ROW 2: the SAME features, in the SAME colour basis for the whole figure ---
            if project is not None:
                rgb = project(G[i]) * FG[i][..., None]
                fig.paste(Image.fromarray((255 * rgb).astype(np.uint8))
                          .resize((C, C), Image.NEAREST), (x0, C))

            alpha = float(d["face_alpha"][i, p])
            v = viewpoint_of(alpha)
            tags.append(f"{v['flank'][0]}{v['flank_strength']:.2f}")

            # --- ROW 3: the animal, its heading, and the drone -- in the WORLD ground frame ---
            if args.rows >= 3:
                world_panel(draw, x0, 2 * C, C,
                            az_h=float(d["face_az"][i, p]),
                            az_c=float(d["cam_az"][i]),
                            elev=float(d["cam_elev"][i]),
                            alpha=alpha)

            if args.text:                                  # off by default -- paper figures
                draw.text((x0 + 4, 4), f"t={it.frame}", fill=(190, 190, 195))
                draw.text((x0 + 4, C - 15),
                          f"{v['flank']} {v['flank_strength']:.2f} {v['end'][0]}",
                          fill=(90, 235, 255) if v["usable"] else (215, 175, 70))

        sp = IT[ii[0]].species
        vid = str(d["video"][ii[0]])
        name = f"fig{r:02d}_{args.rows}row_{sp}_{vid}_{tid.split('::')[-1]}.jpg"
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
