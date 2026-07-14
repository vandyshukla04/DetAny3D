"""SEE the heading progression along a TRACK: crops in time, and the world azimuth over time.

    python -m tools.heading.track_viz --crops data/heading/crops.npz \
        --out data/heading/track --layer 24 --facet token --size 224 --geo-axis

WHY THIS PICTURE
----------------
The per-track accuracy distribution is BIMODAL: tracks are either almost entirely right or almost
entirely WRONG. Measured on zebra (L12/key): 46 of 72 tracks scored under 20% -- not noisy, INVERTED.

That single fact explains the Viterbi null (+0.3%): a decoder can remove a sporadic flip, but a
track that is self-consistently backwards is exactly what a smoothness prior is designed to
PRESERVE. You cannot smooth your way out of being coherently wrong.

So the failure is per-TRACK, not per-frame -- and a table cannot show you what a backwards track
looks like. This can.

WHAT IS DRAWN, per track
------------------------
  TOP    the crops in temporal order.
         GREEN dot = the true head (motion).  RED ring = predicted.  RED border = wrong.
  BOTTOM the WORLD AZIMUTH over time -- the quantity that is actually temporally coherent:
           GREY  = truth (from motion)
           RED   = raw per-frame prediction
           BLUE  = after the Viterbi decode
         A whole-track inversion shows up as RED sitting a constant 180 deg from GREY for the
         entire track -- flat, confident, and wrong. A sporadic error is a single red spike, and
         BLUE should erase it.
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
from tools.heading.tracklet import viterbi_azimuth, wrap


def _plot(draw, x0, y0, W, H, series, frames):
    """A tiny azimuth-vs-time plot. PIL only -- matplotlib is not in the cluster env."""
    draw.rectangle([x0, y0, x0 + W, y0 + H], fill=(24, 24, 28))
    for deg, lab in ((-180, "-180"), (0, "0"), (180, "+180")):
        yy = y0 + H * (1 - (deg + 180) / 360)
        draw.line([(x0, yy), (x0 + W, yy)], fill=(60, 60, 68))
        draw.text((x0 + 2, yy - 6), lab, fill=(110, 110, 120))

    f0, f1 = float(frames[0]), float(frames[-1])
    span = max(f1 - f0, 1.0)
    for name, az, col, r in series:
        pts = [(x0 + W * (float(f) - f0) / span,
                y0 + H * (1 - (float(np.degrees(a)) + 180) / 360)) for f, a in zip(frames, az)]
        for px, py in pts:
            draw.ellipse([px - r, py - r, px + r, py + r], fill=col)
    return


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
    ap.add_argument("--turn-deg", type=float, default=12.0)
    ap.add_argument("--geo-axis", action="store_true")
    ap.add_argument("--tracks", default=None,
                    help="comma-separated track keys. Default: auto-pick the most INSTRUCTIVE ones "
                         "-- the best and the WORST track of each species.")
    ap.add_argument("--max-frames", type=int, default=16)
    ap.add_argument("--cell", type=int, default=120)
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

    with DenseExtractor(args.model, args.device, args.dtype) as ex:
        acc = Accumulator(cfg)
        for b in range(0, len(idx_fit), args.batch):
            items = crops.batch(idx_fit[b: b + args.batch])
            for g, it in zip(ex.grid(np.stack([i.image for i in items]), cfg), items):
                acc.add(g, it)
        tmpl = acc.build()
        print(f"template: {tmpl.n_fitted}")

        S = np.full((len(idx_te), 4), -np.inf)
        for b in range(0, len(idx_te), args.batch):
            items = crops.batch(idx_te[b: b + args.batch])
            for k, (g, it) in enumerate(zip(ex.grid(np.stack([i.image for i in items]), cfg),
                                            items)):
                if it.species not in tmpl.templates:
                    continue
                s = tmpl.score_faces(g, foreground(g, it.instance), it.face_uv,
                                     it.face_ids, it.species)
                S[b + k] = np.where(np.isnan(s), -np.inf, s)
            print(f"  scored {min(b+args.batch, len(idx_te))}/{len(idx_te)}", end="\r", flush=True)
        print()

    # ---- decode every held-out track ----
    by_track = defaultdict(list)
    for k, j in enumerate(idx_te):
        by_track[str(d["track"][j])].append(k)

    turn_std = np.radians(args.turn_deg)
    decoded = {}
    for tid, ks in by_track.items():
        ks = sorted(ks, key=lambda k: int(d["frame"][idx_te[k]]))
        j = idx_te[ks]
        sc = S[ks].copy()
        if args.geo_axis:
            for t in range(len(j)):
                a = int(d["geo_axis"][j[t]])
                ends = {a, opposite_slot(d["face_ids"][j[t]], a)}
                sc[t] = [sc[t][q] if q in ends else -np.inf for q in range(4)]
        if not np.isfinite(sc).any():
            continue
        cand = d["face_az"][j]
        raw = np.argmax(np.where(np.isfinite(sc), sc, -np.inf), axis=1)
        dec = viterbi_azimuth(cand, sc, turn_std=turn_std,
                              dt=np.diff(d["frame"][j].astype(float)))
        truth = d["y_face"][j]
        decoded[tid] = dict(j=j, sc=sc, cand=cand, raw=raw, dec=dec, truth=truth,
                            acc=float((raw == truth).mean()),
                            species=str(d["species"][j[0]]), n=len(j))

    if not decoded:
        print("nothing decoded")
        return 1

    # ---- pick the INSTRUCTIVE tracks: the best AND the worst of each species ----
    if args.tracks:
        picks = [t.strip() for t in args.tracks.split(",") if t.strip() in decoded]
    else:
        picks = []
        for sp in sorted({v["species"] for v in decoded.values()}):
            cand = [(t, v) for t, v in decoded.items()
                    if v["species"] == sp and v["n"] >= 8]
            if not cand:
                continue
            cand.sort(key=lambda tv: tv[1]["acc"])
            picks.append(cand[0][0])                       # the WORST -- this is the one to see
            picks.append(cand[-1][0])                      # and the best, for contrast
    picks = list(dict.fromkeys(picks))
    print(f"rendering {len(picks)} tracks")

    C = args.cell
    PLOT_H = 90
    ROW = C + PLOT_H + 34
    cols = args.max_frames
    sheet = Image.new("RGB", (cols * C, len(picks) * ROW), (14, 14, 16))
    draw = ImageDraw.Draw(sheet)

    for r, tid in enumerate(picks):
        v = decoded[tid]
        y0 = r * ROW
        sel = np.linspace(0, v["n"] - 1, min(cols, v["n"])).astype(int)

        for c, t in enumerate(sel):
            g = int(v["j"][t])
            it = crops.get(g)
            sheet.paste(Image.fromarray(it.image).resize((C, C)), (c * C, y0))
            uv = it.face_uv * C
            th, pr = int(v["truth"][t]), int(v["raw"][t])
            draw.ellipse([c * C + uv[th, 0] - 4, y0 + uv[th, 1] - 4,
                          c * C + uv[th, 0] + 4, y0 + uv[th, 1] + 4], fill=(40, 255, 90))
            draw.ellipse([c * C + uv[pr, 0] - 7, y0 + uv[pr, 1] - 7,
                          c * C + uv[pr, 0] + 7, y0 + uv[pr, 1] + 7],
                         outline=(255, 60, 60), width=2)
            if th != pr:
                draw.rectangle([c * C, y0, (c + 1) * C - 1, y0 + C - 1],
                               outline=(255, 40, 40), width=3)

        # the heading progression, in WORLD azimuth
        fr = d["frame"][v["j"]].astype(float)
        az_t = v["cand"][np.arange(v["n"]), v["truth"]]
        az_r = v["cand"][np.arange(v["n"]), v["raw"]]
        az_d = v["cand"][np.arange(v["n"]), v["dec"]]
        _plot(draw, 0, y0 + C, cols * C - 1, PLOT_H,
              [("truth", az_t, (170, 170, 170), 3),
               ("raw", az_r, (255, 60, 60), 2),
               ("viterbi", az_d, (80, 160, 255), 2)], fr)

        flip = float(np.mean(np.abs(np.degrees(wrap(az_r - az_t))) > 90))
        draw.text((4, y0 + C + PLOT_H + 4),
                  f"{v['species']}  {tid.split('/')[-1]}  n={v['n']}  "
                  f"raw {100*v['acc']:.0f}%  |  frames 180deg-inverted: {100*flip:.0f}%"
                  f"   [GREY truth  RED raw  BLUE viterbi]",
                  fill=(225, 225, 225))

    args.out.mkdir(parents=True, exist_ok=True)
    p = args.out / f"tracks_L{args.layer}_{args.facet}_{args.size}.jpg"
    sheet.save(p, quality=92)
    print(f"\nwrote {p}")
    print("  A WHOLE-TRACK INVERSION looks like: RED flat, a constant 180 deg away from GREY,")
    print("  for the entire track. Confident, coherent, and completely wrong -- and BLUE cannot")
    print("  fix it, because a smoothness prior is designed to PRESERVE exactly that.")
    print("  A sporadic error is a single red spike, and BLUE should erase it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
