"""Motion -> signed 3D WORLD heading. The free evaluation set. No model, no GPU, no human.

    python -m tools.heading.autolabel --root /mnt/d/3DBOX/papersubdata --out data/heading/labels.npz

THE IDEA
--------
A walking animal labels its own heading. Its world-space velocity IS its facing direction,
and -- unlike the 3D box, whose PCA/SVD axis signs are arbitrary -- it is **signed**. So a
quantity the box can never give us falls out of data we already have, for free, on every
species and every video.

LOCOMOTION IS THE TEST SET, NOT THE METHOD.
Nothing downstream depends on the animal moving. These labels exist to SCORE a heading
predictor on 4 species and 60 videos with zero manual annotation. The predictor itself must
work from a single frame on a standing animal.

WHAT IT EMITS: THE FRONT FACE, NOT A SIGN
-----------------------------------------
The box's 4 horizontal faces are the front candidates (the head is never on the dorsal or
ventral face -- verified on 11,084 human-locked instances: 100%). We label the one the animal
is walking toward. That fixes the body AXIS and its SIGN in one step.

Labelling only a *sign* on the "longest horizontal axis" would be a trap: that heuristic is
wrong 5.6% of the time on real boxes (frame.py), and for some tracks the PCA box is genuinely
wider than it is long, so *no* sign can rescue it. Picking the face directly sidesteps this,
and `--report` measures how often the longest-axis shortcut would have been wrong.

THE TWO GATES (purity, not recall -- we want clean labels, not many)
-------------------------------------------------------------------
  A. MOTION     displacement > --min-speed body-lengths per --window frames.
                Body-lengths, never metres: the VGGT world scale is arbitrary.
                Centres are smoothed first -- raw box-fit jitter accumulates ~3-6 BL of
                fake "path" on a *stationary* grazing zebra.
  B. AGREEMENT  cos(velocity, chosen face normal) > --min-agree.
                The box and the motion are INDEPENDENT signals. Measured over all 305
                segments they agree at |cos| = 0.953 (rhino .979, elephant .969, giraffe
                .934, zebra .818; random = 0.707). Requiring them to agree throws away the
                samples where the box fit is bad -- which is exactly where a motion label
                would have been wrong too.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.heading.conventions import FACE_AXIS
from tools.heading.papersub import check_scale, iter_segments

__all__ = ["smooth_centers", "label_track"]


def smooth_centers(C: np.ndarray, k: int) -> np.ndarray:
    """Moving average over the track. Without this, per-frame box-fit jitter dominates the
    velocity of a slow animal and the 'heading' is noise."""
    if k <= 1 or len(C) < k:
        return C
    ker = np.ones(k) / k
    pad = k // 2
    P = np.pad(C, ((pad, pad), (0, 0)), mode="edge")
    return np.stack([np.convolve(P[:, d], ker, mode="valid")[: len(C)] for d in range(3)], axis=1)


def label_track(seg, tr, *, window: int, min_speed: float, min_agree: float, smooth: int):
    """Yield one label per frame that passes both gates."""
    if len(tr) < window + 2:
        return
    L = tr.body_length
    if not np.isfinite(L) or L <= 0:
        return

    Cs = smooth_centers(tr.centers, smooth)
    up = seg.up_unsigned

    for i in range(len(tr) - window):
        v = Cs[i + window] - Cs[i]
        v = v - np.dot(v, up) * up                       # horizontal displacement, world
        speed = float(np.linalg.norm(v)) / L             # body-lengths per window
        if speed <= min_speed:                           # --- GATE A: is it actually walking?
            continue
        h = v / (np.linalg.norm(v) + 1e-12)              # the WORLD heading. Signed. Free.

        faces = seg.horizontal_face_dirs(tr, i)          # the 4 front candidates
        if len(faces) < 2:
            continue
        fid, fdir = max(faces.items(), key=lambda kv: float(np.dot(kv[1], h)))
        agree = float(np.dot(fdir, h))
        if agree <= min_agree:                           # --- GATE B: box and motion must agree
            continue

        # Diagnostic: would the "longest horizontal axis" shortcut have picked this axis?
        # The extent along face f is dims[FACE_AXIS[f]] (a LOCAL index), scaled by how much
        # of that axis survives the projection into the ground plane.
        R = tr.rotations[i]
        extents = {}
        for f in faces:
            c = FACE_AXIS[f]
            a = R[:, c] - np.dot(R[:, c], up) * up
            extents[f] = float(tr.dims[i][c] * np.linalg.norm(a))
        on_long_axis = FACE_AXIS[fid] == FACE_AXIS[max(extents, key=extents.get)]

        yield {
            "seg": seg.key, "group": seg.group, "video": seg.video,
            "species": seg.species, "track": tr.tid,
            "i": i, "frame": int(tr.frames[i]),
            "heading": h, "front_face": int(fid),
            "faces": sorted(faces), "agree": agree, "speed_bl": speed,
            "on_long_axis": bool(on_long_axis),
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("/mnt/d/3DBOX/papersubdata"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window", type=int, default=15, help="frames over which to measure motion")
    ap.add_argument("--smooth", type=int, default=9, help="moving-average window on centres")
    ap.add_argument("--min-speed", type=float, default=0.30,
                    help="GATE A: body-lengths moved per window")
    ap.add_argument("--min-agree", type=float, default=0.80,
                    help="GATE B: cos(velocity, front-face normal)")
    args = ap.parse_args()

    rows, per_sp = [], defaultdict(lambda: defaultdict(int))
    n_seg = 0
    for seg in iter_segments(args.root):
        try:
            check_scale(seg)                              # fail loudly on unexpected conventions
        except ValueError as e:
            print(f"  SCALE CHECK FAILED, skipping: {e}")
            continue
        n_seg += 1
        for tr in seg.tracks.values():
            per_sp[seg.species]["tracks"] += 1
            got = 0
            for r in label_track(seg, tr, window=args.window, min_speed=args.min_speed,
                                 min_agree=args.min_agree, smooth=args.smooth):
                rows.append(r)
                got += 1
                per_sp[seg.species]["on_long_axis"] += int(r["on_long_axis"])
            per_sp[seg.species]["labels"] += got
            per_sp[seg.species]["labelled_tracks"] += int(got > 0)
        if n_seg % 25 == 0:
            print(f"  {n_seg} segments, {len(rows)} labels", flush=True)

    if not rows:
        print("no labels survived the gates")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        seg=np.array([r["seg"] for r in rows]),
        group=np.array([r["group"] for r in rows]),
        video=np.array([r["video"] for r in rows]),
        species=np.array([r["species"] for r in rows]),
        track=np.array([r["track"] for r in rows]),
        i=np.array([r["i"] for r in rows], dtype=np.int32),
        frame=np.array([r["frame"] for r in rows], dtype=np.int32),
        heading=np.stack([r["heading"] for r in rows]).astype(np.float32),
        front_face=np.array([r["front_face"] for r in rows], dtype=np.int8),
        faces=np.stack([np.array(r["faces"], dtype=np.int8) for r in rows]),
        agree=np.array([r["agree"] for r in rows], dtype=np.float32),
        speed_bl=np.array([r["speed_bl"] for r in rows], dtype=np.float32),
        on_long_axis=np.array([r["on_long_axis"] for r in rows]),
    )

    print(f"\n=== MOTION-DERIVED WORLD HEADINGS  ({n_seg} segments) ===")
    print(f"{'species':>9s} {'tracks':>7s} {'labelled':>9s} {'labels':>8s} {'median agree':>13s} "
          f"{'longest-axis OK':>16s}")
    for sp, d in sorted(per_sp.items()):
        m = np.array([r["agree"] for r in rows if r["species"] == sp])
        la = 100 * d["on_long_axis"] / max(d["labels"], 1)
        print(f"{sp:>9s} {d['tracks']:7d} {d['labelled_tracks']:9d} {d['labels']:8d} "
              f"{np.median(m) if len(m) else float('nan'):13.3f} {la:15.0f}%")
    la_all = 100 * sum(d["on_long_axis"] for d in per_sp.values()) / len(rows)
    print(f"{'TOTAL':>9s} {'':7s} {'':9s} {len(rows):8d} "
          f"{np.median([r['agree'] for r in rows]):13.3f} {la_all:15.0f}%")
    print(f"\n  'longest-axis OK' = how often the longest-horizontal-axis shortcut picks the")
    print(f"  same face the animal actually walks toward. {100-la_all:.0f}% of the time it does")
    print(f"  NOT -- which is why the front face is labelled directly, not as a sign.")
    print(f"\n  -> {args.out}   ({len(rows)} labels, 0 human annotations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
