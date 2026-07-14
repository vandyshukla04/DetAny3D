"""X2: recover the heading over a whole TRACK, in WORLD space. Where a weak cue becomes strong.

    python -m tools.heading.tracklet --crops data/heading/crops.npz \
        --out data/heading/track --layer 12 --facet key --size 448 --device cuda

WHY THIS IS THE PAYOFF
----------------------
The per-frame appearance cue is real but imperfect (2-way ~84%, and much worse on occluded
zebras). We do not need it to be better. We need to ACCUMULATE it.

An animal's heading is coherent in the WORLD, but every frame observes it from a different
viewpoint, pose, and light -- so the per-frame errors are largely INDEPENDENT. Integrate them
along a track and they cancel: a 75%-per-frame cue over 40 frames is a near-certain track answer.
That is standard evidence accumulation, and it is the single highest-leverage thing available.

WORLD AZIMUTH, NOT THE IMAGE ANGLE. THIS IS THE WHOLE TRICK.
------------------------------------------------------------
An earlier attempt smoothed the IMAGE heading and made things strictly WORSE (96.8% -> 79.0%).
The reason: the drone moves, so the projected angle of a perfectly stationary animal jumps around.
Smoothing it smooths the CAMERA, not the animal.

World azimuth is the quantity that is actually temporally coherent -- and getting it requires the
camera and the ground plane, which is exactly why the 3D scaffolding is load-bearing rather than
decorative. `crops.npz` carries `face_az`: each candidate face's azimuth in a basis fixed for the
whole segment (papersub.Segment.ground_basis), so the numbers are comparable across frames.

VITERBI, NOT A MAJORITY VOTE
-----------------------------
A track-level vote assumes the animal never turns. It scored 100% once -- but only because grazing
animals barely move, and it is wrong BY CONSTRUCTION for an animal that turns around. Instead we
keep a heading PER FRAME and constrain consecutive frames with a bounded turn-rate prior:

    permits a genuine turn        (the animal really can rotate, and does)
    forbids an instantaneous flip (it physically cannot swap ends between two frames)

The 180-degree flip is precisely the error that matters -- it inverts the LEFT/RIGHT flank tag and
so poisons re-ID -- and it is exactly what a turn-rate prior kills.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.heading.cropset import CropSet
from tools.heading.descriptors import DEFAULT_MODEL, Config, DenseExtractor, foreground
from tools.heading.split import video_split
from tools.heading.template import AxisTemplate

__all__ = ["viterbi_azimuth", "wrap"]


def wrap(a):
    """Angle(s) into (-pi, pi]."""
    return np.arctan2(np.sin(a), np.cos(a))


def viterbi_azimuth(cand_az: np.ndarray, scores: np.ndarray, *, turn_std: float,
                    dt: np.ndarray | None = None) -> np.ndarray:
    """Decode one heading per frame from per-frame candidate azimuths + their scores.

    cand_az : (T, K) candidate WORLD azimuths per frame (the box's horizontal faces)
    scores  : (T, K) appearance evidence for each candidate (-inf where unusable)
    turn_std: radians of turn the animal may plausibly make between consecutive samples

    Returns (T,) chosen candidate INDEX per frame.

    The transition cost is quadratic in the turn actually implied by moving from candidate j at
    t-1 to candidate k at t. A 180-degree jump is enormously expensive, a gentle turn is cheap --
    which is the physics, stated directly.
    """
    T, K = cand_az.shape
    if T == 0:
        return np.zeros(0, dtype=int)
    if T == 1:
        return np.array([int(np.argmax(scores[0]))])

    gap = np.ones(T - 1) if dt is None else np.maximum(np.asarray(dt, float)[: T - 1], 1.0)

    dp = scores[0].astype(np.float64).copy()
    back = np.zeros((T, K), dtype=int)
    for t in range(1, T):
        # turn implied by every (previous candidate -> current candidate) pair
        turn = wrap(cand_az[t][None, :] - cand_az[t - 1][:, None])        # (K_prev, K_cur)
        allow = turn_std * np.sqrt(gap[t - 1])       # longer gap => a bigger turn is plausible
        cost = 0.5 * (turn / max(allow, 1e-6)) ** 2
        M = dp[:, None] - cost                                            # (K_prev, K_cur)
        back[t] = np.argmax(M, axis=0)
        dp = M[back[t], np.arange(K)] + scores[t]

    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(dp))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--facet", default="key", choices=["token", "key"])
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--turn-deg", type=float, default=12.0,
                    help="plausible turn between consecutive SAMPLES. Too small over-smooths a real "
                         "turn; too large lets an impossible 180-deg flip through.")
    ap.add_argument("--geo-axis", action="store_true",
                    help="restrict to the geometric body axis (geometry proposes, appearance "
                         "disposes). Appearance is measurably bad at picking the axis.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    crops = CropSet(args.crops)
    d = crops.d
    if "face_az" not in d:
        raise SystemExit("crops.npz has no `face_az` -- re-cut the crops. World azimuth is what "
                         "makes temporal smoothing valid; smoothing the image angle smooths the "
                         "DRONE and made things worse last time.")

    te, test_v = video_split(crops.species, crops.video, seed=args.seed)
    print(f"{len(crops)} crops | held out {len(test_v)} videos ({te.sum()} crops) | "
          f"SAM masks: {'available' if crops.can_mask else 'NOT FOUND -- zebra stays at chance'}")

    cfg = Config(args.layer, args.facet, args.size, args.bins)
    rng = np.random.default_rng(args.seed)
    idx_fit = np.sort(rng.permutation(np.where(~te)[0])[:1500])

    # ---- per-frame scores on the held-out videos ----
    with DenseExtractor(args.model, args.device) as ex:
        tmpl = AxisTemplate.fit(ex, crops, idx_fit, cfg, args.batch)
        print(f"template: {tmpl.n_fitted}\n")

        idx_te = np.where(te)[0]
        S = np.full((len(idx_te), 4), -np.inf)
        for b in range(0, len(idx_te), args.batch):
            items = crops.batch(idx_te[b: b + args.batch])
            G = ex.grid(np.stack([it.image for it in items]), cfg)
            for k, (g, it) in enumerate(zip(G, items)):
                if it.species not in tmpl.templates:
                    continue
                s = tmpl.score_faces(g, foreground(g, it.instance), it.face_uv,
                                     it.face_ids, it.species)
                S[b + k] = np.where(np.isnan(s), -np.inf, s)
            if b % (args.batch * 20) == 0:
                print(f"  scored {b}/{len(idx_te)}", flush=True)

    # ---- group by TRACK, decode in world azimuth ----
    by_track: dict[str, list[int]] = defaultdict(list)
    for k, j in enumerate(idx_te):
        by_track[str(d["track"][j])].append(k)

    turn_std = np.radians(args.turn_deg)
    rows = []
    for tid, ks in by_track.items():
        ks = sorted(ks, key=lambda k: int(d["frame"][idx_te[k]]))
        j = idx_te[ks]
        frames = d["frame"][j].astype(float)
        cand = d["face_az"][j]                                   # (T, 4) WORLD azimuths
        sc = S[ks]
        if not np.isfinite(sc).any():
            continue

        if args.geo_axis:                                        # geometry proposes
            ga = d["geo_axis"][j]
            from tools.heading.template import opposite_slot
            mask = np.zeros_like(sc, dtype=bool)
            for t in range(len(j)):
                a = int(ga[t])
                mask[t, a] = True
                mask[t, opposite_slot(d["face_ids"][j[t]], a)] = True
            sc = np.where(mask, sc, -np.inf)

        raw = np.argmax(np.where(np.isfinite(sc), sc, -np.inf), axis=1)
        dec = viterbi_azimuth(cand, sc, turn_std=turn_std, dt=np.diff(frames))

        truth = d["y_face"][j]
        rows.append((str(d["species"][j[0]]), tid, len(j),
                     float((raw == truth).mean()), float((dec == truth).mean())))

    if not rows:
        print("no tracks decoded")
        return 1

    # ---- report: does the evidence actually accumulate? ----
    n = np.array([r[2] for r in rows])
    raw = np.array([r[3] for r in rows])
    dec = np.array([r[4] for r in rows])
    W = n / n.sum()

    print(f"\n=== TRACK-LEVEL DECODE  ({len(rows)} held-out tracks) ===")
    print(f"  per-frame, RAW (argmax)      : {100*(raw*W).sum():5.1f}%")
    print(f"  per-frame, VITERBI (world az): {100*(dec*W).sum():5.1f}%   <- the accumulation")
    print(f"  chance                       :  25.0%")

    print(f"\n{'species':>9s} {'tracks':>7s} {'raw':>7s} {'viterbi':>8s}")
    for s in sorted({r[0] for r in rows}):
        m = np.array([r[0] == s for r in rows])
        w = n[m] / n[m].sum()
        print(f"{s:>9s} {m.sum():7d} {100*(raw[m]*w).sum():6.1f}% {100*(dec[m]*w).sum():7.1f}%")

    # THE test of the whole idea: if evidence accumulates, longer tracks must do better.
    print(f"\n=== DOES EVIDENCE ACCUMULATE?  (accuracy vs track length) ===")
    edges = [0, 10, 25, 50, 100, 10**9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (n >= lo) & (n < hi)
        if not m.any():
            continue
        w = n[m] / n[m].sum()
        print(f"  {lo:3d}-{min(hi,999):<3d} frames: {m.sum():3d} tracks   "
              f"raw {100*(raw[m]*w).sum():5.1f}%  ->  viterbi {100*(dec[m]*w).sum():5.1f}%")
    print("  If Viterbi climbs with track length, the errors really are decorrelating and the")
    print("  accumulation is doing what it should. If it is flat, the per-frame errors are")
    print("  SYSTEMATIC (the same animals wrong in every frame) and no decoder can fix that.")

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "tracks.npz",
             species=np.array([r[0] for r in rows]), track=np.array([r[1] for r in rows]),
             n=n, raw=raw, viterbi=dec)
    print(f"\nwrote {args.out}/tracks.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
