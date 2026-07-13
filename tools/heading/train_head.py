"""Train the small heading head on frozen DINOv3 features.

    python -m tools.heading.train_head --features data/heading/features.npz

Predicts the animal's heading as a point on the unit circle (cos, sin) in image space,
then reports the number that actually matters for re-ID:

    FLANK accuracy -- are we looking at the animal's LEFT or RIGHT side?

WHY FLANK IS THE HEADLINE METRIC
--------------------------------
Wildlife re-ID is viewpoint-dependent: a zebra's left stripes are a different pattern
from its right. Matching a left-flank query against a right-flank gallery is a guaranteed
miss. The flank label flips exactly when the heading flips by 180 deg, so flank accuracy
is a direct, task-level read on whether the heading is good enough to be useful.

A 180-deg error is the failure that matters; a 20-deg error is harmless. So we report
the flip rate, not just the mean angular error.

SPLIT BY TRACK, NEVER BY FRAME
------------------------------
Adjacent frames of one animal are near-duplicates. A random frame split would leak the
answer across the split and report a beautiful, meaningless number. We hold out whole
TRACKS -- and, when there is more than one species, whole species too, since that is the
generalisation question we actually care about.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def flank_of(angle: np.ndarray) -> np.ndarray:
    """Which flank faces the camera, given the heading angle in image space.

    With the image y-axis pointing DOWN, an animal heading to image-right (angle ~0) is
    seen from its LEFT side. The flank therefore flips with the sign of sin(angle)... but
    the honest, geometry-free statement is: the flank is determined by which half-plane
    the heading points into, so `cos(angle) >= 0` is a *consistent* binary flank label.
    Consistency is all we need -- an overall naming flip is a single global convention,
    which `faces.py` fixes exactly via `left = up x forward`.
    """
    return (np.cos(angle) >= 0).astype(int)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--holdout-species", default=None,
                    help="train on everything else, test on this species (the gap test)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    d = np.load(args.features, allow_pickle=True)
    X, Y, sp, tr = d["X"], d["Y"], d["species"], d["track"]
    print(f"{len(X)} samples, {X.shape[1]}-d features, species={dict(zip(*np.unique(sp, return_counts=True)))}")

    # --- split by TRACK (never by frame: adjacent frames are near-duplicates) ---
    if args.holdout_species:
        te = sp == args.holdout_species
        tr_mask = ~te
        print(f"holding out species={args.holdout_species}: {te.sum()} test / {tr_mask.sum()} train")
    else:
        tracks = np.unique(tr)
        rng = np.random.default_rng(0)
        rng.shuffle(tracks)
        test_tracks = set(tracks[: max(1, len(tracks) // 5)])
        te = np.array([t in test_tracks for t in tr])
        tr_mask = ~te
        print(f"held-out {len(test_tracks)}/{len(tracks)} TRACKS: {te.sum()} test / {tr_mask.sum()} train")

    if te.sum() == 0 or tr_mask.sum() == 0:
        print("empty split; nothing to do")
        return 1

    mu, sd = X[tr_mask].mean(0, keepdims=True), X[tr_mask].std(0, keepdims=True) + 1e-6
    Xtr = torch.tensor((X[tr_mask] - mu) / sd, device=args.device)
    Ytr = torch.tensor(Y[tr_mask], device=args.device)
    Xte = torch.tensor((X[te] - mu) / sd, device=args.device)
    Yte = Y[te]

    net = nn.Sequential(
        nn.Linear(X.shape[1], args.hidden), nn.GELU(), nn.Dropout(0.2),
        nn.Linear(args.hidden, 2),
    ).to(args.device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    for ep in range(args.epochs):
        net.train()
        opt.zero_grad()
        p = torch.nn.functional.normalize(net(Xtr), dim=-1)   # onto the unit circle
        loss = (1 - (p * Ytr).sum(-1)).mean()                 # cosine loss on the angle
        loss.backward()
        opt.step()
        if ep % 20 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  loss {loss.item():.4f}")

    net.eval()
    with torch.no_grad():
        pred = torch.nn.functional.normalize(net(Xte), dim=-1).cpu().numpy()

    ang_p = np.arctan2(pred[:, 1], pred[:, 0])
    ang_t = np.arctan2(Yte[:, 1], Yte[:, 0])
    err = np.abs(np.degrees(np.arctan2(np.sin(ang_p - ang_t), np.cos(ang_p - ang_t))))
    flank_acc = (flank_of(ang_p) == flank_of(ang_t)).mean()

    print("\n=== HELD-OUT (per frame) ===")
    print(f"  median angular error : {np.median(err):6.1f} deg")
    print(f"  180-deg flips (>90)  : {100*(err>90).mean():6.1f}%   <- the failure that matters")
    print(f"  FLANK accuracy       : {100*flank_acc:6.1f}%   <- the re-ID number")
    print(f"  chance               :   50.0%")

    # --- Are the flips SCATTERED across frames, or concentrated in whole TRACKS? -----
    # This decides everything. An animal's head does not swap ends mid-track, so if the
    # flips are scattered, a track-level majority vote erases them. If instead entire
    # tracks are confidently backwards, voting is useless and the cue itself is wrong.
    te_tracks = tr[te]
    print("\n=== per-track flip rate (held-out) ===")
    per_track = []
    for t in np.unique(te_tracks):
        m = te_tracks == t
        fr = float((err[m] > 90).mean())
        per_track.append(fr)
        name = t.split("::")[-1]
        seg = t.split("/")[-1].split("::")[0]
        flag = "  <-- WHOLE TRACK BACKWARDS" if fr > 0.5 else ""
        print(f"  {seg:>18s} track {name:>3s}: {m.sum():4d} frames, {100*fr:5.1f}% flipped{flag}")

    per_track = np.array(per_track)
    fully = int((per_track > 0.5).sum())
    print(f"\n  tracks >50% flipped : {fully}/{len(per_track)}")

    # Track-level majority vote: one heading decision per animal, not per frame.
    voted_ok = 0
    for t in np.unique(te_tracks):
        m = te_tracks == t
        pred_flank = np.bincount(flank_of(ang_p[m]), minlength=2).argmax()
        true_flank = np.bincount(flank_of(ang_t[m]), minlength=2).argmax()
        voted_ok += int(pred_flank == true_flank)
    print(f"  FLANK accuracy after track-vote : {100*voted_ok/len(np.unique(te_tracks)):5.1f}% "
          f"({voted_ok}/{len(np.unique(te_tracks))} tracks)")
    if fully:
        print("  NOTE: voting cannot rescue a track that is wholly backwards -- those need a better cue.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
