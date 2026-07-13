"""Plot predicted vs ground-truth heading over time, per held-out track.

    python -m tools.heading.plot_tracks --features data/heading/features.npz \
        --out data/heading/tracks.png

THE QUESTION THIS ANSWERS
-------------------------
Aggregate flip rates cannot tell you *how* the model fails. Two very different worlds hide
behind "3% flipped":

  * **scattered** single-frame flips  -> a temporal filter erases them; the heading is fine.
  * **contiguous runs** of flips      -> the model is confidently backwards for a stretch of
                                         time. NO temporal filter can fix that (there is
                                         nothing to smooth toward), and chasing one is a waste.

The plot shows the GT heading (grey) and the prediction (blue) over frame index, with flipped
frames marked in red. Look at the red: speckle = fixable noise; solid blocks = a real failure
mode that needs a better cue, not a better filter.

Note the y-axis is IMAGE-space heading, which legitimately drifts as the drone moves -- so do
not read a slow drift as error. Only the red marks are errors.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/heading/tracks.png"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    # The run-length analysis is the part that actually decides anything, and it needs no
    # plotting. matplotlib is optional so this still works in the bare `dinov3` env.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        plt = None
        print("(matplotlib not installed -- skipping the PNG, printing the analysis only)\n")

    d = np.load(args.features, allow_pickle=True)
    X, Y, tr, fr = d["X"], d["Y"], d["track"], d["frame"]

    # same split as train_head: by TRACK, seed 0
    tracks = np.unique(tr)
    rng = np.random.default_rng(0)
    rng.shuffle(tracks)
    test_tracks = set(tracks[: max(1, len(tracks) // 5)])
    te = np.array([t in test_tracks for t in tr])
    trn = ~te

    mu, sd = X[trn].mean(0, keepdims=True), X[trn].std(0, keepdims=True) + 1e-6
    Xtr = torch.tensor((X[trn] - mu) / sd, device=args.device)
    Ytr = torch.tensor(Y[trn], device=args.device)

    net = nn.Sequential(nn.Linear(X.shape[1], args.hidden), nn.GELU(), nn.Dropout(0.2),
                        nn.Linear(args.hidden, 2)).to(args.device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(args.epochs):
        net.train(); opt.zero_grad()
        p = torch.nn.functional.normalize(net(Xtr), dim=-1)
        (1 - (p * Ytr).sum(-1)).mean().backward()
        opt.step()

    net.eval()
    with torch.no_grad():
        P = torch.nn.functional.normalize(
            net(torch.tensor((X[te] - mu) / sd, device=args.device)), dim=-1).cpu().numpy()

    ang_p = np.degrees(np.arctan2(P[:, 1], P[:, 0]))
    ang_t = np.degrees(np.arctan2(Y[te][:, 1], Y[te][:, 0]))
    err = np.abs((ang_p - ang_t + 180) % 360 - 180)
    te_tr, te_fr = tr[te], fr[te]

    uniq = sorted(np.unique(te_tr))

    # --- THE DECISIVE ANALYSIS (no plotting needed) ------------------------------------
    print("=== are the flips SCATTERED noise, or CONTIGUOUS runs? ===")
    all_runs: list[int] = []
    for t in uniq:
        m = te_tr == t
        o = np.argsort(te_fr[m])
        flip = (err[m][o] > 90).astype(int)
        seg = t.split("/")[-1].split("::")[0]
        tid = t.split("::")[-1]
        if not flip.any():
            print(f"  {seg:>6s} track {tid:>3s}: clean")
            continue
        runs, cur = [], 0
        for v in flip:
            if v:
                cur += 1
            elif cur:
                runs.append(cur); cur = 0
        if cur:
            runs.append(cur)
        all_runs += runs
        verdict = "SCATTERED (a filter fixes this)" if max(runs) <= 3 else \
                  "CONTIGUOUS -- confidently backwards; NO filter can fix this"
        print(f"  {seg:>6s} track {tid:>3s}: {flip.sum():3d} flipped in {len(runs):2d} run(s), "
              f"longest={max(runs):3d} frames   {verdict}")

    if all_runs:
        a = np.array(all_runs)
        print(f"\n  runs: n={len(a)}  median={np.median(a):.0f}  longest={a.max()}  "
              f"| {100*(a<=3).mean():.0f}% of runs are <=3 frames")
        print("  longest ~1-3  => scattered noise; a (world-space) temporal filter will fix it")
        print("  longest  >20  => a real visibility failure; needs a better cue, not a filter")

    # --- optional plot -----------------------------------------------------------------
    if plt is None:
        return 0

    n = len(uniq)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 2.6 * rows), squeeze=False)
    for k, t in enumerate(uniq):
        ax = axes[k // cols][k % cols]
        m = te_tr == t
        o = np.argsort(te_fr[m])
        f = te_fr[m][o]
        gt, pr, e = ang_t[m][o], ang_p[m][o], err[m][o]
        flip = e > 90
        ax.plot(f, gt, color="0.7", lw=2.5, label="ground truth")
        ax.plot(f, pr, color="tab:blue", lw=1.0, label="predicted")
        if flip.any():
            ax.scatter(f[flip], pr[flip], color="red", s=14, zorder=5, label="180-deg flip")
        ax.set_title(f"{t.split('/')[-1].split('::')[0]} track {t.split('::')[-1]}"
                     f"   ({100*flip.mean():.0f}% flipped)", fontsize=9)
        ax.set_ylim(-190, 190)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.legend(fontsize=7, loc="lower left")
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle("Predicted vs GT heading per held-out track — RED = 180-deg flip. "
                 "Speckle => filterable.  Solid blocks => confidently backwards.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
