"""BASELINE: a small supervised head on frozen DINOv3, trained on the motion labels.  [GPU]

    python -m tools.heading.train_head --features data/heading/features.npz \
        --save data/heading/head.pt --device cuda

This is the COMPARISON POINT for the part probe (parts.py), not the proposed method. If the
training-free part cue matches or beats it, we get the heading with no training and no
dependence on locomotion at all.

WHAT IT PREDICTS: THE ALLOCENTRIC ANGLE, NOT AN IMAGE ANGLE
-----------------------------------------------------------
A crop determines the animal's orientation RELATIVE TO THE VIEWING RAY. The same animal at the
left and right edges of a frame looks identical but projects to different image angles -- so
regressing an image angle asks the network to infer something the pixels do not contain, and
conflates appearance with where the animal happened to sit in frame. (This was the flaw in the
earlier version.)

So the target is alpha: the heading measured in the ground-plane basis (r = horizontalised
camera->animal ray, s = up x r). It is view-invariant and purely appearance-determined -- and
it is exactly DetAny3D's `alpha` head, which already exists, already has a live loss, and is
currently fed a hardcoded 0.0.

HOW IT IS SCORED: THE SAME 4-WAY QUESTION AS THE PROBE
------------------------------------------------------
A predicted alpha is snapped to the nearest of the box's 4 horizontal faces, and we ask
whether that is the face the animal actually walks toward. Identical crops, identical metric,
chance 25% -- so `parts.py` and this script produce two numbers that can be put side by side.
We also report the raw allocentric angular error, which is the quantity that matters at
inference (where no box, and hence no 4 candidates, may exist).

SPLIT BY VIDEO, NEVER BY TRACK OR FRAME
---------------------------------------
Every track in a video shares one flight, altitude, light and herd, so holding out a *track*
leaves its whole scene in training. Measured: a track-split reported 96.7% while the model
visibly broke on unseen footage. Videos, or nothing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def face_from_alpha(alpha: np.ndarray, face_alpha: np.ndarray) -> np.ndarray:
    """Snap a predicted allocentric angle to the nearest of the 4 candidate front faces."""
    d = np.abs(np.arctan2(np.sin(alpha[:, None] - face_alpha),
                          np.cos(alpha[:, None] - face_alpha)))
    return d.argmin(axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=Path, default=None)
    ap.add_argument("--holdout-species", default=None,
                    help="train on the other species and test on this one (the transfer test)")
    args = ap.parse_args()

    import torch

    d = np.load(args.features, allow_pickle=True)
    X, Y = d["X"], d["Y"]                                 # Y = (cos alpha, sin alpha)
    sp, vid = d["species"], d["video"]
    y_face, face_alpha = d["y_face"], d["face_alpha"]
    print(f"{len(X)} samples, {X.shape[1]}-d, "
          f"species={dict(zip(*np.unique(sp, return_counts=True)))}, "
          f"{len(np.unique(vid))} videos")

    # --- HOLD OUT WHOLE VIDEOS ---------------------------------------------------------
    # The split lives in split.py so parts.py scores on the SAME videos -- otherwise the
    # training-free and supervised numbers are not comparable, and the drift is silent.
    if args.holdout_species:
        te = sp == args.holdout_species
        print(f"holding out species={args.holdout_species}: {te.sum()} test / {(~te).sum()} train")
    else:
        from tools.heading.split import video_split
        te, test_v = video_split(sp, vid, seed=args.seed)
        print(f"held-out {len(test_v)}/{len(np.unique(vid))} VIDEOS (stratified by species): "
              f"{te.sum()} test / {(~te).sum()} train")
    tr = ~te
    if tr.sum() == 0 or te.sum() == 0:
        print("empty split")
        return 1

    mu, sd = X[tr].mean(0, keepdims=True), X[tr].std(0, keepdims=True) + 1e-6
    Xtr = torch.tensor((X[tr] - mu) / sd, device=args.device)
    Ytr = torch.tensor(Y[tr], device=args.device)
    Xte = torch.tensor((X[te] - mu) / sd, device=args.device)

    from tools.heading.model import HEAD_VERSION, build_head
    torch.manual_seed(args.seed)
    net = build_head(X.shape[1], args.hidden).to(args.device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    n = Xtr.shape[0]
    spe = max(1, n // args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * spe)
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(n, device=args.device)
        tot = 0.0
        for b in range(spe):
            sel = perm[b * args.batch: (b + 1) * args.batch]
            opt.zero_grad()
            p = torch.nn.functional.normalize(net(Xtr[sel]), dim=-1)
            loss = (1 - (p * Ytr[sel]).sum(-1)).mean()      # cosine loss = angular error
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
        if ep % 15 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  loss {tot/spe:.4f}")

    net.eval()
    with torch.no_grad():
        pred = torch.nn.functional.normalize(net(Xte), dim=-1).cpu().numpy()

    a_p = np.arctan2(pred[:, 1], pred[:, 0])
    a_t = np.arctan2(Y[te][:, 1], Y[te][:, 0])
    err = np.abs(np.degrees(np.arctan2(np.sin(a_p - a_t), np.cos(a_p - a_t))))
    face_p = face_from_alpha(a_p, face_alpha[te])
    face_ok = face_p == y_face[te]

    print("\n=== HELD-OUT VIDEOS ===")
    print(f"  allocentric angular error : median {np.median(err):5.1f} deg   "
          f"(>90 deg on {100*(err>90).mean():.1f}%)")
    print(f"  FRONT FACE (4-way)        : {100*face_ok.mean():5.1f}%   <- compare with parts.py")
    print(f"  chance                    :  25.0%")
    print(f"\n{'species':>9s} {'n':>6s} {'median err':>11s} {'4-way face':>11s}")
    for s in sorted(set(sp[te].tolist())):
        m = sp[te] == s
        print(f"{s:>9s} {m.sum():6d} {np.median(err[m]):10.1f} {100*face_ok[m].mean():10.1f}%")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": net.state_dict(), "in_dim": int(X.shape[1]),
                    "hidden": args.hidden, "head_version": HEAD_VERSION,
                    "mu": mu.astype(np.float32), "sd": sd.astype(np.float32),
                    "target": "allocentric_alpha"}, args.save)
        print(f"\nsaved -> {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
