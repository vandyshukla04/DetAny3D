"""THE PROBE: do DINOv3 patch tokens contain a HEAD part, and does it find the animal's front?

    python -m tools.heading.parts --patches data/heading/patches.npz \
        --out data/heading/parts --device cuda

THE QUESTION
------------
Geometry already gives us the animal's body axis (from the 3D box) -- what it can never give
is WHICH END IS THE HEAD, because the box's PCA/SVD axis signs are arbitrary. So the entire
heading problem reduces to picking the front face out of the 4 horizontal candidates.

Can DINOv3 do that from a SINGLE frame, with NO motion and NO training?

Its headline property is dense semantic part correspondence -- the head of one zebra matches
the head of another. If that holds here, one k-means cluster over patch tokens IS "head", and
its centroid lands on the animal's front. That is the claim, and this script measures it.

Note this is a *different mechanism* from open-vocabulary detection, which is why GroundingDINO's
failure on a "head" prompt (domain shift -- it grounds onto the whole animal) does not predict
the outcome here.

HOW IT IS SCORED -- FOR FREE, WITH ZERO HUMAN LABELS
----------------------------------------------------
The probe runs on crops of WALKING animals, where world velocity already told us which face is
the head (autolabel.py; verified by eye in preview_labels.py). So:

    accuracy = how often the head-cluster's centroid is nearest the TRUE front face.
    chance   = 25%   (4 horizontal candidates)

CALIBRATION IS 4 BITS, AND IT IS HELD OUT
-----------------------------------------
"Which cluster is the head" is ONE choice per species -- 4 bits for the whole dataset. It is
made on the calibration VIDEOS and then frozen, and the accuracy is reported on videos the
calibration never saw. Splitting by video (not by track, not by frame) is what stops a model
scoring well on scene cues alone; a track-split once said 96.7% while the thing visibly broke
on new footage.

GO / NO-GO
----------
  >>25%  -> DINOv3 parts carry the head. Build the heading on them; no locomotion needed.
  ~25%   -> they do not. Say so, and pivot -- exactly as we did with GroundingDINO and the
            mask-taper cue, both of which we killed with a measurement like this one.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np

__all__ = ["kmeans", "patch_xy", "cluster_centroids"]


def kmeans(X, k: int, iters: int = 40, seed: int = 0, device: str = "cpu"):
    """Spherical k-means on L2-normalised tokens. Torch-only -- sklearn is not on the cluster."""
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g)[:k]].clone()
    for _ in range(iters):
        a = (X @ C.T).argmax(1)                       # cosine assignment (X, C are unit-norm)
        for j in range(k):
            m = a == j
            if m.any():
                C[j] = torch.nn.functional.normalize(X[m].mean(0), dim=-1)
    return C, (X @ C.T).argmax(1)


def patch_xy(gh: int, gw: int) -> np.ndarray:
    """Centre of each patch, in the crop's [0,1]^2 coordinates -- the same frame as `face_uv`."""
    r, c = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    return np.stack([(c + 0.5) / gw, (r + 0.5) / gh], axis=-1).reshape(-1, 2)


def cluster_centroids(assign: np.ndarray, xy: np.ndarray, k: int) -> np.ndarray:
    """(k, 2) centroid of each cluster's patches in crop coords; NaN where the cluster is absent."""
    out = np.full((k, 2), np.nan)
    for j in range(k):
        m = assign == j
        if m.any():
            out[j] = xy[m].mean(0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patches", type=Path, required=True)
    ap.add_argument("--crops", type=Path, default=None, help="only needed for --viz")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--fit-tokens", type=int, default=300_000)
    ap.add_argument("--border-reject", type=float, default=0.35,
                    help="a cluster with more than this fraction of its patches on the rim of "
                         "the grid is BACKGROUND, not a body part. On a 14x14 grid the rim is "
                         "27%% of patches, so a uniform cluster sits at 0.27 -- which is why the "
                         "old default of 0.55 could never fire.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--viz", action="store_true", help="dump a part-segmentation contact sheet")
    args = ap.parse_args()

    import torch

    d = np.load(args.patches, allow_pickle=True)
    P = d["P"]                                        # (n, gh, gw, D) fp16
    gh, gw = (int(x) for x in d["grid"])
    n, D = len(P), P.shape[-1]
    face_uv, y_face = d["face_uv"], d["y_face"]
    species, video = d["species"], d["video"]
    print(f"{n} crops, {gh}x{gw} patch grid, {D}-d tokens, "
          f"species={dict(zip(*np.unique(species, return_counts=True)))}")

    # ---- split by VIDEO: calibrate the 4 bits on one set, report on the other ----------
    # Shared with train_head.py so the two methods are scored on the SAME held-out videos.
    from tools.heading.split import video_split
    te, test_v = video_split(species, video, seed=args.seed)
    cal = ~te
    print(f"calibration {cal.sum()} crops / {len(np.unique(video)) - len(test_v)} videos   |   "
          f"HELD-OUT {te.sum()} crops / {len(test_v)} videos")

    # ---- one k-means over all patch tokens --------------------------------------------
    X = torch.from_numpy(P.reshape(-1, D)).float()
    X = torch.nn.functional.normalize(X, dim=-1).to(args.device)
    fit = X[torch.randperm(len(X))[: args.fit_tokens]]
    print(f"k-means k={args.k} on {len(fit)} of {len(X)} tokens ...", flush=True)
    C, _ = kmeans(fit, args.k, seed=args.seed, device=args.device)
    assign = (X @ C.T).argmax(1).cpu().numpy().reshape(n, gh * gw)

    # ---- reject BACKGROUND clusters, label-free ---------------------------------------
    xy = patch_xy(gh, gw)
    border = ((xy[:, 0] < 1.0 / gw) | (xy[:, 0] > 1 - 1.0 / gw) |
              (xy[:, 1] < 1.0 / gh) | (xy[:, 1] > 1 - 1.0 / gh))
    B = np.tile(border, (n, 1))                       # (n, gh*gw): is this patch on the rim?
    brate = np.array([float(B[assign == j].mean()) if (assign == j).any() else 1.0
                      for j in range(args.k)])
    # A cluster spread UNIFORMLY over the grid sits at exactly this border rate. Anything well
    # above it is hugging the rim (= background); well below it is animal-core. Comparing
    # against the grid's own baseline is what makes the threshold meaningful -- an absolute
    # cut-off like 0.55 can never fire on a 14x14 grid, where the rim is only 27% of patches.
    base = float(border.mean())
    is_bg = brate > args.border_reject
    print(f"\n  (a uniformly-spread cluster would have border-rate {100*base:.0f}% on this "
          f"{gh}x{gw} grid)")
    print("cluster   share   border-rate")
    for j in range(args.k):
        print(f"  {j}    {100*float((assign == j).mean()):5.1f}%   {100*brate[j]:5.1f}%"
              f"{'   <- BACKGROUND (rejected)' if is_bg[j] else ''}")
    parts = [j for j in range(args.k) if not is_bg[j]]
    if not parts:
        print("\nevery cluster looks like background -- the crops or --border-reject are wrong")
        return 1

    # ---- for every crop, every cluster's centroid -> which of the 4 faces is it nearest? ----
    cent = np.stack([cluster_centroids(assign[i], xy, args.k) for i in range(n)])   # (n, k, 2)
    dist = np.linalg.norm(cent[:, :, None, :] - face_uv[:, None, :, :], axis=-1)    # (n, k, 4)

    # A cluster can be ABSENT from a crop (no patches assigned) -> its whole row is NaN.
    # np.where does NOT short-circuit, so calling nanargmin on those rows raises
    # "All-NaN slice encountered". Fill with +inf and use a plain argmin instead.
    present = ~np.isnan(dist).all(-1)                                               # (n, k)
    nearest = np.where(np.isnan(dist), np.inf, dist).argmin(-1)                     # (n, k)
    correct = (nearest == y_face[:, None]) & present    # absent => counted WRONG, not skipped:
                                                        # a part that vanishes half the time is
                                                        # not a usable head detector.

    # ---- calibrate: which cluster IS the head? one choice per species (4 bits total) ----
    print(f"\n=== PART -> FRONT FACE   (chance 25%) ===")
    print(f"{'species':>9s} {'head':>5s} {'calib':>7s} {'HELD-OUT':>9s} {'present':>8s} "
          f"{'oracle':>7s} {'n':>6s}")
    rows = []
    for s in sorted(set(species.tolist())):
        m_cal, m_te = cal & (species == s), te & (species == s)
        if m_cal.sum() < 20 or m_te.sum() < 20:
            print(f"{s:>9s}   too few crops (cal {m_cal.sum()}, test {m_te.sum()})")
            continue
        scores = {j: float(correct[m_cal, j].mean()) for j in parts}
        head = max(scores, key=scores.get)                     # SELECTED on calibration only
        acc = float(correct[m_te, head].mean())                # SCORED on held-out videos
        # Oracle = the best cluster had we been allowed to peek at the test set. If oracle is
        # far above the selected one, the calibration does not transfer and the "head cluster"
        # is not a stable, species-level concept -- which is itself the finding.
        oracle = max(float(correct[m_te, j].mean()) for j in parts)
        pres = float(present[m_te, head].mean())
        rows.append((s, head, scores[head], acc, int(m_te.sum())))
        print(f"{s:>9s} {head:5d} {100*scores[head]:6.1f}% {100*acc:8.1f}% {100*pres:7.0f}% "
              f"{100*oracle:6.1f}% {m_te.sum():6d}")

    if rows:
        w = np.array([r[4] for r in rows], dtype=float)
        overall = float(np.average([r[3] for r in rows], weights=w))
        print(f"{'ALL':>9s} {'':13s} {'':7s} {100*overall:8.1f}%  <- the number that decides this")
        print(f"\n  chance = 25.0%")
        if overall > 0.60:
            print("  => DINOv3 parts DO carry the head. Build the heading on them; no locomotion.")
        elif overall > 0.35:
            print("  => a real but weak signal. Worth a supervised head on top of the parts.")
        else:
            print("  => NO usable head part. Say so and pivot -- do not paper over this.")

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "parts.npz", C=C.cpu().numpy(), assign=assign,
             is_bg=is_bg, border_rate=brate,
             head_cluster=np.array([r[1] for r in rows]),
             head_species=np.array([r[0] for r in rows]),
             heldout_acc=np.array([r[3] for r in rows]))
    print(f"\nwrote {args.out}/parts.npz")

    # ---- SEE the parts. A number can lie; a picture of the segmentation cannot. --------
    if args.viz and args.crops:
        from PIL import Image
        cr = np.load(args.crops, allow_pickle=True)
        jp, idx = cr["jpeg"], d["idx"]
        pal = np.array([[230, 60, 60], [60, 200, 90], [70, 130, 240], [240, 190, 60],
                        [190, 90, 220], [60, 210, 210], [240, 130, 60], [150, 150, 150]])
        heads = {r[0]: r[1] for r in rows}
        cells, cap = [], []
        for s in sorted(heads):
            got = [i for i in np.where((species == s) & te)[0]][:6]
            for i in got:
                base = Image.open(io.BytesIO(jp[int(idx[i])])).convert("RGB").resize((160, 160))
                seg = Image.fromarray(
                    pal[assign[i].reshape(gh, gw) % len(pal)].astype(np.uint8)
                ).resize((160, 160), Image.NEAREST)
                hm = Image.fromarray(
                    (255 * (assign[i].reshape(gh, gw) == heads[s])).astype(np.uint8)
                ).resize((160, 160), Image.NEAREST).convert("RGB")
                cells.append(Image.blend(base, seg, 0.55))
                cells.append(Image.blend(base, hm, 0.5))
                cap.append(s)
        if cells:
            cols = 12
            rows_n = (len(cells) + cols - 1) // cols
            sheet = Image.new("RGB", (cols * 160, rows_n * 160), (20, 20, 22))
            for i, c in enumerate(cells):
                sheet.paste(c, (160 * (i % cols), 160 * (i // cols)))
            sheet.save(args.out / "parts_viz.jpg", quality=90)
            print(f"wrote {args.out}/parts_viz.jpg   (pairs: all-parts | HEAD cluster only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
