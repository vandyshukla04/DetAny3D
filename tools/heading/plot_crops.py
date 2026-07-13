"""Filmstrip of ONE track: every crop, with predicted vs GT heading drawn on it.

    python -m tools.heading.plot_crops --features data/heading/features.npz \
        --crops data/heading/crops.npz --track seg1::4 --out data/heading/track4.jpg

WHY THIS WORKS WITHOUT THE ORIGINAL FRAMES
------------------------------------------
`crops.npz` already holds every crop, and the crops were **square-padded** before resize --
which is a similarity transform, so it PRESERVES ANGLES. The image-space heading therefore
maps directly onto the crop, and we can draw it there. No frames, no cameras, no /mnt/d --
so this runs on the cluster where the features already live.

WHAT TO LOOK FOR
----------------
  GREY arrow = ground truth heading      RED arrow = prediction
  RED BORDER = a 180-degree flip (the failure that inverts the LEFT/RIGHT flank tag)

The run-length analysis (plot_tracks.py) tells you *that* a track fails in sustained blocks;
this tells you *why*. Scan the red-bordered crops: if the animal is head-on / tail-on, the
head-vs-tail cue is genuinely not visible and no amount of filtering or smoothing will
recover it -- that is an information limit, and the fix is a better cue (dense patch tokens,
the SAM3 mask as a channel), not a better filter.
"""
from __future__ import annotations

import argparse
import io
import math
from pathlib import Path

import numpy as np


def _label(track_key: str) -> str:
    """<video>/<seg> track <id>.  'seg1' alone is AMBIGUOUS -- several videos have a seg1,
    and printing just 'seg1 track 4' made two different animals look like one."""
    seg, tid = track_key.split("::")
    parts = Path(seg).parts
    return f"{parts[-2]}/{parts[-1]} track {tid}"


def draw_arrow(draw, cx, cy, angle_rad, length, colour, width):
    """Arrow from (cx, cy) along `angle_rad`. Image y is DOWN, which is the same convention
    the heading target was measured in -- so no sign flip is needed here."""
    ex = cx + length * math.cos(angle_rad)
    ey = cy + length * math.sin(angle_rad)
    draw.line([(cx, cy), (ex, ey)], fill=colour, width=width)
    for s in (+1, -1):                                   # arrowhead
        a = angle_rad + s * math.radians(150)
        draw.line([(ex, ey), (ex + 0.32 * length * math.cos(a),
                              ey + 0.32 * length * math.sin(a))], fill=colour, width=width)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--track", default=None,
                    help="substring of the track key, e.g. 'seg1::4'. Default: the WORST track.")
    ap.add_argument("--out", type=Path, default=Path("data/heading/track.jpg"))
    ap.add_argument("--cols", type=int, default=12)
    ap.add_argument("--cell", type=int, default=110)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from PIL import Image, ImageDraw

    d = np.load(args.features, allow_pickle=True)
    X, Y, tr, fr = d["X"], d["Y"], d["track"], d["frame"]
    cz = np.load(args.crops, allow_pickle=True)
    jpeg = cz["jpeg"]
    if len(jpeg) != len(X):
        print(f"WARNING: {len(jpeg)} crops vs {len(X)} features -- were they built together?")

    # Same track split as train_head / plot_tracks (seed 0), so "held out" means the same thing.
    tracks = np.unique(tr)
    rng = np.random.default_rng(0)
    rng.shuffle(tracks)
    test_tracks = set(tracks[: max(1, len(tracks) // 5)])
    te = np.array([t in test_tracks for t in tr])
    trn = ~te

    mu, sd = X[trn].mean(0, keepdims=True), X[trn].std(0, keepdims=True) + 1e-6
    from tools.heading.model import build_head
    net = build_head(X.shape[1], args.hidden).to(args.device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr = torch.tensor((X[trn] - mu) / sd, device=args.device)
    Ytr = torch.tensor(Y[trn], device=args.device)
    for _ in range(args.epochs):
        net.train(); opt.zero_grad()
        p = torch.nn.functional.normalize(net(Xtr), dim=-1)
        (1 - (p * Ytr).sum(-1)).mean().backward()
        opt.step()
    net.eval()

    with torch.no_grad():
        P = torch.nn.functional.normalize(
            net(torch.tensor((X - mu) / sd, device=args.device)), dim=-1).cpu().numpy()
    ang_p = np.arctan2(P[:, 1], P[:, 0])
    ang_t = np.arctan2(Y[:, 1], Y[:, 0])
    err = np.abs(np.degrees(np.arctan2(np.sin(ang_p - ang_t), np.cos(ang_p - ang_t))))

    # --- pick the track (NEVER silently disambiguate) --------------------------------
    # 'seg1' exists in several videos, so a substring like "seg1::4" matches MORE THAN ONE
    # track. Quietly taking cands[0] showed the wrong (clean) animal and made a failing
    # track look fine. Refuse to guess.
    if args.track:
        cands = [t for t in np.unique(tr) if args.track in t]
        if not cands:
            print(f"no track matching {args.track!r}. held-out tracks:")
            for t in sorted(test_tracks):
                print(f"    {_label(t)}   [{100*(err[tr==t]>90).mean():.0f}% flipped]")
            return 1
        if len(cands) > 1:
            print(f"AMBIGUOUS: {args.track!r} matches {len(cands)} tracks. Be specific:")
            for t in cands:
                print(f"    --track '{t}'    [{100*(err[tr==t]>90).mean():.0f}% flipped]")
            return 1
        key = cands[0]
    else:
        held = [t for t in np.unique(tr) if t in test_tracks]
        key = max(held, key=lambda t: (err[tr == t] > 90).mean())
        print("no --track given; showing the WORST held-out track")

    m = tr == key
    o = np.argsort(fr[m])
    idx = np.flatnonzero(m)[o][:: args.every]
    flips = err[idx] > 90
    print(f"track {_label(key)}   {len(idx)} frames   {100*flips.mean():.0f}% flipped"
          f"   ({'HELD OUT' if key in test_tracks else 'in TRAIN -- take with salt'})")

    cols = args.cols
    rows = int(np.ceil(len(idx) / cols))
    C = args.cell
    sheet = Image.new("RGB", (cols * C, rows * C), (18, 18, 18))

    for k, i in enumerate(idx):
        im = Image.open(io.BytesIO(jpeg[i])).convert("RGB").resize((C, C), Image.BICUBIC)
        dr = ImageDraw.Draw(im)
        c = C / 2
        draw_arrow(dr, c, c, float(ang_t[i]), C * 0.34, (190, 190, 190), 3)   # GT
        draw_arrow(dr, c, c, float(ang_p[i]), C * 0.30, (255, 40, 40), 2)     # prediction
        if err[i] > 90:
            dr.rectangle([0, 0, C - 1, C - 1], outline=(255, 0, 0), width=4)
        sheet.paste(im, ((k % cols) * C, (k // cols) * C))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, quality=92)
    print(f"wrote {args.out}")
    print("  GREY arrow = ground truth   RED arrow = prediction   RED BORDER = 180-deg flip")
    print("  If the red-bordered crops are HEAD-ON / TAIL-ON, the cue is simply not visible")
    print("  -> an information limit, not a filtering one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
