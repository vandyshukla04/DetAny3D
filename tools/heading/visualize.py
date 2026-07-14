"""SEE the predictions: predicted vs motion-truth heading, on HELD-OUT videos.  [cluster]

    python -m tools.heading.visualize \
        --features data/heading/features.npz --crops data/heading/crops.npz \
        --head data/heading/head.pt --out data/heading/vis --device cuda

Runs entirely on the cluster: `crops.npz` and the trained head are already there, and the crops
carry their own geometry, so NO frames and NO /mnt/d are needed.

WHAT IS DRAWN (one cell per animal, only from videos the model never trained on)
--------------------------------------------------------------------------------
  GREY arrow  : the TRUE front face -- where the animal actually walked (the motion label)
  RED arrow   : the PREDICTED front face -- the model's allocentric angle, snapped to the
                nearest of the box's 4 horizontal faces
  RED BORDER  : the two disagree. This is exactly the error the 4-way number counts, so the
                fraction of red borders you see IS the reported accuracy.
  caption     : allocentric angular error, in degrees

WHY SNAP TO A FACE RATHER THAN DRAW THE RAW ANGLE
-------------------------------------------------
The raw allocentric angle is a WORLD quantity; drawing it in the crop would need the camera,
which lives with the frames. The 4 face centres, by contrast, were already projected into crop
coordinates when the crop was cut. So snapping is both exact and honest: it draws precisely the
decision the metric scores, with no re-projection to get subtly wrong.

Also writes `preds.npz` (segment, track, frame, predicted alpha) so full frames can be rendered
locally later, where the imagery actually is.
"""
from __future__ import annotations

import argparse
import io
import math
from pathlib import Path

import numpy as np

from tools.heading.cropset import load_npz

from tools.heading.split import assert_matches, video_split
from tools.heading.train_head import face_from_alpha


def _arrow(draw, x0, y0, x1, y1, colour, width):
    draw.line([(x0, y0), (x1, y1)], fill=colour, width=width)
    ang = math.atan2(y1 - y0, x1 - x0)
    ln = 0.3 * math.hypot(x1 - x0, y1 - y0)
    for s in (+1, -1):
        a = ang + s * math.radians(150)
        draw.line([(x1, y1), (x1 + ln * math.cos(a), y1 + ln * math.sin(a))],
                  fill=colour, width=width)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--per-species", type=int, default=12)
    ap.add_argument("--cell", type=int, default=180)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from PIL import Image, ImageDraw

    from tools.heading.model import HEAD_VERSION, build_head

    d = load_npz(args.features)
    cr = load_npz(args.crops)
    X, Y, sp, vid = d["X"], d["Y"], d["species"], d["video"]
    y_face, face_uv, face_alpha = d["y_face"], d["face_uv"], d["face_alpha"]

    ck = torch.load(args.head, map_location=args.device, weights_only=False)
    if ck.get("head_version") != HEAD_VERSION:
        raise RuntimeError(f"head_version {ck.get('head_version')} != {HEAD_VERSION}; retrain")
    net = build_head(ck["in_dim"], ck["hidden"]).to(args.device).eval()
    net.load_state_dict(ck["state_dict"])

    te, test_v = video_split(sp, vid, seed=args.seed)
    # HARD FAILURE if this model was trained under a different split. Scoring a model on videos
    # it trained on reported 94.3% where the truth was 79.8% -- a believably good number that
    # nobody would have questioned.
    assert_matches(ck, test_v, seed=args.seed, what=str(args.head))
    print(f"held-out {len(test_v)} videos, {te.sum()} crops -- none seen in training (split verified)")

    with torch.no_grad():
        Z = torch.tensor((X[te] - ck["mu"]) / ck["sd"], device=args.device)
        pred = torch.nn.functional.normalize(net(Z), dim=-1).cpu().numpy()

    a_p = np.arctan2(pred[:, 1], pred[:, 0])
    a_t = np.arctan2(Y[te][:, 1], Y[te][:, 0])
    err = np.abs(np.degrees(np.arctan2(np.sin(a_p - a_t), np.cos(a_p - a_t))))
    f_p = face_from_alpha(a_p, face_alpha[te])
    f_t = y_face[te]
    idx = np.where(te)[0]

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "preds.npz", seg=d["seg"][te], track=d["track"][te],
             frame=d["frame"][te], alpha_pred=a_p.astype(np.float32),
             alpha_true=a_t.astype(np.float32), err_deg=err.astype(np.float32),
             face_pred=f_p.astype(np.int8), face_true=f_t.astype(np.int8),
             species=sp[te], video=vid[te])

    # --- contact sheet: sample across held-out VIDEOS, not just the first crops of one ----
    rng = np.random.default_rng(args.seed)
    species = sorted(set(sp[te].tolist()))
    C, cols = args.cell, args.per_species
    sheet = Image.new("RGB", (cols * C, len(species) * (C + 20)), (16, 16, 18))
    draw = ImageDraw.Draw(sheet)

    for r, s in enumerate(species):
        pool = np.where(sp[te] == s)[0]
        seen, chosen = set(), []
        for j in rng.permutation(pool):                    # spread over videos first
            v = str(vid[te][j])
            if v in seen and len(chosen) < cols:
                continue
            seen.add(v)
            chosen.append(int(j))
            if len(chosen) == cols:
                break

        for c, j in enumerate(chosen):
            g = int(idx[j])
            img = Image.open(io.BytesIO(cr["jpeg"][g])).convert("RGB").resize((C, C))
            x0, y0 = c * C, r * (C + 20)
            sheet.paste(img, (x0, y0))

            uv = face_uv[g] * C
            cx, cy = uv.mean(axis=0)                       # box centre = mean of opposite faces
            t, p = int(f_t[j]), int(f_p[j])
            _arrow(draw, x0 + cx, y0 + cy, x0 + uv[t, 0], y0 + uv[t, 1], (170, 170, 170), 4)
            _arrow(draw, x0 + cx, y0 + cy, x0 + uv[p, 0], y0 + uv[p, 1], (255, 50, 50), 3)
            if t != p:
                draw.rectangle([x0, y0, x0 + C - 1, y0 + C - 1], outline=(255, 40, 40), width=4)
            draw.text((x0 + 4, y0 + C + 3), f"{s[:8]} err {err[j]:.0f}deg",
                      fill=(210, 210, 210))

    sheet.save(args.out / "predictions.jpg", quality=92)

    print(f"\nwrote {args.out}/predictions.jpg  and  {args.out}/preds.npz")
    print(f"  GREY = true front (motion)   RED = predicted   RED BORDER = disagreement")
    print(f"\n  held-out 4-way face accuracy: {100*(f_p == f_t).mean():.1f}%  (chance 25%)")
    for s in species:
        m = sp[te] == s
        print(f"    {s:>9s}: {100*(f_p[m] == f_t[m]).mean():5.1f}%   "
              f"median err {np.median(err[m]):4.0f} deg   ({m.sum()} crops)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
