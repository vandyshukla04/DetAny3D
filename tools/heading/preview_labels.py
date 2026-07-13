"""Contact sheet of the MOTION-DERIVED labels. No model, no GPU, no ground truth.

    python -m tools.heading.preview_labels --crops data/heading/crops.npz \
        --out data/heading/labels_preview.jpg --per-species 8

THIS IS THE STEP WE SKIPPED LAST TIME AND PAID FOR.
An earlier run reported a confident "91.8% flank accuracy" from a model that had never seen
an animal -- every crop had been cut from the background, because `bbox_2d` was read as
full-res when it is in 518-space. The numbers looked fine. Nobody looked at the pixels.

So: look at the pixels.

WHAT IS DRAWN
-------------
  GREEN dot + arrow : the front face the animal is WALKING TOWARD -- i.e. the head, per the
                      motion label. The arrow runs from the box centre to that face.
  GREY dots         : the other 3 horizontal face candidates.
  caption           : species, on-screen size, and cos(velocity, face normal) -- the
                      independent box-vs-motion agreement for this instance.

WHAT TO LOOK FOR
----------------
  1. Is there an ANIMAL in the crop at all? (the bug above)
  2. Does the GREEN arrow point at its HEAD?
If (2) holds across species, the motion labels are sound and they can be trusted to score a
heading predictor. If it does not, everything downstream is measuring noise -- and we find
that out here, for free, before a single GPU-hour is spent.
"""
from __future__ import annotations

import argparse
import io
import math
from pathlib import Path

import numpy as np


def _arrow(draw, x0, y0, x1, y1, colour, width):
    draw.line([(x0, y0), (x1, y1)], fill=colour, width=width)
    ang = math.atan2(y1 - y0, x1 - x0)
    ln = 0.28 * math.hypot(x1 - x0, y1 - y0)
    for s in (+1, -1):
        a = ang + s * math.radians(150)
        draw.line([(x1, y1), (x1 + ln * math.cos(a), y1 + ln * math.sin(a))],
                  fill=colour, width=width)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-species", type=int, default=8)
    ap.add_argument("--cell", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from PIL import Image, ImageDraw

    d = np.load(args.crops, allow_pickle=True)
    sp, jpeg = d["species"], d["jpeg"]
    face_uv, y_face, body_px = d["face_uv"], d["y_face"], d["body_px"]

    # sample across DIFFERENT VIDEOS, not just the first few crops of one flight
    rng = np.random.default_rng(args.seed)
    picks: list[int] = []
    species = sorted(set(sp.tolist()))
    for s in species:
        idx = np.where(sp == s)[0]
        vids = d["video"][idx]
        chosen, seen = [], set()
        for j in rng.permutation(idx):                 # one crop per video, then fill up
            v = str(d["video"][j])
            if v in seen and len(chosen) < args.per_species:
                continue
            seen.add(v)
            chosen.append(int(j))
            if len(chosen) == args.per_species:
                break
        picks.append(chosen)

    C, cols = args.cell, args.per_species
    sheet = Image.new("RGB", (cols * C, len(species) * (C + 22)), (18, 18, 20))
    draw = ImageDraw.Draw(sheet)

    for r, (s, chosen) in enumerate(zip(species, picks)):
        for c, j in enumerate(chosen):
            img = Image.open(io.BytesIO(jpeg[j].tobytes() if isinstance(jpeg[j], np.ndarray)
                                        else jpeg[j])).convert("RGB").resize((C, C))
            x0, y0 = c * C, r * (C + 22)
            sheet.paste(img, (x0, y0))

            uv = face_uv[j] * C                        # (4, 2) crop coords -> cell pixels
            k = int(y_face[j])
            cx, cy = uv.mean(axis=0)                   # box centre ~= mean of opposite faces
            for m in range(4):                         # the 3 rejected candidates
                if m == k:
                    continue
                px, py = x0 + uv[m, 0], y0 + uv[m, 1]
                draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(150, 150, 150))
            _arrow(draw, x0 + cx, y0 + cy, x0 + uv[k, 0], y0 + uv[k, 1], (40, 255, 90), 3)
            draw.ellipse([x0 + uv[k, 0] - 5, y0 + uv[k, 1] - 5,
                          x0 + uv[k, 0] + 5, y0 + uv[k, 1] + 5], fill=(40, 255, 90))
            draw.text((x0 + 4, y0 + C + 4), f"{s[:8]} {body_px[j]:.0f}px", fill=(200, 200, 200))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, quality=92)
    print(f"wrote {args.out}")
    print("  GREEN arrow -> the face the animal walks toward = its HEAD, per the motion label.")
    print("  Check: (1) is there an animal? (2) does the arrow point at its head?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
