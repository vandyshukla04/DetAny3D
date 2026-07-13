"""Render the heading + flank tag on an UNLABELLED segment.  NO 3D BOX.

    python -m tools.heading.predict \
        --segment /storage3/.../WildBox/DJI_2024.../seg5 \
        --head    data/heading/head.pt \
        --out     data/heading/vis/seg5 \
        --every 10 --device cuda

Draws, per animal:
  * the 2D detection box (this is what the crop came from -- it IS judged),
  * a RED arrow from the animal's centre toward the predicted HEAD,
  * the flank tag  LEFT / RIGHT  + confidence  -- the thing re-ID consumes.

WHY NO 3D BOX
-------------
The 3D box's ROTATION is untrustworthy (untrained 6D head; GT rotation has arbitrary PCA/SVD
signs -- see HEADING_PROJECT.md §4c). Drawing it invites you to judge an orientation that is
meaningless, and an earlier version of this script also *used* those broken axes to pick the
front face, which could flip the flank tag on its own.

So the flank now comes from `visibility.py`:  image heading (DINOv3) + CAMERA + gravity.
The camera is essential -- it lifts the 2D angle into a world direction AND decides which side
faces the viewer. Only the BOX is dropped. Verified: identical to the box-derived flank on
614/614 instances.

WHAT TO LOOK FOR
----------------
1. Does the red arrow point at the HEAD?
2. Is the flank tag STABLE across consecutive frames?
3. THE PHYSICS CHECK: the flank may only switch when the animal passes through a head-on /
   tail-on view, i.e. when confidence -> 0. A switch at HIGH confidence is impossible -- an
   animal cannot swap which side faces you while standing broadside. That is a label-free
   error detector; it needs no ground truth.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from tools.heading.frame import sign_up_toward_camera, world_up_from_boxes
from tools.heading.io import load_segment
from tools.heading.visibility import visible_flank

DEFAULT_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def load_head(path: Path, device: str):
    """Load a trained head. The architecture comes from tools.heading.model -- never rebuild
    it by hand here, or training and inference silently drift apart."""
    import torch

    from tools.heading.model import HEAD_VERSION, build_head

    ck = torch.load(path, map_location=device, weights_only=False)
    got = ck.get("head_version", 1)
    if got != HEAD_VERSION:
        raise RuntimeError(
            f"checkpoint {path} was written by head_version={got}, but this code is "
            f"v{HEAD_VERSION}. Retrain (train_head.py --save) rather than force a load."
        )
    net = build_head(ck["in_dim"], ck["hidden"]).to(device).eval()
    net.load_state_dict(ck["state_dict"])
    return net, ck["mu"], ck["sd"]


def _arrow(draw, x, y, ang, length, colour, width):
    ex, ey = x + length * math.cos(ang), y + length * math.sin(ang)
    draw.line([(x, y), (ex, ey)], fill=colour, width=width)
    for s in (+1, -1):
        a = ang + s * math.radians(150)
        draw.line([(ex, ey), (ex + 0.3 * length * math.cos(a),
                              ey + 0.3 * length * math.sin(a))], fill=colour, width=width)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segment", type=Path, required=True)
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--rotations", default="raw", choices=["raw", "canonical"])
    args = ap.parse_args()

    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoImageProcessor, AutoModel

    from tools.heading.extract_crops import square_crop

    seg = load_segment(args.segment, rotations=args.rotations)
    print(f"{seg.name}: {len(seg.tracks)} tracks, {len(seg.cameras)} frames")

    net, mu, sd = load_head(args.head, args.device)
    proc = AutoImageProcessor.from_pretrained(args.model)
    dino = AutoModel.from_pretrained(args.model).to(args.device).eval()

    # Only the box AXES feed the gravity consensus (their signs -- the broken part -- are
    # never trusted). Positions come from the box centres. No box rotation reaches the output.
    up_un = world_up_from_boxes(np.concatenate([t.rotations for t in seg.tracks.values()]))
    args.out.mkdir(parents=True, exist_ok=True)

    history: dict[str, list[tuple[int, str, float]]] = {}

    for fidx in sorted(seg.cameras)[:: args.every]:
        cam = seg.cameras[fidx]
        fp = seg.frame_path(fidx)
        if not fp.is_file():
            continue
        img = np.asarray(Image.open(fp).convert("RGB"))
        canvas = Image.fromarray(img.copy())
        draw = ImageDraw.Draw(canvas)

        for tid, tr in seg.tracks.items():
            try:
                i = tr.index_of_frame(fidx)
            except KeyError:
                continue
            crop = square_crop(img, list(tr.bbox_2d[i]), 0.15)
            if crop is None:
                continue

            # DINOv3 -> head -> heading angle in IMAGE space
            pil = Image.fromarray(crop).resize((224, 224), Image.BICUBIC)
            with torch.no_grad():
                h = dino(**proc(images=[pil], return_tensors="pt").to(args.device)).last_hidden_state
                f = torch.cat([h[:, 0], h[:, 1:].mean(1)], -1).float().cpu().numpy()
                v = torch.nn.functional.normalize(
                    net(torch.tensor((f - mu) / sd, device=args.device)), dim=-1)[0].cpu().numpy()
            ang = float(np.arctan2(v[1], v[0]))

            # image angle + CAMERA + gravity -> flank   (no box rotation involved)
            cam_c = -cam.extrinsic[:, :3].T @ cam.extrinsic[:, 3]
            up_w = sign_up_toward_camera(up_un, tr.centers[i], cam_c)
            up_cam = cam.rotate_world_to_cam(up_w)[0]
            p_cam = cam.world_to_cam(tr.centers[i])[0]        # POSITION only
            fl = visible_flank(ang, up_cam, p_cam)
            history.setdefault(tid, []).append((fidx, fl.side, fl.confidence))

            # ---- draw: 2D box + heading arrow + tag (NO 3D box) ----
            x1, y1, x2, y2 = tr.bbox_2d[i]
            draw.rectangle([x1, y1, x2, y2], outline=(80, 200, 255), width=2)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            _arrow(draw, cx, cy, ang, 0.45 * max(x2 - x1, y2 - y1), (255, 40, 40), 4)
            colour = (0, 255, 120) if fl.confidence > 0.35 else (255, 200, 0)
            draw.text((x1, max(0, y1 - 12)),
                      f"t{tid} {fl.side} {fl.confidence:.2f}", fill=colour)

        canvas.save(args.out / f"frame_{fidx:06d}.jpg", quality=90)

    # ---- the label-free physics check ----
    print(f"\nwrote frames -> {args.out}")
    print("\n=== PHYSICS CHECK: a flank may only switch when edge-on (low confidence) ===")
    bad_total = 0
    for tid, seq in sorted(history.items()):
        sw = [k for k in range(1, len(seq)) if seq[k][1] != seq[k - 1][1]]
        bad = [k for k in sw if seq[k][2] > 0.35 and seq[k - 1][2] > 0.35]
        bad_total += len(bad)
        note = ""
        if bad:
            note = "  <-- IMPOSSIBLE: flipped while broadside, at frames " + \
                   ",".join(str(seq[k][0]) for k in bad[:5])
        print(f"  track {tid:>3s}: {len(seq):3d} frames, {len(sw):2d} switch(es), "
              f"{len(bad):2d} impossible{note}")
    print(f"\n  impossible switches: {bad_total}  (should be 0)")
    print("  A switch at LOW confidence is fine -- the animal turned through head-on/tail-on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
