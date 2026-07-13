"""Run the trained heading head on an UNLABELLED segment and draw what it thinks.

    python -m tools.heading.predict \
        --segment /storage3/.../WildBox/DJI_2024.../seg5 \
        --head    data/heading/head.pt \
        --out     data/heading/vis/seg5 \
        --every 10 --device cuda

For every animal in every Nth frame this renders:
  * the 3D box wireframe,
  * a RED arrow from the body centre toward the predicted HEAD,
  * the derived flank tag (LEFT / RIGHT) -- the thing re-ID actually consumes,
  * the model's confidence.

WHY THIS IS THE REAL TEST
-------------------------
Held-out metrics are computed on zebra tracks from the same few videos. Rendering on
segments the model has never seen -- different species, different flight, no labels --
is the only way to see the failure modes the aggregate hides. Look specifically for the
arrow pointing at the TAIL: that is the 180-degree confusion, and it is the failure that
flips the flank tag and would poison a re-ID gallery.

Uses `rotations="raw"`, i.e. the same per-frame PCA rotations WildBox itself was built
from -- so what you see is exactly what a WildBox-time consumer would get.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tools.heading.conventions import corners_of, face_centers_world
from tools.heading.frame import (
    face_map,
    frame_from_front_face,
    horizontal_faces,
    sign_up_toward_camera,
    world_up_from_boxes,
)
from tools.heading.io import load_segment

DEFAULT_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def load_head(path: Path, device: str):
    import torch
    import torch.nn as nn

    ck = torch.load(path, map_location=device, weights_only=False)
    net = nn.Sequential(
        nn.Linear(ck["in_dim"], ck["hidden"]), nn.GELU(), nn.Dropout(0.2),
        nn.Linear(ck["hidden"], 2),
    ).to(device).eval()
    net.load_state_dict(ck["state_dict"])
    return net, ck["mu"], ck["sd"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segment", type=Path, required=True)
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--every", type=int, default=10, help="render every Nth frame")
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

    up_un = world_up_from_boxes(np.concatenate([t.rotations for t in seg.tracks.values()]))
    args.out.mkdir(parents=True, exist_ok=True)

    frames = sorted(seg.cameras)[:: args.every]
    for fidx in frames:
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

            # --- DINOv3 -> head -> predicted heading angle (image space) -------------
            pil = Image.fromarray(crop).resize((224, 224), Image.BICUBIC)
            with torch.no_grad():
                out = dino(**proc(images=[pil], return_tensors="pt").to(args.device)).last_hidden_state
                feat = torch.cat([out[:, 0], out[:, 1:].mean(1)], dim=-1).float().cpu().numpy()
                z = torch.tensor((feat - mu) / sd, device=args.device)
                v = torch.nn.functional.normalize(net(z), dim=-1)[0].cpu().numpy()
            ang = float(np.arctan2(v[1], v[0]))

            # --- angle -> front face: pick the candidate whose projected direction
            #     from the body centre best matches the predicted heading -------------
            R, ctr, dims = tr.rotations[i], tr.centers[i], tr.dimensions[i]
            cam_c = -cam.extrinsic[:, :3].T @ cam.extrinsic[:, 3]
            up = sign_up_toward_camera(up_un, ctr, cam_c)
            body_uv = cam.project(cam.world_to_cam(ctr))[0]
            fc = face_centers_world(ctr, dims, R)

            best, best_cos = None, -2.0
            for f in horizontal_faces(R, up):
                p_cam = cam.world_to_cam(fc[f])[0]
                if p_cam[2] <= 1e-6:
                    continue
                d = cam.project(p_cam)[0] - body_uv
                n = float(np.linalg.norm(d))
                if n < 1e-6:
                    continue
                c = float(np.cos(ang) * d[0] / n + np.sin(ang) * d[1] / n)
                if c > best_cos:
                    best, best_cos = f, c
            if best is None:
                continue

            fr = frame_from_front_face(R, up, best)
            fm = face_map(R, fr)

            # --- box wireframe ------------------------------------------------------
            cc = cam.world_to_cam(corners_of(ctr, dims, R))
            if np.any(cc[:, 2] <= 1e-6):
                continue                                    # box behind the camera
            uv = cam.project(cc)
            for a, b in EDGES:
                draw.line([tuple(uv[a]), tuple(uv[b])], fill=(80, 200, 255), width=2)

            # --- heading arrow: body centre -> predicted HEAD ------------------------
            head_uv = cam.project(cam.world_to_cam(fc[best]))[0]
            draw.line([tuple(body_uv), tuple(head_uv)], fill=(255, 40, 40), width=4)
            r = 5
            draw.ellipse([head_uv[0] - r, head_uv[1] - r, head_uv[0] + r, head_uv[1] + r],
                         fill=(255, 40, 40))

            # --- which flank does the camera SEE? that IS the re-ID tag --------------
            # In camera coords the camera sits at the origin, so a face is visible iff
            # its outward normal points back toward the origin: n . c < 0, where c is the
            # face centre. Test the LEFT face specifically.
            left_c = cam.world_to_cam(fc[fm["left"]])[0]
            left_n = cam.rotate_world_to_cam(fr.left)[0]
            flank = "LEFT" if float(np.dot(left_n, left_c)) < 0 else "RIGHT"

            x1, y1 = tr.bbox_2d[i][0], tr.bbox_2d[i][1]
            draw.text((x1, max(0, y1 - 12)),
                      f"t{tid} {flank} conf={best_cos:.2f}", fill=(255, 255, 0))

        canvas.save(args.out / f"frame_{fidx:06d}.jpg", quality=90)

    print(f"wrote {len(frames)} annotated frames -> {args.out}")
    print("Look for: the RED arrow pointing at the TAIL -- that is the 180-deg failure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
