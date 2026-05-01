"""
Wildbox cross-architecture composite panel.

For each selected image, builds a 2-row × N-col composite where each column
is one architecture (ZS GT-2D, OVMono3D-LIFT FT, DetAny3D FT, …) and:
  - Top row    = input image + 2D-bbox overlay (per-class colors)
  - Bottom row = input image + 3D-cuboid wireframe overlay + BEV mini next to it
                 (per-class colors, same palette as top row)

Why this exists:
  - OVMono3D's `visualize_class_agnostic.py` uses random per-detection colors,
    so when comparing the same image across architectures, the same species
    appears in different colors. Useless for cross-arch reading.
  - It also bakes the GT panel into every image; for an N-col composite that
    repeats the GT N times. We avoid that.

This tool reads the raw predictions (`instances_predictions.pth`) for each
architecture, the GT JSON (for K, image dimensions, source paths, GT 3D
boxes), and renders everything itself with a fixed 6-species palette.

Usage:
  python tools/wildbox_compose_panel.py \
      --gt   datasets/Omni3D/WildBox_val.json \
      --out  output/wildbox_panels \
      --col "ZS GT-2D=output/wildbox_detany3d_gt2d_int1_v3/inference/iter_final/WildBox_val/instances_predictions.pth" \
      --col "OVMono3D-LIFT init5sp=output/wl6_init5sp_multiseed/seed0/eval/inference/iter_final/WildBox_val/instances_predictions.pth" \
      --col "DetAny3D FT 2ep=output/wildbox_detany3d_ft_ep2_seed0_int1_v3/inference/iter_final/WildBox_val/instances_predictions.pth" \
      --every 200 --limit 80 \
      --score-min 0.0 \
      [--image-root <prefix-replacement>]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


# Fixed class palette (RGB). Same colors across all arches.
CLASSES = ["giraffe", "grevys_zebra", "elephant", "plains_zebra", "rhino", "gazelle"]
PALETTE = {
    "giraffe":      (235,  90,  90),    # red
    "grevys_zebra": (245, 165,  60),    # orange
    "elephant":     ( 75, 130, 220),    # blue
    "plains_zebra": (240, 210,  60),    # yellow
    "rhino":        (140, 140, 150),    # gray
    "gazelle":      ( 90, 190, 110),    # green
}
DATASET_ID_TO_CONTIG = {1000+i: i for i in range(6)}


# ---------- 3D cuboid corners and projection ----------

def cuboid_corners_cam(center_cam: np.ndarray,
                       dims_cam: Tuple[float, float, float],
                       R_cam: np.ndarray) -> np.ndarray:
    """Return 8 corners of a 3D cuboid in camera coords.

    dims_cam = (W, H, L) per Omni3D convention. The local cuboid is centered
    at origin with extents (±W/2, ±H/2, ±L/2) along the local axes; we then
    rotate by R_cam and translate to center_cam. Returns shape (8, 3).
    """
    W, H, L = float(dims_cam[0]), float(dims_cam[1]), float(dims_cam[2])
    # Local corners — same convention as ovmono3d's get_cuboid_verts_faces
    local = np.array([
        [-L/2, -H/2, -W/2],
        [+L/2, -H/2, -W/2],
        [+L/2, -H/2, +W/2],
        [-L/2, -H/2, +W/2],
        [-L/2, +H/2, -W/2],
        [+L/2, +H/2, -W/2],
        [+L/2, +H/2, +W/2],
        [-L/2, +H/2, +W/2],
    ], dtype=np.float64)
    R = np.asarray(R_cam, dtype=np.float64)
    if R.shape == (9,):
        R = R.reshape(3, 3)
    if R.shape != (3, 3):
        R = np.eye(3)
    rotated = local @ R.T
    return rotated + np.asarray(center_cam, dtype=np.float64)


def project(K: np.ndarray, points_cam: np.ndarray) -> np.ndarray:
    """Project Nx3 cam points to Nx2 pixels."""
    K = np.asarray(K, dtype=np.float64)
    if K.shape == (9,):
        K = K.reshape(3, 3)
    pts_h = (K @ points_cam.T).T
    z = np.maximum(pts_h[:, 2:3], 1e-6)
    return pts_h[:, :2] / z


# 12 edges of the cuboid (pairs of corner indices)
CUBOID_EDGES = [
    (0,1),(1,2),(2,3),(3,0),    # bottom
    (4,5),(5,6),(6,7),(7,4),    # top
    (0,4),(1,5),(2,6),(3,7),    # vertical
]


def draw_2d_box(draw: ImageDraw.ImageDraw, xywh: List[float], color, width=3):
    x, y, w, h = xywh
    draw.rectangle([x, y, x+w, y+h], outline=color, width=width)


def draw_3d_wireframe(draw: ImageDraw.ImageDraw, corners_2d: np.ndarray,
                      color, width=2):
    for a, b in CUBOID_EDGES:
        ax, ay = corners_2d[a]
        bx, by = corners_2d[b]
        draw.line([(ax, ay), (bx, by)], fill=color, width=width)


# ---------- BEV ----------

def render_bev(boxes: List[dict], canvas_size=(280, 280),
               x_range=(-15, 15), z_range=(0, 30)) -> Image.Image:
    """Top-down view: x is lateral, z is depth into scene.

    Each box has 'x', 'z', 'W', 'L', 'yaw', 'color'. Draws each as a
    rotated rectangle viewed from above.
    """
    W, H = canvas_size
    img = Image.new("RGB", (W, H), (235, 235, 240))
    draw = ImageDraw.Draw(img)

    def world_to_canvas(x, z):
        u = (x - x_range[0]) / (x_range[1] - x_range[0]) * W
        v = H - (z - z_range[0]) / (z_range[1] - z_range[0]) * H  # flip v so far-z is top
        return (u, v)

    # Background grid
    for gx in np.linspace(x_range[0], x_range[1], 7):
        u, _ = world_to_canvas(gx, z_range[0])
        draw.line([(u, 0), (u, H)], fill=(210, 210, 215), width=1)
    for gz in np.linspace(z_range[0], z_range[1], 7):
        _, v = world_to_canvas(0, gz)
        draw.line([(0, v), (W, v)], fill=(210, 210, 215), width=1)

    # Camera at origin marker
    cu, cv = world_to_canvas(0, 0)
    draw.ellipse([cu-3, cv-3, cu+3, cv+3], fill=(50, 50, 60))

    for b in boxes:
        cx, cz = b["x"], b["z"]
        bw, bl = b["W"], b["L"]
        yaw = b.get("yaw", 0.0)
        c, s = math.cos(yaw), math.sin(yaw)
        # 4 corners of the rectangle in world (top-down)
        local = np.array([
            [-bl/2, -bw/2], [+bl/2, -bw/2],
            [+bl/2, +bw/2], [-bl/2, +bw/2],
        ])
        rot = np.array([[c, -s], [s, c]])
        world = local @ rot.T + np.array([cx, cz])
        canv = [world_to_canvas(x, z) for x, z in world]
        draw.polygon(canv, outline=b["color"], width=2)
    return img


# ---------- IO ----------

def load_predictions(path: Path) -> Dict[int, List[dict]]:
    """Load instances_predictions.pth → image_id → list of instances."""
    preds = torch.load(path, weights_only=False, map_location="cpu")
    out = {}
    for entry in preds:
        out[int(entry["image_id"])] = entry.get("instances", [])
    return out


def load_gt_index(gt_json_path: Path) -> Tuple[Dict[int, dict], Dict[int, list]]:
    gt = json.loads(gt_json_path.read_text())
    image_meta = {}
    for im in gt["images"]:
        image_meta[int(im["id"])] = {
            "width": int(im["width"]),
            "height": int(im["height"]),
            "K": im.get("K"),
            "file_path": im.get("file_path"),
        }
    gt_anns = {}
    for ann in gt["annotations"]:
        img_id = int(ann["image_id"])
        gt_anns.setdefault(img_id, []).append(ann)
    return image_meta, gt_anns


def class_color(category_id: int) -> Tuple[int, int, int]:
    contig = DATASET_ID_TO_CONTIG.get(category_id, category_id)
    if 0 <= contig < len(CLASSES):
        return PALETTE[CLASSES[contig]]
    return (180, 180, 180)


# ---------- Per-image rendering ----------

def render_arch_cells(image_path: Path, K: np.ndarray, image_size: Tuple[int, int],
                       instances: List[dict], score_min: float,
                       max_dets: int = 12) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """Return (img_with_2d, img_with_3d, bev_mini) for one architecture's preds."""
    # Filter + sort by score
    selected = [it for it in instances if float(it.get("score", 1.0)) >= score_min]
    selected = sorted(selected, key=lambda it: -float(it.get("score", 1.0)))[:max_dets]

    # Load source image; if missing, gray placeholder
    if image_path and image_path.exists():
        try:
            base = Image.open(image_path).convert("RGB")
        except Exception:
            base = Image.new("RGB", image_size, (220, 220, 220))
    else:
        base = Image.new("RGB", image_size, (220, 220, 220))

    img_2d = base.copy()
    img_3d = base.copy()
    d2d = ImageDraw.Draw(img_2d)
    d3d = ImageDraw.Draw(img_3d)

    bev_boxes = []
    for it in selected:
        cid = int(it.get("category_id", 0))
        color = class_color(cid)

        # 2D box (xywh)
        bbox = it.get("bbox")
        if bbox and len(bbox) == 4:
            draw_2d_box(d2d, bbox, color, width=3)

        # 3D wireframe
        bbox3d_cam = it.get("bbox3D_cam") or it.get("bbox3D")
        center_cam = it.get("center_cam")
        dims = it.get("dimensions")
        pose = it.get("pose")

        if bbox3d_cam and len(bbox3d_cam) == 8:
            corners3d = np.asarray(bbox3d_cam, dtype=np.float64)
        elif center_cam and dims and pose:
            corners3d = cuboid_corners_cam(np.asarray(center_cam),
                                            dims, np.asarray(pose))
        else:
            continue

        if K is not None:
            corners2d = project(K, corners3d)
            draw_3d_wireframe(d3d, corners2d, color, width=2)

        # BEV record
        if center_cam and dims:
            yaw = 0.0
            if isinstance(pose, list) and len(pose) == 9:
                R = np.asarray(pose).reshape(3, 3)
                yaw = math.atan2(R[0, 2], R[2, 2])
            elif isinstance(pose, list) and len(pose) == 3 and len(pose[0]) == 3:
                R = np.asarray(pose)
                yaw = math.atan2(R[0, 2], R[2, 2])
            W_, H_, L_ = float(dims[0]), float(dims[1]), float(dims[2])
            bev_boxes.append({
                "x": float(center_cam[0]),
                "z": float(center_cam[2]),
                "W": W_, "L": L_, "yaw": yaw, "color": color,
            })

    bev = render_bev(bev_boxes, canvas_size=(280, 280),
                     x_range=(-15, 15), z_range=(0, 30))
    return img_2d, img_3d, bev


# ---------- Composite layout ----------

def compose_panel(per_arch: List[Tuple[str, Image.Image, Image.Image, Image.Image]],
                  cell_h: int = 360, label_h: int = 30, gap: int = 8) -> Image.Image:
    """Layout:
      Row 1 (2D): label_h | [arch1 2D] [arch2 2D] [arch3 2D]
      Row 2 (3D+BEV): label_h | [arch1 3D | bev] [arch2 3D | bev] [arch3 3D | bev]
    Each cell resized to cell_h height, BEV resized to cell_h on the side.
    """
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 13)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font

    def fit_h(im: Image.Image, h: int) -> Image.Image:
        w, h0 = im.size
        new_w = max(1, int(w * h / h0))
        return im.resize((new_w, h), Image.LANCZOS)

    # Resize all cells to cell_h
    row1_imgs = [fit_h(p[1], cell_h) for p in per_arch]
    row2_imgs = [fit_h(p[2], cell_h) for p in per_arch]
    bev_imgs  = [fit_h(p[3], cell_h) for p in per_arch]

    # Each row-2 column = 3D image + BEV side-by-side
    row2_widths = [r.size[0] + gap + b.size[0] for r, b in zip(row2_imgs, bev_imgs)]
    col_widths  = [max(r1.size[0], w2) for r1, w2 in zip(row1_imgs, row2_widths)]
    total_w = sum(col_widths) + gap * (len(col_widths) - 1)
    total_h = label_h + cell_h + gap + label_h + cell_h

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    d = ImageDraw.Draw(canvas)

    x = 0
    for i, (label, _img2d, _img3d, _bev) in enumerate(per_arch):
        col_w = col_widths[i]

        # Column header (architecture name) — top of row 1
        d.text((x + 6, 6), label, fill=(20, 20, 20), font=font)

        # Row 1: 2D image, centered in column
        r1 = row1_imgs[i]
        x_r1 = x + (col_w - r1.size[0]) // 2
        canvas.paste(r1, (x_r1, label_h))

        # Row 2 mini-header
        y_row2 = label_h + cell_h + gap
        d.text((x + 6, y_row2 + 4), label + "  3D + BEV",
               fill=(20, 20, 20), font=font_small)

        # Row 2: 3D image + BEV side-by-side
        y_row2_img = y_row2 + label_h
        r2 = row2_imgs[i]
        bev = bev_imgs[i]
        x_r2 = x + (col_w - (r2.size[0] + gap + bev.size[0])) // 2
        canvas.paste(r2, (x_r2, y_row2_img))
        canvas.paste(bev, (x_r2 + r2.size[0] + gap, y_row2_img))

        x += col_w + gap

    # Class palette legend at the bottom
    legend_h = 28
    legend = Image.new("RGB", (total_w, legend_h), (245, 245, 248))
    ld = ImageDraw.Draw(legend)
    lx = 8
    for cls in CLASSES:
        col = PALETTE[cls]
        ld.rectangle([lx, 6, lx+18, 22], fill=col, outline=(60, 60, 70))
        ld.text((lx + 24, 8), cls, fill=(20, 20, 20), font=font_small)
        try:
            tw = ld.textlength(cls, font=font_small)
        except AttributeError:
            tw = font_small.getsize(cls)[0] if hasattr(font_small, "getsize") else 80
        lx += int(tw) + 50

    out = Image.new("RGB", (total_w, total_h + legend_h), (255, 255, 255))
    out.paste(canvas, (0, 0))
    out.paste(legend, (0, total_h))
    return out


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True,
                    help="Path to WildBox_val.json")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory for composite panels")
    ap.add_argument("--col", action="append", required=True,
                    help="Per-column spec 'label=path/to/instances_predictions.pth'. "
                         "Pass once per column (recommended: 3 cols).")
    ap.add_argument("--every", type=int, default=200,
                    help="Stride through image_ids (default: 200)")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--score-min", type=float, default=0.0)
    ap.add_argument("--max-dets", type=int, default=12,
                    help="Max detections drawn per arch per image")
    ap.add_argument("--image-root", type=str, default="",
                    help="Optional path-prefix replacement for img file_path")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Parse cols
    cols = []
    for spec in args.col:
        if "=" not in spec:
            print(f"  WARN: bad --col spec, skipping: {spec}")
            continue
        label, path = spec.split("=", 1)
        p = Path(path.strip())
        if not p.exists():
            print(f"  WARN: predictions not found: {p}")
            continue
        cols.append((label.strip(), p))

    if not cols:
        print("FATAL: no valid --col specs")
        return 2

    # Load GT
    print(f"Loading GT from {args.gt}")
    image_meta, gt_anns = load_gt_index(args.gt)

    # Load all predictions
    arch_preds = []
    for label, path in cols:
        print(f"Loading {label}: {path}")
        arch_preds.append((label, load_predictions(path)))

    # Choose image_ids: those present in ALL columns
    common_ids = set(image_meta.keys())
    for _, preds in arch_preds:
        common_ids = common_ids & set(preds.keys())
    common_ids = sorted(common_ids)
    print(f"common image_ids across all archs: {len(common_ids)}")

    selected = common_ids[::args.every][:args.limit]
    print(f"will render {len(selected)} composites (every={args.every}, limit={args.limit})")

    n_done = 0
    for img_id in selected:
        meta = image_meta.get(img_id)
        if meta is None:
            continue
        K = meta.get("K") or [[1000, 0, meta["width"]/2],
                              [0, 1000, meta["height"]/2],
                              [0, 0, 1]]
        K = np.asarray(K, dtype=np.float64)
        if K.shape == (3, 3, 1) or len(np.asarray(K).shape) == 3:
            K = np.asarray(K).reshape(3, 3)

        # Resolve image path
        fp = meta.get("file_path", "") or ""
        if args.image_root and fp:
            fp = args.image_root + fp
        image_path = Path(fp) if fp else None

        per_arch = []
        for arch_label, preds in arch_preds:
            insts = preds.get(img_id, [])
            img2d, img3d, bev = render_arch_cells(
                image_path, K, (meta["width"], meta["height"]),
                insts, args.score_min, args.max_dets,
            )
            per_arch.append((arch_label, img2d, img3d, bev))

        composite = compose_panel(per_arch)
        out_path = args.out / f"img_{img_id:06d}.jpg"
        composite.save(out_path, quality=92, optimize=True)
        n_done += 1
        if n_done <= 3 or n_done % 20 == 0:
            print(f"  wrote {out_path.name} ({composite.size[0]}×{composite.size[1]})")

    print(f"\nDone. {n_done} composites in {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
