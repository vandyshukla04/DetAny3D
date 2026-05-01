"""
Wildbox cross-architecture composite panel — paper-figure version.

Layout (per image):
  Row 1 (zero-shot):  [arch1: img+3D + BEV] [arch2: img+3D + BEV] ...
  Row 2 (fine-tuned): [arch1: img+3D + BEV] [arch2: img+3D + BEV] ...

The 2D-only row was removed (it was awkwardly placed). Each cell is
img-with-3D-wireframe paired with a BEV mini next to it.

Fixed 6-species palette so the same color = same species across all archs.
Box edges have a black stroke for visibility against varied backgrounds.

CLI uses '<row> / <label>=path' to assign each prediction file to a row:
  --col "ZS / OVMono3D-LIFT=output/.../instances_predictions.pth"
  --col "ZS / DetAny3D=...pth"
  --col "FT / OVMono3D-LIFT init5sp=...pth"
  --col "FT / DetAny3D FT 2ep=...pth"

Row-label canonicalisation: strings starting with 'ZS' or 'Zero-shot' go to
the top row; 'FT' or 'Fine-tuned' → bottom. Other strings are taken as-is
and grouped lexicographically.

Usage example:
  python tools/wildbox_compose_panel.py \
      --gt   datasets/Omni3D/WildBox_val.json \
      --out  output/wildbox_panels \
      --col  "ZS / OVMono3D-LIFT=output/wl6_zeroshot_oracle2d/inference/iter_final/WildBox_val/instances_predictions.pth" \
      --col  "ZS / DetAny3D=output/wildbox_detany3d_zs_int1_v3/inference/iter_final/WildBox_val/instances_predictions.pth" \
      --col  "FT / OVMono3D-LIFT init5sp=output/wl6_init5sp_multiseed/seed0/eval/inference/iter_final/WildBox_val/instances_predictions.pth" \
      --col  "FT / DetAny3D FT 2ep=output/wildbox_detany3d_ft_ep2_seed0_int1_v3/inference/iter_final/WildBox_val/instances_predictions.pth" \
      --every 200 --limit 80 --score-min 0.05 --max-dets 12
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
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


# Row ordering: ZS row above FT row.
ROW_ORDER = ["ZS", "FT", "OTHER"]


def canonical_row(label: str) -> str:
    s = label.strip().lower()
    if s.startswith("zs") or s.startswith("zero"):
        return "ZS"
    if s.startswith("ft") or s.startswith("fine"):
        return "FT"
    return "OTHER"


# ---------- 3D cuboid corners and projection ----------

def cuboid_corners_cam(center_cam: np.ndarray,
                       dims_cam: Tuple[float, float, float],
                       R_cam: np.ndarray) -> np.ndarray:
    """8 corners of a 3D cuboid in camera coords.

    Omni3D convention: dims=(W, H, L). Local cuboid centered at origin,
    extents (±W/2, ±H/2, ±L/2) along local axes; rotated by R_cam, then
    translated to center_cam. Returns shape (8, 3).
    """
    W, H, L = float(dims_cam[0]), float(dims_cam[1]), float(dims_cam[2])
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
    if R.shape == (9,): R = R.reshape(3, 3)
    if R.shape != (3, 3): R = np.eye(3)
    rotated = local @ R.T
    return rotated + np.asarray(center_cam, dtype=np.float64)


def project(K: np.ndarray, points_cam: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64)
    if K.shape == (9,): K = K.reshape(3, 3)
    pts_h = (K @ points_cam.T).T
    z = np.maximum(pts_h[:, 2:3], 1e-6)
    return pts_h[:, :2] / z


CUBOID_EDGES = [
    (0,1),(1,2),(2,3),(3,0),    # bottom
    (4,5),(5,6),(6,7),(7,4),    # top
    (0,4),(1,5),(2,6),(3,7),    # vertical
]
# Front-face edges (drawn thicker / with cross to indicate front)
FRONT_FACE_EDGES = [(0,1),(1,5),(5,4),(4,0)]    # +X face for L-axis convention


def draw_3d_wireframe(draw: ImageDraw.ImageDraw, corners_2d: np.ndarray,
                      color: Tuple[int,int,int], line_width: int = 3,
                      stroke_width: int = 1):
    """Draw 3D cuboid wireframe with a black stroke under each line for contrast."""
    # Black stroke layer (under)
    if stroke_width > 0:
        for a, b in CUBOID_EDGES:
            ax, ay = corners_2d[a]; bx, by = corners_2d[b]
            draw.line([(ax, ay), (bx, by)], fill=(0, 0, 0),
                      width=line_width + 2 * stroke_width)
    # Color layer (over)
    for a, b in CUBOID_EDGES:
        ax, ay = corners_2d[a]; bx, by = corners_2d[b]
        draw.line([(ax, ay), (bx, by)], fill=color, width=line_width)


# ---------- BEV ----------

def render_bev(boxes: List[dict], canvas_size=(280, 280),
               x_range=(-15, 15), z_range=(0, 30)) -> Image.Image:
    """Top-down (x, z) view. Camera at origin (bottom-center), +z forward (up).

    Box rectangle convention (matches Omni3D dims=(W, H, L)):
      - Width W extends along the local x-axis (lateral)
      - Length L extends along the local z-axis (depth)
    yaw rotates the rectangle around the vertical (Y) axis.
    """
    W_canvas, H_canvas = canvas_size
    img = Image.new("RGB", (W_canvas, H_canvas), (245, 245, 248))
    draw = ImageDraw.Draw(img)

    def world_to_canvas(x, z):
        # x grows left→right; z grows bottom→top (camera at bottom looking up)
        u = (x - x_range[0]) / (x_range[1] - x_range[0]) * W_canvas
        v = H_canvas - (z - z_range[0]) / (z_range[1] - z_range[0]) * H_canvas
        return (u, v)

    # Background grid every 5 m in x and z
    for gx in range(int(math.ceil(x_range[0] / 5)) * 5,
                    int(math.floor(x_range[1] / 5)) * 5 + 1, 5):
        u, _ = world_to_canvas(gx, z_range[0])
        col = (200, 200, 210) if gx == 0 else (220, 220, 228)
        draw.line([(u, 0), (u, H_canvas)], fill=col, width=1)
    for gz in range(int(math.ceil(z_range[0] / 5)) * 5,
                    int(math.floor(z_range[1] / 5)) * 5 + 1, 5):
        _, v = world_to_canvas(0, gz)
        col = (200, 200, 210)
        draw.line([(0, v), (W_canvas, v)], fill=col, width=1)

    # Camera at world origin (0, 0)
    cu, cv = world_to_canvas(0, 0)
    # Camera triangle (FOV indicator pointing up = +z forward)
    fov_half = math.radians(35)
    far_z = z_range[1]
    far_left  = world_to_canvas(-far_z * math.tan(fov_half), far_z)
    far_right = world_to_canvas(+far_z * math.tan(fov_half), far_z)
    draw.polygon([(cu, cv), far_left, far_right], fill=(252, 252, 255), outline=(180, 180, 195))
    draw.ellipse([cu-4, cv-4, cu+4, cv+4], fill=(50, 50, 60))

    # Boxes — rotated rectangles
    for b in boxes:
        cx, cz = b["x"], b["z"]
        bw, bl = b["W"], b["L"]
        yaw = b.get("yaw", 0.0)
        c, s = math.cos(yaw), math.sin(yaw)
        # Local rectangle: x extent = W (lateral), z extent = L (depth)
        local = np.array([
            [-bw/2, -bl/2],   # back-left
            [+bw/2, -bl/2],   # back-right
            [+bw/2, +bl/2],   # front-right
            [-bw/2, +bl/2],   # front-left
        ])
        rot = np.array([[c, -s], [s, c]])
        world = local @ rot.T + np.array([cx, cz])
        canv = [world_to_canvas(x, z) for x, z in world]
        # Black stroke + color fill outline for visibility
        draw.polygon(canv, outline=(0, 0, 0), width=4)
        draw.polygon(canv, outline=b["color"], width=2)
        # Front-edge highlight (thicker on the front face — between corners 2,3 = front-left/right)
        front_a = canv[2]; front_b = canv[3]
        draw.line([front_a, front_b], fill=b["color"], width=3)

    # Labels along axes
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 10)
    except (OSError, IOError):
        font = ImageFont.load_default()
    draw.text((4, H_canvas - 12), f"x∈[{x_range[0]},{x_range[1]}] z∈[{z_range[0]},{z_range[1]}] m",
              fill=(80, 80, 90), font=font)
    draw.text((W_canvas - 30, 4), "+z", fill=(60, 60, 80), font=font)
    return img


# ---------- IO ----------

def load_predictions(path: Path) -> Dict[int, List[dict]]:
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


# ---------- Per-cell rendering ----------

def render_cell(image_path: Optional[Path], K: np.ndarray, image_size: Tuple[int, int],
                instances: List[dict], score_min: float,
                max_dets: int = 12) -> Tuple[Image.Image, Image.Image]:
    """Return (img_with_3d_wireframe, bev_mini)."""
    selected = [it for it in instances if float(it.get("score", 1.0)) >= score_min]
    selected = sorted(selected, key=lambda it: -float(it.get("score", 1.0)))[:max_dets]

    if image_path and image_path.exists():
        try:
            base = Image.open(image_path).convert("RGB")
        except Exception:
            base = Image.new("RGB", image_size, (220, 220, 220))
    else:
        base = Image.new("RGB", image_size, (220, 220, 220))

    img_3d = base.copy()
    d3d = ImageDraw.Draw(img_3d)

    bev_boxes = []
    for it in selected:
        cid = int(it.get("category_id", 0))
        color = class_color(cid)

        bbox3d_cam = it.get("bbox3D_cam") or it.get("bbox3D")
        center_cam = it.get("center_cam")
        dims = it.get("dimensions")
        pose = it.get("pose")

        # 3D wireframe
        if bbox3d_cam and len(bbox3d_cam) == 8:
            corners3d = np.asarray(bbox3d_cam, dtype=np.float64)
        elif center_cam and dims and pose:
            corners3d = cuboid_corners_cam(np.asarray(center_cam), dims, np.asarray(pose))
        else:
            continue

        if K is not None:
            corners2d = project(K, corners3d)
            draw_3d_wireframe(d3d, corners2d, color, line_width=3, stroke_width=1)

        # BEV record
        if center_cam and dims:
            yaw = 0.0
            R = None
            if isinstance(pose, list):
                arr = np.asarray(pose, dtype=np.float64)
                if arr.shape == (3, 3):
                    R = arr
                elif arr.shape == (9,):
                    R = arr.reshape(3, 3)
                elif arr.ndim == 2 and arr.shape == (3, 3):
                    R = arr
            if R is not None:
                yaw = math.atan2(R[0, 2], R[2, 2])
            W_, _, L_ = float(dims[0]), float(dims[1]), float(dims[2])
            bev_boxes.append({
                "x": float(center_cam[0]),
                "z": float(center_cam[2]),
                "W": W_, "L": L_, "yaw": yaw, "color": color,
            })

    bev = render_bev(bev_boxes)
    return img_3d, bev


# ---------- Composite layout ----------

def compose_panel(rows: Dict[str, List[Tuple[str, Image.Image, Image.Image]]],
                  cell_h: int = 380, label_h: int = 26, gap: int = 6,
                  row_label_w: int = 28) -> Image.Image:
    """Layout:

      [row label "ZS"] | col1: img-3D + BEV | col2: img-3D + BEV | ...
      [row label "FT"] | col1: img-3D + BEV | col2: img-3D + BEV | ...

    Each cell label sits ABOVE the cell. Each cell pair = (image-with-3D, BEV)
    side by side. row_label is a thin vertical band on the left of each row.
    """
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
        font_row = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except (OSError, IOError):
        font = ImageFont.load_default(); font_row = font

    def fit_h(im: Image.Image, h: int) -> Image.Image:
        w, h0 = im.size
        new_w = max(1, int(w * h / h0))
        return im.resize((new_w, h), Image.LANCZOS)

    # Pre-resize everything
    resized_rows: Dict[str, List[Tuple[str, Image.Image, Image.Image]]] = {}
    for row_key, cells in rows.items():
        resized_rows[row_key] = [(label, fit_h(img3d, cell_h), fit_h(bev, cell_h))
                                  for label, img3d, bev in cells]

    # Compute per-column widths: max across rows for matching col index
    n_cols = max(len(c) for c in resized_rows.values()) if resized_rows else 0
    col_widths = []
    for ci in range(n_cols):
        ws = []
        for row_key, cells in resized_rows.items():
            if ci < len(cells):
                _, img3d, bev = cells[ci]
                ws.append(img3d.size[0] + gap + bev.size[0])
        col_widths.append(max(ws) if ws else 0)

    row_total_w = row_label_w + sum(col_widths) + gap * (n_cols - 1)
    n_rows = len(resized_rows)
    row_total_h = label_h + cell_h
    total_h = n_rows * row_total_h + (n_rows - 1) * gap

    canvas = Image.new("RGB", (row_total_w, total_h), (255, 255, 255))
    d = ImageDraw.Draw(canvas)

    y = 0
    sorted_rows = sorted(resized_rows.keys(),
                         key=lambda k: (ROW_ORDER.index(k) if k in ROW_ORDER else 99, k))
    for row_key in sorted_rows:
        cells = resized_rows[row_key]
        # Row label band (vertical strip on left)
        # Draw a translucent band; rotate text 90°
        band_color = (240, 245, 250) if row_key == "ZS" else (240, 250, 240)
        d.rectangle([0, y, row_label_w, y + row_total_h], fill=band_color, outline=(220, 220, 230))
        # Vertical text via temporary image rotation
        tmp = Image.new("RGBA", (row_total_h, row_label_w), (0, 0, 0, 0))
        td = ImageDraw.Draw(tmp)
        td.text((10, 4), row_key, fill=(40, 40, 50), font=font_row)
        tmp = tmp.rotate(90, expand=True)
        canvas.paste(tmp, (0, y), tmp)

        x = row_label_w
        for ci, (label, img3d, bev) in enumerate(cells):
            col_w = col_widths[ci]
            # Cell label (top of each cell)
            d.text((x + 6, y + 4), label, fill=(20, 20, 20), font=font)
            # Cell content: img3d + BEV side-by-side, centered in column
            content_w = img3d.size[0] + gap + bev.size[0]
            x_cell = x + (col_w - content_w) // 2
            canvas.paste(img3d, (x_cell, y + label_h))
            canvas.paste(bev, (x_cell + img3d.size[0] + gap, y + label_h))
            x += col_w + gap

        y += row_total_h + gap

    return canvas


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--col", action="append", required=True,
                    help="'<row> / <label>=path' where row in {ZS, FT}.")
    ap.add_argument("--every", type=int, default=200)
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--score-min", type=float, default=0.0)
    ap.add_argument("--max-dets", type=int, default=12)
    ap.add_argument("--image-root", type=str, default="")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Parse cols → row → list of (label, path)
    row_specs: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)
    for spec in args.col:
        if "=" not in spec:
            print(f"  WARN bad --col: {spec}"); continue
        full_label, path_s = spec.split("=", 1)
        # Split on " / " (or "/") to get row + arch label
        if "/" in full_label:
            row_part, arch_part = full_label.split("/", 1)
            row_part = row_part.strip(); arch_part = arch_part.strip()
        else:
            row_part = full_label.strip(); arch_part = full_label.strip()
        row_key = canonical_row(row_part)
        p = Path(path_s.strip())
        if not p.exists():
            print(f"  WARN missing predictions: {p}"); continue
        row_specs[row_key].append((arch_part, p))

    if not row_specs:
        print("FATAL: no valid columns"); return 2
    print("Rows:")
    for r, items in row_specs.items():
        print(f"  {r}: {[label for label, _ in items]}")

    # Load GT + all predictions
    print(f"Loading GT from {args.gt}")
    image_meta, gt_anns = load_gt_index(args.gt)
    arch_preds: Dict[str, Dict[str, Dict[int, list]]] = {}
    for row_key, items in row_specs.items():
        arch_preds[row_key] = {}
        for label, path in items:
            print(f"  loading {row_key} / {label}: {path}")
            arch_preds[row_key][label] = load_predictions(path)

    # Common image_ids = those present in all archs across all rows
    common_ids = set(image_meta.keys())
    for row_key, by_arch in arch_preds.items():
        for label, preds in by_arch.items():
            common_ids = common_ids & set(preds.keys())
    common_ids = sorted(common_ids)
    print(f"common image_ids: {len(common_ids)}")

    selected = common_ids[::args.every][:args.limit]
    print(f"will render {len(selected)} composites (every={args.every}, limit={args.limit})")

    n_done = 0
    for img_id in selected:
        meta = image_meta.get(img_id)
        if meta is None: continue
        K = meta.get("K") or [[1000, 0, meta["width"]/2],
                              [0, 1000, meta["height"]/2],
                              [0, 0, 1]]
        K = np.asarray(K, dtype=np.float64)
        if len(K.shape) == 3:
            K = K.reshape(3, 3)

        fp = meta.get("file_path", "") or ""
        if args.image_root and fp:
            fp = args.image_root + fp
        image_path = Path(fp) if fp else None

        rows: Dict[str, List[Tuple[str, Image.Image, Image.Image]]] = {}
        for row_key, by_arch in arch_preds.items():
            row_cells = []
            for label, preds in by_arch.items():
                insts = preds.get(img_id, [])
                img3d, bev = render_cell(
                    image_path, K, (meta["width"], meta["height"]),
                    insts, args.score_min, args.max_dets,
                )
                row_cells.append((label, img3d, bev))
            rows[row_key] = row_cells

        composite = compose_panel(rows)
        out_path = args.out / f"img_{img_id:06d}.jpg"
        composite.save(out_path, quality=92, optimize=True)
        n_done += 1
        if n_done <= 3 or n_done % 20 == 0:
            print(f"  wrote {out_path.name} ({composite.size[0]}×{composite.size[1]})")

    print(f"\nDone. {n_done} composites in {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
