"""
Wildbox cross-architecture composite panel — paper-figure version.

PORTS the OVMono3D `visualize_class_agnostic.py` rendering primitives
(`novel_view_panel`, `draw_3d_wireframe`, `cuboid_corners`, `euler2mat`)
verbatim, with the SINGLE modification of swapping per-instance hue
coloring for per-class coloring (so the same species is the same color
across architectures, not a different random hue per detection).

Layout (per image):
  Row 1 (zero-shot):  [arch1 cell] [arch2 cell] ...
  Row 2 (fine-tuned): [arch1 cell] [arch2 cell] ...

  Each cell = [3D wireframe on input image] + [novel-view (60° pitch)
              with ground-plane grid + cuboid wireframes]

The 2D-only panel was dropped per user request. Cell labels at the top
of each cell, vertical row labels (ZS / FT) on the left.

CLI uses '<row> / <label>=path':
  --col "ZS / OVMono3D-LIFT=...pth"
  --col "ZS / DetAny3D=...pth"
  --col "FT / OVMono3D-LIFT init5sp=...pth"
  --col "FT / DetAny3D FT 2ep=...pth"

Run:
  python tools/wildbox_compose_panel.py \\
      --gt   datasets/Omni3D/WildBox_val.json \\
      --out  output/wildbox_panels \\
      --col  "ZS / OVMono3D-LIFT=...pth" \\
      --col  "ZS / DetAny3D=...pth" \\
      --col  "FT / OVMono3D-LIFT init5sp=...pth" \\
      --col  "FT / DetAny3D FT 2ep=...pth" \\
      --every 200 --limit 80 --score-min 0.05 --max-dets 12
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


# Fixed class palette in BGR (cv2 convention). Same colors across all archs.
CLASSES = ["giraffe", "grevys_zebra", "elephant", "plains_zebra", "rhino", "gazelle"]
PALETTE_BGR = {
    "giraffe":      ( 90,  90, 235),    # red
    "grevys_zebra": ( 60, 165, 245),    # orange
    "elephant":     (220, 130,  75),    # blue
    "plains_zebra": ( 60, 210, 240),    # yellow
    "rhino":        (150, 140, 140),    # gray
    "gazelle":      (110, 190,  90),    # green
}
DATASET_ID_TO_CONTIG = {1000+i: i for i in range(6)}


def class_color_bgr(category_id: int) -> Tuple[int, int, int]:
    contig = DATASET_ID_TO_CONTIG.get(category_id, category_id)
    if 0 <= contig < len(CLASSES):
        return PALETTE_BGR[CLASSES[contig]]
    return (180, 180, 180)


# --- Row ordering ---
ROW_ORDER = ["ZS", "FT", "OTHER"]


def canonical_row(label: str) -> str:
    s = label.strip().lower()
    if s.startswith("zs") or s.startswith("zero"):
        return "ZS"
    if s.startswith("ft") or s.startswith("fine"):
        return "FT"
    return "OTHER"


# =====================================================================
# PORT: ovmono3d/tools/visualize_class_agnostic.py — geometry primitives
# =====================================================================

def euler2mat(euler):
    """Same as cubercnn.util.math_util.euler2mat."""
    rx = np.array([[1, 0, 0],
                   [0, math.cos(euler[0]), -math.sin(euler[0])],
                   [0, math.sin(euler[0]),  math.cos(euler[0])]])
    ry = np.array([[math.cos(euler[1]), 0, math.sin(euler[1])],
                   [0, 1, 0],
                   [-math.sin(euler[1]), 0, math.cos(euler[1])]])
    rz = np.array([[math.cos(euler[2]), -math.sin(euler[2]), 0],
                   [math.sin(euler[2]),  math.cos(euler[2]), 0],
                   [0, 0, 1]])
    return rz @ ry @ rx


def project(points3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    xs = points3d[:, 0] / points3d[:, 2]
    ys = points3d[:, 1] / points3d[:, 2]
    u = K[0, 0] * xs + K[0, 2]
    v = K[1, 1] * ys + K[1, 2]
    return np.stack([u, v], axis=1)


def cuboid_corners(center, dims_whl, R):
    """Omni3D corner ordering: L=X-ext, H=Y-ext, W=Z-ext."""
    W, H, L = float(dims_whl[0]), float(dims_whl[1]), float(dims_whl[2])
    local = np.array([
        [-L/2, -H/2, -W/2], [+L/2, -H/2, -W/2],
        [+L/2, +H/2, -W/2], [-L/2, +H/2, -W/2],
        [-L/2, -H/2, +W/2], [+L/2, -H/2, +W/2],
        [+L/2, +H/2, +W/2], [-L/2, +H/2, +W/2],
    ], dtype=np.float64)
    return (R @ local.T).T + np.asarray(center, dtype=np.float64)


# Edge list used by cubercnn's draw_3d_box_from_verts
BB3D_EDGES = [[0, 1], [1, 2], [2, 3], [3, 0],
              [1, 5], [5, 6], [6, 2], [4, 5],
              [4, 7], [6, 7], [0, 4], [3, 7]]


def draw_3d_wireframe(im, K, verts3d, color, thickness,
                     zplane=0.05, eps=1e-4):
    """Same behavior as cubercnn.vis.vis.draw_3d_box_from_verts: clips edges
    that cross the camera plane so lines project sensibly."""
    K = np.asarray(K, dtype=np.float64)
    v = np.asarray(verts3d, dtype=np.float64)
    for (i, j) in BB3D_EDGES:
        v0 = v[i].copy()
        v1 = v[j].copy()
        z0, z1 = v0[-1], v1[-1]
        if z0 >= zplane or z1 >= zplane:
            s = (zplane - z0) / max((z1 - z0), eps)
            new_v = v0 + s * (v1 - v0)
            if z0 < zplane <= z1:
                v0 = new_v
            elif z1 < zplane <= z0:
                v1 = new_v
            p0 = (K @ v0) / max(v0[-1], eps)
            p1 = (K @ v1) / max(v1[-1], eps)
            cv2.line(im,
                     (int(p0[0]), int(p0[1])),
                     (int(p1[0]), int(p1[1])),
                     color, thickness, cv2.LINE_AA)


def _draw_label(canvas, xy, text, color, font_scale=0.6):
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                   font_scale, 2)
    x = max(0, int(xy[0]))
    y = max(int(xy[1]), th + 4)
    cv2.rectangle(canvas, (x, y - th - 4), (x + tw + 4, y + bl),
                  color, -1)
    cv2.putText(canvas, text, (x + 2, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), 2)


# =====================================================================
# Front-view (image with 3D cuboid wireframes overlaid)
# =====================================================================

def draw_3d_front_panel(base, instances, K, thickness):
    """Image with per-class-colored 3D cuboid wireframes overlaid."""
    out = base.copy()
    font = max(0.5, out.shape[0] / 1000.0)
    # Depth-sort: far first, so near ones end up on top
    order = sorted(
        range(len(instances)),
        key=lambda i: -(instances[i]["corners3d"].mean(0)[2]
                        if instances[i]["corners3d"] is not None else -1e9),
    )
    for i in order:
        inst = instances[i]
        c = inst["corners3d"]
        if c is None:
            continue
        color = inst["color_bgr"]
        draw_3d_wireframe(out, K, c, color, thickness)
        if c[4, 2] > 0:
            p = project(c[4:5], K)[0]
            _draw_label(out, (p[0], p[1]), inst.get("label", "?"),
                        color, font_scale=font)
    return out


# =====================================================================
# Novel-view panel — VERBATIM port from visualize_class_agnostic.py
# (only change: per-instance color comes from the inst dict, not _color_for)
# =====================================================================

def novel_view_panel(K, im_shape, instances, out_size,
                     pitch_rad=math.pi / 3,
                     with_grid=True,
                     thickness=4,
                     bg=(245, 245, 245),
                     grid_color=(175, 175, 175)):
    """Render cuboid wireframes from a novel viewpoint rotated by pitch_rad
    about the scene center. Optionally draws a projected ground-plane grid.
    CPU-only, no pytorch3d.
    """
    H, W = out_size, out_size
    canvas = np.full((H, W, 3), bg, dtype=np.uint8)

    vis_insts = [inst for inst in instances if inst["corners3d"] is not None]
    if not vis_insts:
        cv2.putText(canvas, "no 3D boxes to render", (20, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        return canvas

    all_verts = np.concatenate([i["corners3d"] for i in vis_insts], axis=0)
    vmin = all_verts.min(axis=0)
    vmax = all_verts.max(axis=0)
    center = (vmin + vmax) / 2.0
    max_y = vmax[1]

    view_R = euler2mat([pitch_rad, 0, 0])

    K_nv = np.array(K, dtype=np.float64).copy()
    K_nv[0, -1] *= W / im_shape[1]
    K_nv[1, -1] *= H / im_shape[0]

    def rotate_verts(v):
        return (view_R @ (v - center).T).T

    base_rot = rotate_verts(all_verts)

    zoom_factor = 100.0
    zoom_factor_in = zoom_factor
    margin = 0.01
    max_trials = 10000
    best_zoom = zoom_factor

    while max_trials > 0:
        zoom_factor_in *= 0.95
        verts = base_rot.copy()
        verts[:, -1] += center[-1] * zoom_factor_in
        if (verts[:, -1] < 0.25).any():
            break
        proj = (K_nv @ verts.T) / verts[:, -1]
        if (proj[:2, :] < W * margin).any():
            break
        if (proj[:2, :] > W * (1 - margin)).any():
            break
        best_zoom = zoom_factor_in
        max_trials -= 1

    zoom_out_bias = float(center[-1])
    zoom_factor = best_zoom

    def transform(v):
        r = rotate_verts(v)
        r[:, -1] += zoom_out_bias * zoom_factor
        return r

    # Ground-plane grid
    if with_grid:
        min_x3d, _, min_z3d = vmin.tolist()
        max_x3d, _, max_z3d = vmax.tolist()
        span_x = max(max_x3d - min_x3d, 1e-3)
        span_z = max(max_z3d - min_z3d, 1e-3)
        x0 = round(min_x3d - span_x * 50)
        x1 = round(max_x3d + span_x * 50)
        z0 = round(min_z3d - span_z * 50)
        z1 = round(max_z3d + span_z * 50)
        step = max(1.0, int(max(x1 - x0, z1 - z0) / 200))
        grid_xs = np.arange(x0, x1, step)
        grid_zs = np.arange(z0, z1, step)
        xs_mesh, zs_mesh = np.meshgrid(grid_xs, grid_zs)
        ys_mesh = np.ones_like(xs_mesh) * max_y
        pts = np.stack([xs_mesh, ys_mesh, zs_mesh], axis=-1).reshape(-1, 3)
        pts_t = transform(pts)
        pts_t[:, -1] = np.clip(pts_t[:, -1], 0.25, None)
        pts_2d = (K_nv @ pts_t.T).T
        pts_2d[:, :2] /= pts_2d[:, 2:]
        in_x = (pts_2d[:, 0] >= -50) & (pts_2d[:, 0] < W + 50) & (pts_2d[:, 2] > 0)
        in_z = (pts_2d[:, 1] >= -50) & (pts_2d[:, 1] < H + 50) & (pts_2d[:, 2] > 0)
        if in_x.any() and in_z.any():
            x3d_start = round(pts[:, 0][in_x].min() - 10)
            x3d_end = round(pts[:, 0][in_x].max() + 10)
            z3d_start = round(pts[:, 2][in_z].min() - 10)
            z3d_end = round(pts[:, 2][in_z].max() + 10)
            grid_xs = np.arange(x3d_start, x3d_end + 1)
            grid_zs = np.arange(z3d_start, z3d_end + 1)
            max_lines = 60
            if len(grid_xs) > max_lines:
                grid_xs = grid_xs[::max(1, len(grid_xs) // max_lines)]
            if len(grid_zs) > max_lines:
                grid_zs = grid_zs[::max(1, len(grid_zs) // max_lines)]
            xs_mesh, zs_mesh = np.meshgrid(grid_xs, grid_zs)
            ys_mesh = np.ones_like(xs_mesh) * max_y
            pts = np.stack([xs_mesh, ys_mesh, zs_mesh], axis=-1)
            shape0 = pts.shape
            pts_flat = pts.reshape(-1, 3)
            pts_t = transform(pts_flat)
            pts_t[:, -1] = np.clip(pts_t[:, -1], 0.25, None)
            pts_2d = (K_nv @ pts_t.T).T
            pts_2d[:, :2] /= pts_2d[:, 2:]
            pts_2d = pts_2d.reshape(shape0[0], shape0[1], 3)
            for r in range(shape0[0] - 1):
                for c in range(shape0[1] - 1):
                    a = pts_2d[r, c]; b = pts_2d[r, c + 1]
                    if a[2] > 0 and b[2] > 0:
                        cv2.line(canvas, (int(a[0]), int(a[1])),
                                 (int(b[0]), int(b[1])),
                                 grid_color, 1, cv2.LINE_AA)
                    a = pts_2d[r, c]; b = pts_2d[r + 1, c]
                    if a[2] > 0 and b[2] > 0:
                        cv2.line(canvas, (int(a[0]), int(a[1])),
                                 (int(b[0]), int(b[1])),
                                 grid_color, 1, cv2.LINE_AA)

    # Cuboids, depth-sorted (far first)
    vis_order = sorted(
        range(len(vis_insts)),
        key=lambda i: -vis_insts[i]["corners3d"].mean(0)[2]
    )
    font = max(0.5, H / 1000.0)
    for i in vis_order:
        inst = vis_insts[i]
        color = inst["color_bgr"]
        verts_t = transform(inst["corners3d"])
        draw_3d_wireframe(canvas, K_nv, verts_t, color, thickness)
        top_c = verts_t[4]
        if top_c[-1] > 0:
            p = (K_nv @ top_c) / top_c[-1]
            _draw_label(canvas, (p[0], p[1]), inst.get("label", "?"),
                        color, font_scale=font)

    cv2.putText(canvas,
                f"novel view  pitch={math.degrees(pitch_rad):.0f}°",
                (12, H - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (90, 90, 90), 2)
    return canvas


# =====================================================================
# Build per-arch instances dict (with class colors attached)
# =====================================================================

def pred_instances_with_class_colors(preds_for_image: list, top_k: int,
                                       score_min: float) -> list:
    """Convert prediction list → instances format used by panel renderers.
    Each instance gets a color_bgr based on its category_id."""
    insts = [it for it in preds_for_image if float(it.get("score", 0)) >= score_min]
    insts.sort(key=lambda x: -float(x.get("score", 0)))
    insts = insts[:top_k]
    out = []
    for i, inst in enumerate(insts):
        try:
            x, y, w, h = inst["bbox"]
            box2d = (x, y, x + w, y + h)
        except Exception:
            box2d = None
        corners3d = None
        if inst.get("bbox3D_cam") is not None:
            try:
                c = np.array(inst["bbox3D_cam"], dtype=np.float64)
                if c.shape == (8, 3):
                    corners3d = c
            except Exception:
                pass
        if corners3d is None and all(k in inst for k in ("center_cam", "dimensions", "pose")):
            try:
                c = np.array(inst["center_cam"], dtype=np.float64).reshape(-1)
                dd = np.array(inst["dimensions"], dtype=np.float64).reshape(-1)
                p = np.array(inst["pose"], dtype=np.float64)
                if p.size == 9: p = p.reshape(3, 3)
                if c.size == 3 and dd.size == 3 and p.shape == (3, 3):
                    corners3d = cuboid_corners(c, dd, p)
            except Exception:
                pass
        cid = int(inst.get("category_id", 0))
        score = float(inst.get("score", 0))
        contig = DATASET_ID_TO_CONTIG.get(cid, cid)
        cls_name = CLASSES[contig] if 0 <= contig < len(CLASSES) else f"cls{cid}"
        out.append({
            "box2d_xyxy": box2d,
            "corners3d": corners3d,
            "color_bgr": class_color_bgr(cid),
            "label": f"{cls_name} {score:.2f}",
            "_idx": i,
        })
    return out


# =====================================================================
# Composite layout: 2 rows × N cols, each cell = front + novel side-by-side
# =====================================================================

def make_cell(im_base: np.ndarray, K: np.ndarray, instances: list,
              novel_size: int, thickness: int,
              with_grid: bool = True) -> np.ndarray:
    """Build a single cell: [3D-on-image | novel-view] side-by-side, banner above."""
    front = draw_3d_front_panel(im_base, instances, K, thickness)
    H = front.shape[0]
    nv_size = min(novel_size, H)
    novel = novel_view_panel(K, im_base.shape[:2], instances, nv_size,
                              pitch_rad=math.pi / 3, with_grid=with_grid,
                              thickness=thickness)
    if novel.shape[0] < H:
        pad = np.full((H - novel.shape[0], novel.shape[1], 3), 245, dtype=np.uint8)
        novel = np.vstack([novel, pad])
    elif novel.shape[0] > H:
        novel = cv2.resize(novel,
                           (int(novel.shape[1] * H / novel.shape[0]), H))
    cell = np.hstack([front, novel])
    return cell


def add_cell_banner(cell: np.ndarray, label: str,
                    banner_color=(45, 45, 50)) -> np.ndarray:
    h_banner = max(34, cell.shape[0] // 24)
    banner = np.full((h_banner, cell.shape[1], 3), 32, dtype=np.uint8)
    cv2.rectangle(banner, (0, h_banner - 3), (cell.shape[1], h_banner),
                  banner_color, -1)
    scale = h_banner / 55.0
    thick = max(2, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.putText(banner, label,
                ((cell.shape[1] - tw) // 2, (h_banner + th) // 2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick)
    return np.vstack([banner, cell])


def add_row_label(row_img: np.ndarray, label: str, label_w: int = 60) -> np.ndarray:
    """Prepend a vertical label band to a row."""
    H = row_img.shape[0]
    band_color = (250, 245, 240) if label == "ZS" else (240, 250, 240)
    band = np.full((H, label_w, 3), band_color, dtype=np.uint8)
    # Vertical text via rotation
    tmp = np.full((label_w, H, 3), band_color, dtype=np.uint8)
    scale = max(0.7, label_w / 60.0)
    thick = max(2, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.putText(tmp, label, ((H - tw) // 2, (label_w + th) // 2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (40, 40, 50), thick)
    band = cv2.rotate(tmp, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return np.hstack([band, row_img])


def assemble_row(cells: list[Tuple[str, np.ndarray]], gap: int = 8) -> np.ndarray:
    """Concatenate cells horizontally with a gap; pad shorter cells to max H."""
    bannered = [add_cell_banner(c, label) for label, c in cells]
    Hmax = max(b.shape[0] for b in bannered)
    padded = []
    for b in bannered:
        if b.shape[0] < Hmax:
            pad = np.full((Hmax - b.shape[0], b.shape[1], 3), 32, dtype=np.uint8)
            padded.append(np.vstack([b, pad]))
        else:
            padded.append(b)
    if gap > 0:
        gap_img = np.full((Hmax, gap, 3), 32, dtype=np.uint8)
        out = padded[0]
        for nxt in padded[1:]:
            out = np.hstack([out, gap_img, nxt])
        return out
    return np.hstack(padded)


def assemble_grid(rows: Dict[str, list], gap: int = 8) -> np.ndarray:
    """Build full 2-row × N-col grid with row labels."""
    sorted_keys = sorted(rows.keys(),
                         key=lambda k: ROW_ORDER.index(k) if k in ROW_ORDER else 99)
    row_imgs = []
    for key in sorted_keys:
        row = assemble_row(rows[key], gap=gap)
        row = add_row_label(row, key)
        row_imgs.append(row)
    if len(row_imgs) == 1:
        return row_imgs[0]
    Wmax = max(r.shape[1] for r in row_imgs)
    padded = []
    for r in row_imgs:
        if r.shape[1] < Wmax:
            pad = np.full((r.shape[0], Wmax - r.shape[1], 3), 32, dtype=np.uint8)
            padded.append(np.hstack([r, pad]))
        else:
            padded.append(r)
    if gap > 0:
        gap_img = np.full((gap, Wmax, 3), 32, dtype=np.uint8)
        out = padded[0]
        for nxt in padded[1:]:
            out = np.vstack([out, gap_img, nxt])
        return out
    return np.vstack(padded)


# =====================================================================
# Main
# =====================================================================

def load_predictions(path: Path) -> Dict[int, list]:
    preds = torch.load(path, weights_only=False, map_location="cpu")
    return {int(e["image_id"]): e.get("instances", []) for e in preds}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--col", action="append", required=True,
                    help="'<row> / <label>=path'.  row in {ZS, FT}.")
    ap.add_argument("--every", type=int, default=200)
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--score-min", type=float, default=0.0)
    ap.add_argument("--max-dets", type=int, default=12)
    ap.add_argument("--novel-size", type=int, default=720)
    ap.add_argument("--thickness", type=int, default=4)
    ap.add_argument("--image-root", type=str, default="")
    ap.add_argument("--no-grid", action="store_true",
                    help="Disable the ground-plane grid in the novel-view BEV panel.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Parse cols
    row_specs: Dict[str, list[Tuple[str, Path]]] = defaultdict(list)
    for spec in args.col:
        if "=" not in spec:
            print(f"  WARN bad --col: {spec}"); continue
        full_label, path_s = spec.split("=", 1)
        if "/" in full_label:
            row_part, arch_part = full_label.split("/", 1)
            row_part = row_part.strip(); arch_part = arch_part.strip()
        else:
            row_part = full_label.strip(); arch_part = full_label.strip()
        row_key = canonical_row(row_part)
        p = Path(path_s.strip())
        if not p.exists():
            print(f"  WARN missing: {p}"); continue
        row_specs[row_key].append((arch_part, p))

    if not row_specs:
        print("FATAL: no valid columns"); return 2
    print("Rows:")
    for r, items in row_specs.items():
        print(f"  {r}: {[lbl for lbl, _ in items]}")

    print(f"Loading GT from {args.gt}")
    gt = json.loads(args.gt.read_text())
    img_by_id = {im["id"]: im for im in gt["images"]}

    # Load all predictions
    arch_preds: Dict[str, Dict[str, Dict[int, list]]] = {}
    for row_key, items in row_specs.items():
        arch_preds[row_key] = {}
        for label, path in items:
            print(f"  loading {row_key} / {label}: {path}")
            arch_preds[row_key][label] = load_predictions(path)

    # Common image_ids across all archs
    common = set(img_by_id.keys())
    for row_key, by_arch in arch_preds.items():
        for label, preds in by_arch.items():
            common = common & set(preds.keys())
    common = sorted(common)
    print(f"common image_ids: {len(common)}")

    selected = common[::args.every][:args.limit]
    print(f"will render {len(selected)} composites")

    n_done = 0
    for img_id in selected:
        info = img_by_id.get(img_id)
        if info is None: continue
        fp = info.get("file_path", "") or ""
        if args.image_root and fp:
            fp = args.image_root + fp
        img_path = Path(fp)
        if not img_path.is_absolute():
            img_path = Path("datasets") / img_path
        if not img_path.exists():
            continue
        im_base = cv2.imread(str(img_path))
        if im_base is None:
            continue
        K = np.array(info["K"], dtype=np.float64)

        rows: Dict[str, list[Tuple[str, np.ndarray]]] = {}
        for row_key, by_arch in arch_preds.items():
            row_cells = []
            for label, preds in by_arch.items():
                insts = pred_instances_with_class_colors(
                    preds.get(img_id, []), args.max_dets, args.score_min,
                )
                cell = make_cell(im_base, K, insts,
                                 novel_size=args.novel_size,
                                 thickness=args.thickness,
                                 with_grid=not args.no_grid)
                row_cells.append((label, cell))
            rows[row_key] = row_cells

        composite = assemble_grid(rows, gap=8)
        out_path = args.out / f"img_{img_id:06d}.jpg"
        cv2.imwrite(str(out_path), composite)
        n_done += 1
        if n_done <= 3 or n_done % 20 == 0:
            print(f"  wrote {out_path.name} ({composite.shape[1]}×{composite.shape[0]})")

    print(f"\nDone. {n_done} composites in {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
