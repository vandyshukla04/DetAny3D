"""
Reorganize per-architecture viz folders into per-image folders.

OVMono3D's visualizer emits TWO files per image per architecture:
  img_NNNNNN.jpg          — multi-panel grid (input | GT | prediction)
  img_NNNNNN_nogrid.jpg   — clean single-panel prediction overlay

Input  layout: output/viz_summary/<arch_label>/img_<id>[_nogrid].jpg
Output layout: output/viz_summary_per_image/img_<id>/{grid,nogrid}/<arch_label>.jpg
                                              and        composite/composite.png

Per image_id present in ≥2 architectures:
  output/viz_summary_per_image/img_NNNNNN/
    grid/                — multi-panel rendering of each arch
      OVMono3D-LIFT_ZS_oracle.jpg
      ...
      DetAny3D_FT_2ep_seed0.jpg
    nogrid/              — clean single-panel rendering of each arch
      OVMono3D-LIFT_ZS_oracle.jpg
      ...
    composite/
      composite_nogrid.png   — single image with all archs side-by-side from nogrid

Usage:
    python tools/reorg_vis_per_image.py \
        --src output/viz_summary \
        --dst output/viz_summary_per_image \
        [--min-archs 2] [--n-cols 3]
"""
from __future__ import annotations

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

# Architecture label → display order in composite grid
PREFERRED_ORDER = [
    "OVMono3D-LIFT_ZS_oracle",
    "OVMono3D-LIFT_ZS_rpn",
    "OVMono3D-LIFT_ZS_gt2d",
    "OVMono3D-LIFT_FT_init5sp",
    "OVMono3D-LIFT_FT_25k",
    "DetAny3D_ZS_oracle",
    "DetAny3D_ZS_gt2d",
    "DetAny3D_FT_2ep_seed0",
    "DetAny3D_FT_3ep_partial",
]


# Filename: img_NNNNNN.jpg  or  img_NNNNNN_nogrid.jpg
# Captures (id, has_nogrid_suffix). Non-matching files are skipped.
_FILENAME_RE = re.compile(
    r"^img_(\d+)(_nogrid)?\.(?:jpg|jpeg|png)$",
    re.IGNORECASE,
)


def parse_filename(name: str) -> tuple[str, str] | None:
    """Return (image_id_str_zero_padded, kind) where kind is 'grid' or 'nogrid'."""
    m = _FILENAME_RE.match(name)
    if not m:
        return None
    img_id = m.group(1)        # keep the zero-padding so '000000' stays distinct
    kind = "nogrid" if m.group(2) else "grid"
    return img_id, kind


def make_composite(image_paths: list[tuple[str, Path]],
                   out_path: Path, n_cols: int = 3,
                   cell_h: int = 360, label_height: int = 28) -> bool:
    """Compose architecture images into a single labeled grid composite.

    image_paths: list of (arch_label, source_path).
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False
    if not image_paths:
        return False

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    cells = []
    for arch_label, p in image_paths:
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        w, h = im.size
        new_w = max(1, int(w * cell_h / h))
        im = im.resize((new_w, cell_h), Image.LANCZOS)
        # Top label band
        canvas = Image.new("RGB", (new_w, cell_h + label_height), (245, 245, 245))
        canvas.paste(im, (0, label_height))
        d = ImageDraw.Draw(canvas)
        d.text((6, 5), arch_label, fill=(20, 20, 20), font=font)
        cells.append(canvas)

    if not cells:
        return False

    max_w = max(c.size[0] for c in cells)
    full_h = cell_h + label_height
    padded = []
    for c in cells:
        if c.size[0] == max_w:
            padded.append(c)
        else:
            canvas = Image.new("RGB", (max_w, full_h), (245, 245, 245))
            canvas.paste(c, ((max_w - c.size[0]) // 2, 0))
            padded.append(canvas)

    n = len(padded)
    n_cols = min(n_cols, n)
    n_rows = (n + n_cols - 1) // n_cols
    grid_w = max_w * n_cols
    grid_h = full_h * n_rows
    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    for i, c in enumerate(padded):
        r, col = divmod(i, n_cols)
        grid.paste(c, (col * max_w, r * full_h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path, optimize=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("output/viz_summary"))
    ap.add_argument("--dst", type=Path, default=Path("output/viz_summary_per_image"))
    ap.add_argument("--min-archs", type=int, default=2)
    ap.add_argument("--n-cols", type=int, default=3)
    args = ap.parse_args()

    if not args.src.is_dir():
        print(f"FATAL: {args.src} not found")
        return 2

    # arch → image_id → {kind ('grid'/'nogrid') → Path}
    arch_imgs: dict[str, dict[str, dict[str, Path]]] = defaultdict(lambda: defaultdict(dict))

    for arch_dir in sorted(args.src.iterdir()):
        if not arch_dir.is_dir():
            continue
        if arch_dir.name.startswith("_"):
            continue
        for f in arch_dir.iterdir():
            if not f.is_file():
                continue
            parsed = parse_filename(f.name)
            if parsed is None:
                continue
            img_id, kind = parsed
            arch_imgs[arch_dir.name][img_id][kind] = f

    archs = list(arch_imgs.keys())
    print(f"Architectures discovered: {len(archs)}")
    for a in archs:
        n_imgs = len(arch_imgs[a])
        n_grid = sum(1 for d in arch_imgs[a].values() if "grid" in d)
        n_nogrid = sum(1 for d in arch_imgs[a].values() if "nogrid" in d)
        print(f"  {a:<35}: {n_imgs} unique IDs | {n_grid} grid + {n_nogrid} nogrid files")

    # Invert: image_id → arch → {kind: Path}
    img_to_archs: dict[str, dict[str, dict[str, Path]]] = defaultdict(dict)
    for arch in archs:
        for img_id, kinds in arch_imgs[arch].items():
            img_to_archs[img_id][arch] = kinds

    keep = {iid: data for iid, data in img_to_archs.items() if len(data) >= args.min_archs}
    print(f"\nimage_ids in ≥{args.min_archs} archs: {len(keep)}  (dropped {len(img_to_archs) - len(keep)})")

    args.dst.mkdir(parents=True, exist_ok=True)

    pref_idx = {a: i for i, a in enumerate(PREFERRED_ORDER)}
    arch_key = lambda a: (pref_idx.get(a, len(PREFERRED_ORDER)), a)

    n_done = 0
    for img_id in sorted(keep.keys(), key=lambda x: int(x) if x.isdigit() else x):
        per_img_dir = args.dst / f"img_{img_id}"
        grid_dir   = per_img_dir / "grid"
        nogrid_dir = per_img_dir / "nogrid"
        comp_dir   = per_img_dir / "composite"
        grid_dir.mkdir(parents=True, exist_ok=True)
        nogrid_dir.mkdir(parents=True, exist_ok=True)

        # Copy each arch's grid + nogrid versions
        archs_for_img = sorted(keep[img_id].keys(), key=arch_key)
        nogrid_paths_for_composite: list[tuple[str, Path]] = []
        for arch in archs_for_img:
            kinds = keep[img_id][arch]
            for kind, src in kinds.items():
                out_dir = grid_dir if kind == "grid" else nogrid_dir
                ext = src.suffix
                dst = out_dir / f"{arch}{ext}"
                shutil.copy2(src, dst)
                if kind == "nogrid":
                    nogrid_paths_for_composite.append((arch, dst))

        # Build a composite from the nogrid versions (cleanest for visual comparison)
        if nogrid_paths_for_composite:
            make_composite(
                nogrid_paths_for_composite,
                comp_dir / "composite_nogrid.png",
                n_cols=args.n_cols,
            )

        n_done += 1
        if n_done <= 3 or n_done % 25 == 0:
            print(f"  built {per_img_dir.name}: "
                  f"{len(archs_for_img)} archs, "
                  f"{sum(len(k) for k in keep[img_id].values())} files")

    print(f"\nDone. {n_done} per-image folders in {args.dst}")
    print(f"\nExample inspection:")
    print(f"  ls {args.dst}/ | head")
    if n_done:
        first = sorted(keep.keys(), key=lambda x: int(x) if x.isdigit() else x)[0]
        print(f"  ls {args.dst}/img_{first}/")
        print(f"  ls {args.dst}/img_{first}/grid/")
        print(f"  ls {args.dst}/img_{first}/nogrid/")


if __name__ == "__main__":
    raise SystemExit(main())
