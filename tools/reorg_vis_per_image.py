"""
Reorganize per-architecture viz folders into per-image folders.

Input  layout: output/viz_summary/<arch_label>/<image_filename>.png
Output layout: output/viz_summary_per_image/<imgid>/{nogrid,grid}/

For each image_id present in ≥2 architectures:
  output/viz_summary_per_image/<imgid>/
    nogrid/
      <arch_label>.png       — copy of that arch's vis for this image
      <arch_label>.png
      ...
    grid/
      composite.png          — single composite image with all archs in a grid

Robust to different filename conventions per arch — uses a regex to extract
the image_id (the integer that appears after the last separator and before
.png/.jpg).

Usage:
    python tools/reorg_vis_per_image.py \
        --src output/viz_summary \
        --dst output/viz_summary_per_image \
        [--min-archs 2]   # only keep image_ids present in >= N archs
"""
from __future__ import annotations

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

# Architecture label → display order (controls grid layout)
PREFERRED_ORDER = [
    "OVMono3D-LIFT_ZS_rpn",
    "OVMono3D-LIFT_ZS_oracle",
    "OVMono3D-LIFT_ZS_gt2d",
    "OVMono3D-LIFT_FT_init5sp",
    "OVMono3D-LIFT_FT_25k",
    "DetAny3D_ZS_oracle",
    "DetAny3D_ZS_gt2d",
    "DetAny3D_FT_2ep_seed0",
    "DetAny3D_FT_3ep_partial",
]


# Regex tries (in order) to pull a numeric image_id out of a filename.
# Add more fallbacks here if your filenames don't fit.
_ID_PATTERNS = [
    re.compile(r"image[_-]?(\d+)", re.IGNORECASE),       # image_123.png, image123.png
    re.compile(r"img[_-]?(\d+)", re.IGNORECASE),         # img_123.png
    re.compile(r"(\d+)_3d", re.IGNORECASE),              # 123_3d_overlay.png
    re.compile(r"frame[_-]?(\d+)", re.IGNORECASE),       # frame_123.png
    re.compile(r"^(\d+)[._-]"),                           # 123_*.png
    re.compile(r"^(\d+)\."),                              # 123.png
    re.compile(r"_(\d+)\.\w+$"),                          # arbitrary_123.png
]


def extract_id(filename: str) -> str | None:
    for p in _ID_PATTERNS:
        m = p.search(filename)
        if m:
            return m.group(1).lstrip("0") or "0"
    return None


def make_grid(images: list[Path], out_path: Path, n_cols: int = 3,
              cell_h: int = 360, label_height: int = 28) -> bool:
    """Compose images into a grid. Each cell is resized to (cell_h, ?) keeping aspect.
    Labels each cell with the architecture name (parent stem).
    Returns False if PIL is missing.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print(f"  WARN: PIL not available; skipping grid for {out_path.parent.name}")
        return False

    if not images:
        return False

    # Try to find a common font; fall back to default
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Resize each to a common height
    pil_imgs = []
    for p in images:
        try:
            im = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"    WARN: couldn't open {p.name}: {e}")
            continue
        w, h = im.size
        new_w = int(w * cell_h / h)
        im = im.resize((new_w, cell_h), Image.LANCZOS)
        # Add label band on top
        canvas = Image.new("RGB", (new_w, cell_h + label_height), (245, 245, 245))
        canvas.paste(im, (0, label_height))
        d = ImageDraw.Draw(canvas)
        label = p.stem  # arch label (we copied with that name in nogrid)
        d.text((4, 4), label, fill=(20, 20, 20), font=font)
        pil_imgs.append(canvas)

    if not pil_imgs:
        return False

    # Pad all to same width (use max width)
    max_w = max(im.size[0] for im in pil_imgs)
    cell_full_h = cell_h + label_height
    padded = []
    for im in pil_imgs:
        if im.size[0] == max_w:
            padded.append(im)
        else:
            canvas = Image.new("RGB", (max_w, cell_full_h), (245, 245, 245))
            canvas.paste(im, ((max_w - im.size[0]) // 2, 0))
            padded.append(canvas)

    n = len(padded)
    n_cols = min(n_cols, n)
    n_rows = (n + n_cols - 1) // n_cols

    grid_w = max_w * n_cols
    grid_h = cell_full_h * n_rows
    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    for i, im in enumerate(padded):
        r, c = divmod(i, n_cols)
        grid.paste(im, (c * max_w, r * cell_full_h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path, optimize=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("output/viz_summary"),
                    help="root containing per-arch subdirs")
    ap.add_argument("--dst", type=Path, default=Path("output/viz_summary_per_image"),
                    help="root for per-image output")
    ap.add_argument("--min-archs", type=int, default=2,
                    help="only keep image_ids present in >= N archs")
    ap.add_argument("--n-cols", type=int, default=3, help="grid columns")
    args = ap.parse_args()

    if not args.src.is_dir():
        print(f"FATAL: {args.src} not found")
        return 2

    # arch → image_id → source path
    arch_to_imgs: dict[str, dict[str, Path]] = defaultdict(dict)
    for arch_dir in sorted(args.src.iterdir()):
        if not arch_dir.is_dir():
            continue
        if arch_dir.name.startswith("_"):
            continue   # skip _comparison_sheet etc
        for f in arch_dir.iterdir():
            if f.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            img_id = extract_id(f.name)
            if img_id is None:
                continue
            # Keep first occurrence per (arch, image) — typically only one anyway
            if img_id not in arch_to_imgs[arch_dir.name]:
                arch_to_imgs[arch_dir.name][img_id] = f

    archs = list(arch_to_imgs.keys())
    print(f"Discovered {len(archs)} architectures with vis output:")
    for a in archs:
        print(f"  {a:<35}: {len(arch_to_imgs[a])} images")

    # Invert: image_id → list of (arch, src_path)
    img_to_archs: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for arch in archs:
        for img_id, src in arch_to_imgs[arch].items():
            img_to_archs[img_id].append((arch, src))

    # Filter to image_ids with >= min_archs coverage
    keep = {iid: pairs for iid, pairs in img_to_archs.items() if len(pairs) >= args.min_archs}
    print(f"\n{len(keep)} image_ids present in ≥{args.min_archs} archs (keeping)")
    print(f"{len(img_to_archs) - len(keep)} image_ids in fewer archs (dropped)")

    args.dst.mkdir(parents=True, exist_ok=True)

    # Order helper: sort archs by PREFERRED_ORDER, then alphabetic
    pref_idx = {a: i for i, a in enumerate(PREFERRED_ORDER)}
    def arch_key(a):
        return (pref_idx.get(a, len(PREFERRED_ORDER)), a)

    n_done = 0
    for img_id in sorted(keep.keys(), key=lambda x: int(x) if x.isdigit() else 1e18):
        per_img_dir = args.dst / f"img_{int(img_id):07d}" if img_id.isdigit() \
                      else args.dst / f"img_{img_id}"
        nogrid_dir = per_img_dir / "nogrid"
        grid_dir   = per_img_dir / "grid"
        nogrid_dir.mkdir(parents=True, exist_ok=True)
        grid_dir.mkdir(parents=True, exist_ok=True)

        # Copy per-arch images into nogrid/ with renamed-to-arch filename
        pairs = sorted(keep[img_id], key=lambda p: arch_key(p[0]))
        for arch, src in pairs:
            ext = src.suffix
            shutil.copy2(src, nogrid_dir / f"{arch}{ext}")

        # Build the grid composite from the nogrid images
        nogrid_files = sorted(
            (p for p in nogrid_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")),
            key=lambda p: arch_key(p.stem),
        )
        make_grid(nogrid_files, grid_dir / "composite.png", n_cols=args.n_cols)

        n_done += 1
        if n_done <= 5 or n_done % 25 == 0:
            print(f"  built {per_img_dir.name}: {len(pairs)} archs")

    print(f"\nWrote {n_done} per-image folders to {args.dst}")
    print(f"\nExample inspection:")
    print(f"  ls {args.dst}/ | head")
    print(f"  ls {args.dst}/$(ls {args.dst} | head -1)/")


if __name__ == "__main__":
    raise SystemExit(main())
