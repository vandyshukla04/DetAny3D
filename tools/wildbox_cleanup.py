"""
Inventory + classify the WildBox-experiment artifact directories under
``exps/`` (DetAny3D side) and ``$OVMONO3D_REPO/output/`` (ovmono3d side).

Tags each directory as ``CANONICAL`` (keep — referenced by the
paper-ready 4-row report or the final FT checkpoint chain) or ``STALE``
(safe to delete — superseded by a v3 / final / ep2 run, smoke leftover,
or hung-attempt remnant).

Default mode: **inventory only** (nothing is deleted). Prints total
sizes + a `rm -rf` line you can copy/paste for each STALE directory.

To actually delete, pass ``--apply``. The script will refuse to delete
anything tagged CANONICAL even if you ask.

Canonical set (what the paper depends on):

  exps/wildbox_final_ft/         — 1-epoch fine-tune training run + ckpt
  exps/wildbox_final_ft_ep2/     — 2-epoch fine-tune training run + ckpt
  exps/wildbox_final_zeroshot_oracle_v3/   — ZS oracle predictions JSON
  exps/wildbox_zeroshot_gt2d_v3/           — ZS GT-2D predictions JSON
  exps/wildbox_final_ft_eval/0426-112003/  — FT 1-epoch eval JSON
  exps/wildbox_final_ft_eval_ep2/          — FT 2-epoch eval JSON
  data/pkls/wildbox/WildBox_{train,val}.pkl
  reports/RESULTS_DETANY3D.md
  reports/EXPERIMENT_DESIGN_DETANY3D.md
  logs/                          — keep all logs for paper transparency
  checkpoints/                   — model weights (sam, dino, unidepth, detany3d)

Plus the four ovmono3d-side scoring directories::

  $OVMONO3D_REPO/output/wildbox_detany3d_zs_v3/
  $OVMONO3D_REPO/output/wildbox_detany3d_zs_gt2d_v3/
  $OVMONO3D_REPO/output/wildbox_detany3d_ft_v3/
  $OVMONO3D_REPO/output/wildbox_detany3d_ft_ep2_v3/
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple


# Canonical dir patterns — KEEP. Substring match against directory name.
CANONICAL_DA3D = [
    "wildbox_final_ft/",
    "wildbox_final_ft_ep2/",
    "wildbox_final_zeroshot_oracle_v3/",
    "wildbox_zeroshot_gt2d_v3/",
    "wildbox_final_ft_eval_ep2/",
]
CANONICAL_OVMONO3D = [
    "wildbox_detany3d_zs_v3",
    "wildbox_detany3d_zs_gt2d_v3",
    "wildbox_detany3d_ft_v3",
    "wildbox_detany3d_ft_ep2_v3",
]


def _du_bytes(path: Path) -> int:
    total = 0
    try:
        for entry in path.rglob("*"):
            try:
                if entry.is_file():
                    total += entry.stat().st_size
            except (FileNotFoundError, PermissionError):
                pass
    except (FileNotFoundError, PermissionError):
        pass
    return total


def _humansize(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"


def _is_canonical(dir_name: str, patterns: List[str]) -> bool:
    """Substring match against any of the canonical patterns."""
    return any(p in dir_name + "/" for p in patterns)


def _classify(da3d_root: Path, ov_root: Path) -> List[Tuple[Path, str, int]]:
    """Walk both roots and tag each immediate-child dir as KEEP/STALE."""
    rows: List[Tuple[Path, str, int]] = []

    exps = da3d_root / "exps"
    if exps.exists():
        for child in sorted(exps.iterdir()):
            if not child.is_dir():
                continue
            tag = "CANONICAL" if _is_canonical(child.name + "/", CANONICAL_DA3D) else "STALE"
            # Special case: wildbox_final_ft_eval contains MULTIPLE timestamp
            # subdirs from different runs; only the 12:03 one is canonical for
            # the FT 1-epoch row. Tag the parent as MIXED so the user knows.
            if child.name == "wildbox_final_ft_eval":
                tag = "MIXED — keep 0426-112003 timestamp subdir; others are stale"
            rows.append((child, tag, _du_bytes(child)))

    if ov_root.exists():
        ov_out = ov_root / "output"
        if ov_out.exists():
            for child in sorted(ov_out.iterdir()):
                if not child.is_dir():
                    continue
                if not child.name.startswith("wildbox_detany3d"):
                    continue  # other people's runs; ignore
                tag = "CANONICAL" if _is_canonical(child.name, CANONICAL_OVMONO3D) else "STALE"
                rows.append((child, tag, _du_bytes(child)))

    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--da3d-root", type=Path,
                    default=Path("/storage3/3DOM/vshukla/DetAny3D"))
    ap.add_argument("--ovmono3d-repo", type=Path,
                    default=Path("/storage2/3DOM/vshukla/repos/ovmono3d"))
    ap.add_argument("--apply", action="store_true",
                    help="Actually delete STALE directories. Without this flag, "
                         "the script only prints what it would do.")
    args = ap.parse_args(argv)

    rows = _classify(args.da3d_root, args.ovmono3d_repo)

    print(f"DetAny3D root:  {args.da3d_root}")
    print(f"ovmono3d root:  {args.ovmono3d_repo}")
    print()
    print(f"{'TAG':<11} {'SIZE':>10}   PATH")
    print("-" * 100)

    total_canonical = 0
    total_stale = 0
    stale_dirs: List[Path] = []
    for path, tag, size in rows:
        if tag == "CANONICAL":
            total_canonical += size
        elif tag == "STALE":
            total_stale += size
            stale_dirs.append(path)
        print(f"{tag:<11} {_humansize(size):>10}   {path}")

    print("-" * 100)
    print(f"Canonical total: {_humansize(total_canonical)}")
    print(f"Stale total:     {_humansize(total_stale)} (would be freed)")
    print()

    if not stale_dirs:
        print("No stale dirs. Nothing to do.")
        return 0

    if args.apply:
        print("--apply was set; deleting STALE dirs now...")
        for path in stale_dirs:
            try:
                shutil.rmtree(path)
                print(f"  rm -rf  {path}")
            except (FileNotFoundError, PermissionError) as e:
                print(f"  FAIL    {path}: {e}")
        print("Done.")
    else:
        print("Suggested deletion (review first, then re-run with --apply OR copy/paste):")
        for path in stale_dirs:
            print(f"  rm -rf {path}")
        print()
        print("To actually delete: re-run with --apply.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
