#!/usr/bin/env python
"""Aggregate DetAny3D x WildBox per-seed eval rows into mean +/- std.

The multi-seed study's payoff: turn the N per-seed row dirs
(``wildbox_detany3d_ft_ep2_seed*_int1_v3/``) into one mean +/- sample-std
table so the paper can report, e.g., ``3D AP = 4.39 +/- 0.35`` over 5 seeds.

This is the DetAny3D analogue of ovmono3d's ``tools/aggregate_seed_ap.py``.
That one parses ovmono3d-format text logs; DetAny3D writes structured JSON
(``full_metrics/summary.json`` + ``bev_ap.json``), so this reads those
directly. UNIT NOTE: ``summary.json`` APs are fractions in [0,1] (multiplied
by 100 here to get percent); ``bev_ap.json`` per-class values are already in
percent -- both are normalised to percent below so the table is consistent.

Each --run-dir is one seed's row dir (the OUT_DIR that wildbox_multiseed.sh
writes), containing::

    full_metrics/summary.json     # 2D / 3D / 3D-Rel AP + per_class_AP
    bev_ap.json                   # per_class IoU=0.25 / IoU=0.50

Usage (glob-expand all 5 seeds; existing seed0/seed2 + new seed1/3/4)::

    python tools/aggregate_seed_ap_detany3d.py \\
        --run-dirs /path/to/wildbox_detany3d_ft_ep2_seed*_int1_v3 \\
        --out reports/ep2_multiseed

Writes ``table_multiseed.md``, ``table_multiseed.tex`` and ``aggregated.json``.
"""
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

SPECIES = ["giraffe", "grevys_zebra", "elephant", "plains_zebra", "rhino", "gazelle"]


def _load_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"WARN: cannot read {p}: {e}", file=sys.stderr)
        return None


def gather_one_seed(run_dir: Path) -> Optional[dict]:
    """Return normalised (percent) metrics for one seed, or None if unreadable."""
    summ = _load_json(run_dir / "full_metrics" / "summary.json")
    bev = _load_json(run_dir / "bev_ap.json")
    if summ is None and bev is None:
        return None

    def pc(mode: str, cls: str) -> Optional[float]:
        v = (summ or {}).get(mode, {}).get("per_class_AP", {}).get(cls)
        return v * 100.0 if v is not None else None

    def overall(mode: str) -> Optional[float]:
        v = (summ or {}).get(mode, {}).get("AP")
        return v * 100.0 if v is not None else None

    def bev_pc(cls: str, iou: str) -> Optional[float]:
        # bev per_class values are already in percent
        return (bev or {}).get("per_class", {}).get(cls, {}).get(iou)

    def bev_macro(iou: str) -> Optional[float]:
        vals = [bev_pc(c, iou) for c in SPECIES]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "overall": {
            "2D AP (COCO 0.50:0.95)": overall("2D"),
            "3D AP (0.05:0.50)": overall("3D"),
            "Rel-AP_3D (0.05:0.50)": overall("3D-Rel"),
            "AP_BEV @ 0.50 (macro)": bev_macro("IoU=0.50"),
            "AP_BEV @ 0.25 (macro)": bev_macro("IoU=0.25"),
        },
        "per_class": {
            "3D AP": {c: pc("3D", c) for c in SPECIES},
            "Rel-AP_3D": {c: pc("3D-Rel", c) for c in SPECIES},
            "2D AP": {c: pc("2D", c) for c in SPECIES},
            "AP_BEV @ 0.50": {c: bev_pc(c, "IoU=0.50") for c in SPECIES},
            "AP_BEV @ 0.25": {c: bev_pc(c, "IoU=0.25") for c in SPECIES},
        },
    }


def mean_std(xs: List[float]) -> Tuple[float, float]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)  # sample std (ddof=1)
    return m, math.sqrt(var)


def fmt_cell(m: float, s: float, n: int) -> str:
    if math.isnan(m):
        return "-"
    return f"{m:.2f} ± {s:.2f}" if n > 1 else f"{m:.2f}"


def build_markdown(seeds: List[dict], seed_labels: List[str],
                   classes: List[str]) -> Tuple[str, dict, dict]:
    n = len(seeds)
    lines: List[str] = []
    lines.append(f"# DetAny3D x WildBox -- {n}-seed aggregated results (ep2, oracle-2D, int1)\n")
    lines.append(f"_Seeds: {', '.join(seed_labels)}. Cells are mean ± sample std "
                 f"(ddof=1) across {n} seeds; all values in percent._\n")

    # -- Overall metrics --
    lines.append("## Overall\n")
    lines.append("| Metric | mean ± std |")
    lines.append("|---|---:|")
    overall_keys = list(seeds[0]["overall"].keys()) if seeds else []
    agg_overall: Dict[str, Tuple[float, float]] = {}
    for k in overall_keys:
        m, s = mean_std([sd["overall"].get(k) for sd in seeds])
        agg_overall[k] = (m, s)
        lines.append(f"| {k} | {fmt_cell(m, s, n)} |")
    lines.append("")

    # -- Per-class metrics --
    lines.append("## Per-class\n")
    lines.append("| Metric | " + " | ".join(classes) + " |")
    lines.append("|" + "---|" * (len(classes) + 1))
    pc_keys = list(seeds[0]["per_class"].keys()) if seeds else []
    agg_pc: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for metric in pc_keys:
        cells = []
        agg_pc[metric] = {}
        for c in classes:
            m, s = mean_std([sd["per_class"].get(metric, {}).get(c) for sd in seeds])
            agg_pc[metric][c] = (m, s)
            cells.append(fmt_cell(m, s, n))
        lines.append(f"| **{metric}** | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines), agg_overall, agg_pc


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dirs", nargs="+", required=True, type=Path,
                   help="Per-seed row dirs (glob-expandable), e.g. "
                        ".../wildbox_detany3d_ft_ep2_seed*_int1_v3")
    p.add_argument("--classes", nargs="+", default=SPECIES,
                   help="Class column order (default 6-species wildlife)")
    p.add_argument("--out", type=Path, required=True,
                   help="Output dir for table_multiseed.{md,tex} + aggregated.json")
    args = p.parse_args()

    seeds, labels = [], []
    for d in args.run_dirs:
        if not d.exists():
            print(f"WARN: {d} does not exist, skipping", file=sys.stderr)
            continue
        row = gather_one_seed(d)
        if row is None:
            print(f"WARN: {d} has no readable summary.json/bev_ap.json, skipping",
                  file=sys.stderr)
            continue
        seeds.append(row)
        labels.append(d.name)
        print(f"Loaded: {d.name}")

    if not seeds:
        sys.exit("No readable run dirs. Pass --run-dirs paths that contain "
                 "full_metrics/summary.json and bev_ap.json.")

    md, agg_overall, agg_pc = build_markdown(seeds, labels, args.classes)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "table_multiseed.md").write_text(md)
    (args.out / "aggregated.json").write_text(json.dumps({
        "n_seeds": len(seeds),
        "seeds": labels,
        "classes": args.classes,
        "overall": {k: {"mean": m, "std": s} for k, (m, s) in agg_overall.items()},
        "per_class": {metric: {c: {"mean": m, "std": s}
                               for c, (m, s) in cols.items()}
                      for metric, cols in agg_pc.items()},
    }, indent=2))

    print(f"\nWrote:\n  {args.out}/table_multiseed.md\n  {args.out}/aggregated.json\n")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
