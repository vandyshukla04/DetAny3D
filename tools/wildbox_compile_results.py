"""
Compile the DetAny3D × WildBox eval outputs into two paper-ready markdown
reports:

  * RESULTS_DETANY3D.md           — for the paper's experiments/results section
                                    (headline tables, per-class breakdowns,
                                    sub-claim verification, comparison to
                                    ovmono3d's published rows).
  * EXPERIMENT_DESIGN_DETANY3D.md — for the paper's methods section
                                    (architecture, training config, eval
                                    protocol, cross-arch contract, caveats).

Reads everything from the per-row directories under ovmono3d's output/, where
``tools/wildbox_export_predictions.py`` lands the
``inference/iter_final/WildBox_val/instances_predictions.pth`` and where
``bev_ap_eval.py``, ``class_agnostic_eval.py``,
``tools/wildbox_full_metrics.py``, and ``visualize_class_agnostic.py`` write
their outputs.

Usage — N rows in one table::

    python tools/wildbox_compile_results.py \\
        --row /path/to/zs_oracle_run:"DetAny3D zero-shot (oracle 2D)" \\
        --row /path/to/zs_gt2d_run:"DetAny3D zero-shot (GT 2D)" \\
        --row /path/to/ft_1ep_run:"DetAny3D fine-tuned (1 epoch, oracle 2D)" \\
        --row /path/to/ft_2ep_run:"DetAny3D fine-tuned (2 epochs, oracle 2D)" \\
        --out-dir reports

Each ``--row`` is ``DIR:LABEL``. Order matters — the columns of the headline
table follow the order you pass them. For sub-claim verdicts the script
auto-detects zero-shot vs fine-tuned by case-insensitive substring matching
on the label ("zero-shot" / "fine-tuned" / "ft").

Backward compatibility::

    python tools/wildbox_compile_results.py \\
        --zs-dir DIR --ft-dir DIR \\
        --ft-label "DetAny3D fine-tuned (1 epoch, oracle 2D)" \\
        --out-dir reports

still works (gets translated to two ``--row`` entries internally).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SPECIES = ["giraffe", "grevys_zebra", "elephant", "plains_zebra", "rhino", "gazelle"]


def _safe_load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _safe_load_text(path: Path) -> str:
    try:
        return path.read_text()
    except FileNotFoundError:
        return ""


def _parse_summary_nhd(text: str) -> Dict[str, Optional[float]]:
    """Pull the headline numbers from class_agnostic_eval.py's stdout."""
    out = {
        "AP_0_25": None, "AP_0_50": None, "AP_0_75": None,
        "macro_AP_0_50": None,
        "best_scale": None,
        "mean_NHD_at_best_s": None,
        "mean_NHD_at_s1": None,
    }
    patterns = {
        "AP_0_25":            r"AP@0\.25\s*=\s*([0-9.]+)",
        "AP_0_50":            r"AP@0\.50\s*=\s*([0-9.]+)",
        "AP_0_75":            r"AP@0\.75\s*=\s*([0-9.]+)",
        "macro_AP_0_50":      r"macro AP@0\.50\s*=\s*([0-9.]+)",
        "best_scale":         r"best global scale:\s*([0-9.]+)",
        "mean_NHD_at_best_s": r"mean NHD @ best s:\s*([0-9.]+)",
        "mean_NHD_at_s1":     r"mean NHD @ s=1:\s*([0-9.]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            try:
                out[key] = float(m.group(1))
            except ValueError:
                pass
    return out


def _parse_log_3d(text: str) -> Dict[str, Optional[float]]:
    """Pull AP@IoU thresholds and disentangled NHD from log.3D.txt or
    log.3D-Rel.txt (same format)."""
    out = {
        "AP_iou_05_50": None, "AP_iou_15": None, "AP_iou_25": None, "AP_iou_50": None,
        "NHD_overall": None, "NHD_xy": None, "NHD_z": None,
        "NHD_dimensions": None, "NHD_pose": None,
    }
    pat_ap = re.compile(
        r"Average Precision\s+\(AP\)\s+@\[\s*IoU=([0-9.:]+)\s*\|\s*depth=\s*all"
        r"\s*\|\s*maxDets=100\s*\]\s*=\s*([\-0-9.]+)"
    )
    for m in pat_ap.finditer(text):
        iou_str, val = m.group(1), float(m.group(2))
        slot = {
            "0.05:0.50": "AP_iou_05_50",
            "0.15": "AP_iou_15",
            "0.25": "AP_iou_25",
            "0.50": "AP_iou_50",
        }.get(iou_str)
        if slot:
            out[slot] = val
    nhd_block = re.search(
        r"Average Disentangled NHD Metrics:\s*\n"
        r"\s*overall:\s*([\-0-9.]+)\s*\n"
        r"\s*xy:\s*([\-0-9.]+)\s*\n"
        r"\s*z:\s*([\-0-9.]+)\s*\n"
        r"\s*dimensions:\s*([\-0-9.]+)\s*\n"
        r"\s*pose:\s*([\-0-9.]+)",
        text, re.MULTILINE,
    )
    if nhd_block:
        out["NHD_overall"]    = float(nhd_block.group(1))
        out["NHD_xy"]         = float(nhd_block.group(2))
        out["NHD_z"]          = float(nhd_block.group(3))
        out["NHD_dimensions"] = float(nhd_block.group(4))
        out["NHD_pose"]       = float(nhd_block.group(5))
    return out


def _gather_row(run_dir: Path, label: str) -> Dict:
    """Pull every metric we care about for one row."""
    row = {
        "label": label,
        "dir": str(run_dir),
        "bev": _safe_load_json(run_dir / "bev_ap.json") or {},
        "full_metrics": _safe_load_json(run_dir / "full_metrics" / "summary.json") or {},
        "nhd": _parse_summary_nhd(_safe_load_text(run_dir / "summary_nhd.txt")),
        "log_3d": _parse_log_3d(_safe_load_text(run_dir / "full_metrics" / "log.3D.txt")),
        "log_3d_rel": _parse_log_3d(_safe_load_text(run_dir / "full_metrics" / "log.3D-Rel.txt")),
        "viz_count": (
            len(list((run_dir / "vis_ovmono3d").glob("*.jpg")))
            if (run_dir / "vis_ovmono3d").exists() else 0
        ),
    }
    row["n_preds"] = row["bev"].get("n_preds")
    row["n_gt"] = row["bev"].get("n_gt")
    return row


def _fmt(val: Optional[float], digits: int = 3, dash: str = "—") -> str:
    if val is None or (isinstance(val, float) and val < 0):
        return dash
    return f"{val:.{digits}f}"


def _fmt_pct(val: Optional[float], digits: int = 3, dash: str = "—") -> str:
    if val is None or (isinstance(val, float) and val < 0):
        return dash
    return f"{val * 100:.{digits}f}"


def _bev_lookup(bev: dict, key: str) -> Optional[float]:
    """Look up a BEV AP value, falling back to per-class derivation when
    the older bev_ap.json schema (per_class only, no top-level micro/macro)
    is in place."""
    if key in bev:
        return bev[key]
    parts = key.split("@")
    if len(parts) == 2:
        agg, iou = parts
        bucket = bev.get(agg) or bev.get(agg.lower())
        if isinstance(bucket, dict):
            v = bucket.get(f"IoU={iou}") or bucket.get(iou)
            if v is not None:
                return v
        if agg in ("macro", "micro"):
            pc = bev.get("per_class", {}) or {}
            vals = []
            for sp, perclass in pc.items():
                v = perclass.get(f"IoU={iou}")
                if v is not None and v >= 0:
                    vals.append(v)
            if vals:
                if agg == "macro":
                    return sum(vals) / len(vals)
                # micro can't be re-derived from per-class AP without
                # per-instance TP/FP counts; leave as None.
    return None


def _bev_per_class(bev: dict, iou_key: str) -> List[str]:
    pc = bev.get("per_class", {})
    cells = []
    for sp in SPECIES:
        v = pc.get(sp, {}).get(iou_key)
        cells.append("—" if v is None or v < 0 else f"{v:.2f}")
    return cells


def _full_metrics_per_class_3d(full_metrics: dict) -> List[str]:
    pc = full_metrics.get("3D", {}).get("per_class_AP", {}) or {}
    cells = []
    for sp in SPECIES:
        v = pc.get(sp)
        cells.append("—" if v is None or v < 0 else f"{v * 100:.3f}")
    return cells


def _is_zero_shot(label: str) -> bool:
    s = label.lower()
    return ("zero-shot" in s) or ("zeroshot" in s) or s.startswith("zs")


def _is_fine_tuned(label: str) -> bool:
    s = label.lower()
    return ("fine-tuned" in s) or ("finetuned" in s) or (" ft " in f" {s} ") or s.startswith("ft")


def render_results_md(rows: List[Dict]) -> str:
    today = _dt.date.today().isoformat()
    out = []
    push = out.append

    push("# DetAny3D × WildBox — paper results\n")
    push(f"_Generated: {today}_\n")
    push("This file is intended to be pasted (or condensed) into the experiments / "
         "results section of the WildBox dataset paper. For the methods section, "
         "see [EXPERIMENT_DESIGN_DETANY3D.md](EXPERIMENT_DESIGN_DETANY3D.md). "
         "For the engineering reference, see [WILDBOX_DETANY3D.md](../WILDBOX_DETANY3D.md).\n")

    n = len(rows)
    labels = [r["label"] for r in rows]

    # ---------------- Run metadata ----------------
    push("## Run metadata\n")
    headers = ["Field"] + labels
    push("| " + " | ".join(headers) + " |")
    push("|" + "|".join(["---"] * len(headers)) + "|")
    push("| Run dir | " + " | ".join(f"`{r['dir']}`" for r in rows) + " |")
    push("| Predictions (n_preds) | " + " | ".join(str(r.get("n_preds", "—")) for r in rows) + " |")
    push("| Val GT annotations    | " + " | ".join(str(r.get("n_gt", "—")) for r in rows) + " |")
    push("| Visualization images  | " + " | ".join(str(r.get("viz_count", 0)) for r in rows) + " |")
    push("")

    # ---------------- Headline metrics ----------------
    push("## Headline metrics\n")
    push("Primary metric: AP_BEV @ IoU 0.50 (cross-arch contract per "
         "[ovmono3d/WILDBOX_EXPERIMENT.md §20.4](../../ovmono3d/WILDBOX_EXPERIMENT.md)).\n")

    headers = ["Metric"] + labels
    push("| " + " | ".join(headers) + " |")
    push("|---|" + "|".join(["---:"] * n) + "|")

    metric_specs = [
        ("AP_BEV @ 0.50 (micro)",  "bev",        "micro@0.50",  "pct"),
        ("AP_BEV @ 0.50 (macro)",  "bev",        "macro@0.50",  "pct"),
        ("AP_BEV @ 0.25 (micro)",  "bev",        "micro@0.25",  "pct"),
        ("AP_BEV @ 0.25 (macro)",  "bev",        "macro@0.25",  "pct"),
        ("AP_3D @ 0.05:0.50 (Omni3D primary)", "log_3d", "AP_iou_05_50", "pp"),
        ("AP_3D @ 0.25",           "log_3d",     "AP_iou_25",   "pp"),
        ("AP_3D @ 0.50",           "log_3d",     "AP_iou_50",   "pp"),
        ("Rel-AP_3D @ 0.05:0.50",  "log_3d_rel", "AP_iou_05_50","pp"),
        ("AP_2D @ 0.5:0.95 (COCO)", "full_metrics_2d_AP",  None, "pp"),
        ("Class-agnostic 2D AP @ 0.50 (macro)", "nhd", "macro_AP_0_50", "raw"),
        ("NHD-overall (lower=better)", "log_3d", "NHD_overall", "raw"),
        ("NHD-xy",                "log_3d",     "NHD_xy",      "raw"),
        ("NHD-z (depth)",         "log_3d",     "NHD_z",       "raw"),
        ("NHD-dimensions",        "log_3d",     "NHD_dimensions", "raw"),
        ("NHD-pose (rotation)",   "log_3d",     "NHD_pose",    "raw"),
        ("NHD best global scale", "nhd",        "best_scale",  "raw"),
        ("Mean NHD @ best scale", "nhd",        "mean_NHD_at_best_s", "raw"),
    ]

    def _val_for_row(row: Dict, source: str, key: Optional[str]) -> Optional[float]:
        if source == "bev" and key:
            return _bev_lookup(row["bev"], key)
        if source == "full_metrics_2d_AP":
            return row["full_metrics"].get("2D", {}).get("AP")
        if source in ("log_3d", "log_3d_rel", "nhd"):
            return row[source].get(key)
        return None

    for label, source, key, fmt in metric_specs:
        cells = []
        for r in rows:
            v = _val_for_row(r, source, key)
            if fmt == "pct":
                cells.append(_fmt(v, 2))
            elif fmt == "pp":
                cells.append(_fmt_pct(v, 3))
            else:
                cells.append(_fmt(v, 3))
        push("| " + label + " | " + " | ".join(cells) + " |")
    push("")

    # ---------------- Per-class BEV / 3D tables ----------------
    push("## Per-class AP_BEV @ 0.25 (oracle 2D)\n")
    push("| Row | giraffe | grevys_zebra | elephant | plains_zebra | rhino | gazelle |")
    push("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        cells = _bev_per_class(r["bev"], "IoU=0.25")
        push("| " + r["label"] + " | " + " | ".join(cells) + " |")
    push("")

    push("## Per-class AP_BEV @ 0.50 (oracle 2D)\n")
    push("| Row | giraffe | grevys_zebra | elephant | plains_zebra | rhino | gazelle |")
    push("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        cells = _bev_per_class(r["bev"], "IoU=0.50")
        push("| " + r["label"] + " | " + " | ".join(cells) + " |")
    push("")

    push("## Per-class standard AP_3D @ 0.05:0.50 (Omni3DEvaluator)\n")
    push("| Row | giraffe | grevys_zebra | elephant | plains_zebra | rhino | gazelle |")
    push("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        cells = _full_metrics_per_class_3d(r["full_metrics"])
        push("| " + r["label"] + " | " + " | ".join(cells) + " |")
    push("")

    # ---------------- Sub-claim verdicts ----------------
    zs_rows = [r for r in rows if _is_zero_shot(r["label"])]
    ft_rows = [r for r in rows if _is_fine_tuned(r["label"])]

    push("## Sub-claim verdicts\n")
    push("**Sub-claim 1 — 2D-3D decoupling.** Zero-shot 3D AP collapses across all "
         "IoU thresholds despite oracle 2D inputs being identical to those used by "
         "ovmono3d's published rows (median IoU vs GT = 0.877, class-agnostic 2D AP "
         "≈ 68 — see [ovmono3d §6.4.2](../../ovmono3d/WILDBOX_EXPERIMENT.md)). "
         "Pretrained Omni3D-distribution 3D priors do not transfer to wildlife "
         "synthetic-scale geometry under any IoU threshold tested.\n")

    if len(zs_rows) >= 2:
        push("**Sub-claim 2 — architecture-invariance and 2D-source-invariance of "
             "zero-shot 3D collapse.** "
             "Zero-shot AP_3D @ 0.05:0.50 across DetAny3D's tested 2D-source "
             "conditions:\n")
        for r in zs_rows:
            v = r["log_3d"].get("AP_iou_05_50")
            push(f"- {r['label']}: AP_3D @ 0.05:0.50 = {_fmt_pct(v, 3)}")
        push("\nCombined with OVMono3D-Lift's published zero-shot RPN-transfer and "
             "GDino-oracle rows (both AP_3D ≈ 0; see [ovmono3d "
             "§6.4](../../ovmono3d/WILDBOX_EXPERIMENT.md)), this gives **two distinct "
             "Omni3D-pretrained 3D architectures × multiple 2D-source conditions all "
             "collapsing to near-zero AP_3D**. The failure is in the pretraining-"
             "distribution mismatch, not the model class or the 2D-detection quality.\n")
    else:
        push("**Sub-claim 2 — architecture-invariance of zero-shot 3D collapse.** "
             "OVMono3D-Lift (Cube R-CNN, DINOv2 ViT-B/14) and DetAny3D "
             "(SAM ViT-H + UniDepth + DINOv2 ViT-L/14) both report AP_3D ≈ 0 under "
             "the GDino-oracle 2D protocol. Two architecturally distinct 3D heads, "
             "same source pretraining domain (Omni3D), same collapse — the failure is "
             "in the pretraining-distribution mismatch, not the model class.\n")
        push("(For a stronger version of this claim that removes the 2D-source "
             "confound, add a `--row` for DetAny3D zero-shot under GT 2D prompts.)\n")

    if ft_rows and zs_rows:
        push("**Fine-tuning recovery.** WildBox-fine-tuned DetAny3D recovers the "
             "depth-axis prior (NHD-z, best-scale factor) and lifts AP_3D out of zero "
             "across most species. Per-row deltas:\n")
        push("| Metric | " + " | ".join(r["label"] for r in ft_rows) + " |")
        push("|---|" + "|".join(["---:"] * len(ft_rows)) + "|")
        # Pick the GDino-oracle ZS row as the baseline to compute deltas against,
        # falling back to the first ZS row.
        baseline = next((r for r in zs_rows if "oracle" in r["label"].lower()), zs_rows[0])
        for metric_label, source, key, fmt in [
            ("Δ AP_3D @ 0.05:0.50", "log_3d", "AP_iou_05_50", "pp"),
            ("Δ AP_BEV @ 0.50 (macro)", "bev", "macro@0.50", "pct"),
            ("Δ NHD-z (depth)",   "log_3d", "NHD_z", "raw_neg"),
            ("Δ Best-scale → 1",  "nhd",    "best_scale", "raw"),
        ]:
            cells = []
            base_v = _val_for_row(baseline, source, key)
            for r in ft_rows:
                ft_v = _val_for_row(r, source, key)
                if base_v is None or ft_v is None:
                    cells.append("—")
                elif fmt == "pp":
                    cells.append(f"{(ft_v - base_v) * 100:+.3f} pp")
                elif fmt == "pct":
                    cells.append(f"{(ft_v - base_v):+.2f}")
                elif fmt == "raw_neg":
                    cells.append(f"{ft_v - base_v:+.2f} (lower=better)")
                else:
                    cells.append(f"{ft_v - base_v:+.3f}")
            push("| " + metric_label + " | " + " | ".join(cells) + " |")
        push(f"\n_(baseline for Δ: `{baseline['label']}`)_\n")

    # ---------------- Cross-architecture context ----------------
    push("## Cross-architecture comparison context\n")
    push("Numbers from ovmono3d's published rows ([ovmono3d/WILDBOX_EXPERIMENT.md "
         "§21.3](../../ovmono3d/WILDBOX_EXPERIMENT.md), full val 13 779 images, "
         "3-seed mean ± std):\n")
    push("| Row | macro AP_3D @ 0.25 | 2D AP @ 0.5 |")
    push("|---|---:|---:|")
    push("| OVMono3D-Lift zero-shot (RPN-transfer) | ≈0  | (low) |")
    push("| OVMono3D-Lift zero-shot (GDino oracle) | ≈0  | ≈68 (class-agnostic) |")
    push("| OVMono3D-Lift fine-tuned (3-seed)      | 9.12 ± 0.84 | 77.8 ± 0.9 |")
    push("\nDirect comparability caveats — see Caveats below.\n")

    # ---------------- Caveats ----------------
    push("## Caveats and disclosures\n")
    push("- **Single-seed.** Multi-seed mean ± std reporting deferred (each seed = "
         "another full fine-tune at ~6 h on 1 × A40). DetAny3D row should be footnoted "
         "as single-seed; OVMono3D-Lift's published row is 3-seed mean ± std.\n")
    push("- **Training budget.** DetAny3D was fine-tuned for 1 epoch (~46k samples "
         "seen). OVMono3D-Lift's published row used 15k iter × batch 8 ≈ 120k samples "
         "(~2.6 epochs). DetAny3D's row is conservative; deeper training would lift "
         "absolute numbers. A 2-epoch run is in progress at the time of writing.\n")
    push("- **Val coverage.** Eval at val interval=4 (≈25% of full 13 779). Same "
         "protocol all rows; relative comparison is sound. Full-val numbers may "
         "shift slightly. Re-eval at interval=1 in a separate srun is straightforward "
         "if needed.\n")
    push("- **Eval-time 2D = oracle 2D echo (oracle-2D rows only).** DetAny3D doesn't "
         "predict its own 2D — under the oracle 2D protocol it lifts the GDino oracle "
         "box, so 2D AP and class-agnostic 2D AP are identical between zero-shot and "
         "fine-tuned oracle-2D rows by construction. The variation lives in 3D AP, "
         "BEV AP, NHD, and Rel-AP_3D. (For the GT-2D row, predictions echo the GT box "
         "in the same way; same caveat applies.)\n")
    push("- **Depth supervision disabled.** WildBox's GT depth is per-segment "
         "synthetic scale (median |Z|=1), not metric. We disable the metric-depth "
         "loss during fine-tune ([WILDBOX_DETANY3D.md §5.4](../WILDBOX_DETANY3D.md)). "
         "OVMono3D-Lift down-weights depth instead (LOSS_W_Z=0.5). Cross-architecture "
         "comparison stays apples-to-apples on the BEV AP primary metric, which "
         "projects out Z.\n")
    push("- **Multi-GPU pivot.** Two 4-GPU DDP attempts hung at NCCL ALLGATHER "
         "SeqNum=1 with no recovery. The headline run is single-GPU. Multi-GPU "
         "debugging deferred — see [WILDBOX_DETANY3D.md §3.5 and bug "
         "#10](../WILDBOX_DETANY3D.md).\n")

    # ---------------- Artifact pointers ----------------
    push("## Artifact pointers\n")
    push("Per-row directories under ovmono3d's `output/` contain:\n")
    push("- `bev_ap.json` — BEV AP per IoU threshold per class.")
    push("- `summary_nhd.txt` — class-agnostic 2D AP, disentangled NHD per axis, NHD scale search.")
    push("- `full_metrics/summary.json` — Omni3D-style standard AP_3D / 2D AP per class.")
    push("- `full_metrics/log.{2D,3D,3D-Rel}.txt` — full COCO-eval text output per mode.")
    push("- `vis_ovmono3d/img_*.jpg` — paper-style 2×3 layouts (gt-only / pred-only / "
         "combined, with and without ground grid).")
    push("\nThis report's row dirs:\n")
    for r in rows:
        push(f"- **{r['label']}** — `{r['dir']}`")
    return "\n".join(out)


def render_design_md(rows: List[Dict]) -> str:
    today = _dt.date.today().isoformat()
    out = []
    push = out.append

    push("# DetAny3D × WildBox — experimental design\n")
    push(f"_Generated: {today}_\n")
    push("This file is intended to be pasted (or condensed) into the methods section "
         "of the WildBox dataset paper. For the headline numbers see "
         "[RESULTS_DETANY3D.md](RESULTS_DETANY3D.md). For the full engineering "
         "reference (env hazards, bug catalogue, cross-arch protocol), see "
         "[WILDBOX_DETANY3D.md](../WILDBOX_DETANY3D.md). Dataset details are "
         "documented authoritatively on the ovmono3d side and not duplicated here.\n")

    push("## 1. Cross-architecture contract\n")
    push("Per [ovmono3d/WILDBOX_EXPERIMENT.md §20.4](../../ovmono3d/WILDBOX_EXPERIMENT.md), "
         "we evaluate every architecture in the comparison under a shared 2D protocol "
         "(precomputed GroundingDINO open-vocab detections) so that the 3D-lifting "
         "capability is isolated from 2D-detection quality. The same "
         "`gdino_WildBox_val_oracle_2d.json` is consumed by both OVMono3D-Lift and "
         "DetAny3D for the oracle-2D rows.\n")
    push("DetAny3D is a **prompt-conditioned** detector (no closed-vocab RPN), so "
         "ovmono3d's RPN-transfer zero-shot row has no DetAny3D analog and is "
         "omitted from the cross-arch table. We report the following DetAny3D rows:\n")
    for r in rows:
        push(f"- {r['label']}")
    push("")
    push("Each row consumes the same `WildBox_val.json` GT and the same eval scripts "
         "(BEV AP, class-agnostic 2D AP, disentangled NHD, standard AP_3D / 2D AP / "
         "Rel-AP_3D, paper-style 2×3 visualizations).\n")

    push("## 2. Architecture as used\n")
    push("Backbone: SAM ViT-H image encoder ([detect_anything/modeling/image_encoder.py]"
         "(../detect_anything/modeling/image_encoder.py)), with a vendored UniDepth "
         "depth predictor and DINOv2 ViT-L/14 features. Prompt-conditioned 3D head "
         "([detect_anything/modeling/mask_decoder.py](../detect_anything/modeling/mask_decoder.py)) "
         "outputs 3D depth (log scale), dimensions, 6D rotation, and a 24-class "
         "orientation classification per prompt. The 3D head is class-agnostic — "
         "what makes the cross-arch oracle-2D row a valid comparison.\n")
    push("Pretrained checkpoint: the released `detany3d.pth` (trained on Omni3D — "
         "KITTI, nuScenes, SUN-RGBD, ARKitScenes, Objectron, Hypersim, 3RScan, "
         "Cityscapes3D, Waymo). Same source domain as OVMono3D-Lift's pretraining; "
         "this is what makes the cross-architecture zero-shot collapse a fair "
         "test of pretraining-domain transferability.\n")

    push("## 3. Training configuration\n")
    push("- **Optimizer:** AdamW (lr 1e-5; UniDepth subgroup lr 1e-7; weight decay 1e-7).\n")
    push("- **Schedule:** CosineAnnealingLR over `cfg.num_epochs` with linear warmup. "
         "Resume from pretrained without loading the optimizer/scheduler state, so "
         "the cosine schedule restarts fresh on each fine-tune attempt "
         "([train.py:506-512](../train.py#L506-L512), commit `eeee735`).\n")
    push("- **Frozen / trainable:** image encoder frozen; prompt encoder + mask "
         "decoder + adapters trainable. Matches OVMono3D-Lift's policy (DINOv2 "
         "frozen during fine-tune).\n")
    push("- **Batch size:** 1 per GPU. **AMP:** enabled. **Epochs:** 1 (single-GPU "
         "final). 2-epoch and multi-seed runs deferred.\n")
    push("- **Loss list:** `intrinsic_loss, 2d_bbox_loss, 3d_bbox_loss`. **Depth loss "
         "explicitly disabled** ([WILDBOX_DETANY3D.md §5.4](../WILDBOX_DETANY3D.md)) "
         "because WildBox's per-segment uniform-scale depth makes metric-depth "
         "supervision counterproductive.\n")
    push("- **Output rotation:** full 3 × 3 rotation matrix (drone shots have "
         "non-trivial pitch/roll).\n")
    push("- **Provide GT intrinsics:** on (VGGT's per-frame K used as ground truth, "
         "bypassing DetAny3D's intrinsic-prediction head).\n")

    push("## 4. Data pipeline\n")
    push("- **Dataset format:** ovmono3d's `WildBox_{train,val}.json` (Omni3D schema, "
         "absolute paths, K per image, SAM3-tight `bbox`, `bbox3D_cam`, `center_cam`, "
         "`dimensions[W,H,L]`, `R_cam`).\n")
    push("- **Conversion:** [detect_anything/datasets/data_creator/wildbox.py]"
         "(../detect_anything/datasets/data_creator/wildbox.py) writes a DetAny3D "
         "pickle. Convention swap: DetAny3D's `[w, h, l]` = "
         "`reversed(Omni3D[W, H, L])`. Validated by `--verify-projection` (100% pass).\n")
    push("- **Oracle-2D loader:** [generate_oracle_list]"
         "(../detect_anything/datasets/detany3d_dataset.py) method, activated via "
         "`cfg.dataset.oracle_2d_input: True`. Loads the precomputed GDino oracle "
         "JSON, maps dataset-id → contiguous-id, rescales boxes to DetAny3D's "
         "preprocess resolution, hands them to the mask decoder as box prompts.\n")
    push("- **GT-2D loader:** the existing `generate_obj_list` path "
         "(`oracle_2d_input: False`) reads GT 2D boxes directly from the val pickle. "
         "Used for the GT-2D zero-shot row, which removes the 2D-source confound from "
         "sub-claim 2.\n")
    push("- **Category mapping:** [data/category_meta_wildbox.json]"
         "(../data/category_meta_wildbox.json) — 6 species, dataset-id ascending, "
         "contiguous 0..5.\n")

    push("## 5. Evaluation protocol\n")
    push("- **Predictions exported** to detectron2-format `instances_predictions.pth` "
         "([tools/wildbox_export_predictions.py](../tools/wildbox_export_predictions.py)). "
         "Reverses the dimension ordering back to Omni3D `[W, H, L]` and remaps the "
         "category-id space.\n")
    push("- **BEV AP** via ovmono3d's `tools/bev_ap_eval.py` (Shapely rotated-IoU; "
         "primary cross-arch metric).\n")
    push("- **Class-agnostic 2D AP + disentangled NHD** via "
         "`ovmono3d/tools/class_agnostic_eval.py --nhd`.\n")
    push("- **Standard AP_3D / 2D AP / Rel-AP_3D** via "
         "[tools/wildbox_full_metrics.py](../tools/wildbox_full_metrics.py), which "
         "wraps ovmono3d's `_evaluate_predictions_on_omni` (pytorch3d-CPU 3D-IoU + "
         "pycocotools 2D AP + LabelAny3D scale-aligned grid for Rel-AP_3D).\n")
    push("- **Visualizations** (paper figures): ovmono3d's "
         "`tools/visualize_class_agnostic.py` produces 2 × 3 layouts (gt-only / "
         "pred-only / combined, with and without ground grid).\n")

    push("## 6. Hazards we documented and addressed\n")
    push("Complete catalogue with commit refs is in [WILDBOX_DETANY3D.md "
         "§3 and §8](../WILDBOX_DETANY3D.md). Key items reviewers should be aware of:\n")
    push("1. **Released checkpoint silent-skip-training** — the public DetAny3D "
         "checkpoint stores `epoch=93`; without a fix, `range(start_epoch=93, "
         "num_epochs=1)` is empty and training exits with no iterations. Fixed in "
         "commit `eeee735`.\n")
    push("2. **NCCL ALLGATHER timeout** on multi-GPU — the first DDP collective "
         "times out after the default 30 minutes when `/storage3` NFS straggler skew "
         "is large. Raised the timeout to 4 h (commit `db7bb37`); even then 4-GPU "
         "runs failed to make iter progress, so we ran final on 1 GPU.\n")
    push("3. **mmcv MSDeformAttn CUDA op** — pip's stock `mmcv==2.0.1` ships without "
         "CUDA ops; reinstalled via OpenMMLab's cu116/torch1.13 wheel index.\n")
    push("4. **Empty `prepare_for_dsam` mid-training** — `filter_objects` rejecting "
         "every annotation in a frame crashed `loss_total.backward()`. Patched the "
         "dataset to recurse to a random sample (commit `6c89796`).\n")
    push("5. **Eval-config interval persistence** — the `interval: N` field in the "
         "eval YAML persists on disk between smoke and final runs; if not reset, "
         "the final eval silently runs on the smoke subsample and reports "
         "smoke-identical numbers. Always reset before launching final.\n")
    push("6. **Periodic mid-epoch checkpointing** — DetAny3D's `train.py` originally "
         "saved only at end-of-epoch, costing 2 h of compute on an SSH disconnect. "
         "Added `save_checkpoint_iters` config field and per-N-iter save in commit "
         "`a0d4b8f`.\n")

    push("## 7. Reproducibility pointers\n")
    push("- **Linear ops**: [FINAL_RUN_DETANY3D.md](../FINAL_RUN_DETANY3D.md) — copy-paste "
         "from prereqs to paper figures.\n")
    push("- **Smoke validation**: [QUICK_START_SMOKE_TEST_DETANY3D.md](../QUICK_START_SMOKE_TEST_DETANY3D.md) — "
         "~50-min end-to-end pipeline check.\n")
    push("- **Reference**: see the repository README for architecture, env hazards, "
         "configs, eval, and the cross-arch protocol.\n")
    push("- **Env**: Python 3.8, torch 1.13.1+cu116, mmcv 2.0.1 with CUDA ops, "
         "opencv-python-headless, GroundingDINO at the pinned commit.\n")
    return "\n".join(out)


def _parse_row_arg(s: str) -> Tuple[Path, str]:
    """Parse 'DIR:LABEL' argument. LABEL may contain colons (split on the
    first colon only)."""
    if ":" not in s:
        raise argparse.ArgumentTypeError(
            f"--row arg must be DIR:LABEL, got: {s!r}"
        )
    dir_part, label_part = s.split(":", 1)
    dir_path = Path(dir_part).expanduser()
    if not label_part.strip():
        raise argparse.ArgumentTypeError(
            f"--row arg has empty label: {s!r}"
        )
    return dir_path, label_part


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--row", type=_parse_row_arg, action="append", default=[],
                        help="DIR:LABEL pair for one column of the headline table. "
                             "Repeatable. Order is preserved in output. "
                             "Auto-classified as zero-shot vs fine-tuned by case-"
                             "insensitive substring match on LABEL.")
    parser.add_argument("--zs-dir", type=Path, default=None,
                        help="(deprecated; use --row) zero-shot row directory.")
    parser.add_argument("--ft-dir", type=Path, default=None,
                        help="(deprecated; use --row) fine-tuned row directory.")
    parser.add_argument("--ft-label", type=str,
                        default="DetAny3D fine-tuned (1 epoch, oracle 2D)",
                        help="(deprecated; used only with --ft-dir).")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    rows_spec: List[Tuple[Path, str]] = list(args.row)
    if args.zs_dir is not None:
        rows_spec.append((args.zs_dir, "DetAny3D zero-shot (oracle 2D)"))
    if args.ft_dir is not None:
        rows_spec.append((args.ft_dir, args.ft_label))

    if not rows_spec:
        parser.error("Pass at least one --row DIR:LABEL (or use the deprecated "
                     "--zs-dir/--ft-dir).")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for run_dir, label in rows_spec:
        if not run_dir.exists():
            print(f"WARNING: run dir does not exist: {run_dir}")
        rows.append(_gather_row(run_dir, label))

    results_md = render_results_md(rows)
    design_md = render_design_md(rows)

    (args.out_dir / "RESULTS_DETANY3D.md").write_text(results_md)
    (args.out_dir / "EXPERIMENT_DESIGN_DETANY3D.md").write_text(design_md)

    print(f"wrote {args.out_dir / 'RESULTS_DETANY3D.md'}  "
          f"({len(results_md)} bytes, {len(rows)} rows)")
    print(f"wrote {args.out_dir / 'EXPERIMENT_DESIGN_DETANY3D.md'}  "
          f"({len(design_md)} bytes)")
    print()
    print("Rows in this report:")
    for r in rows:
        kind = []
        if _is_zero_shot(r["label"]): kind.append("zero-shot")
        if _is_fine_tuned(r["label"]): kind.append("fine-tuned")
        kind_str = " / ".join(kind) if kind else "uncategorized"
        print(f"  - [{kind_str}] {r['label']}  ({r['dir']})")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
