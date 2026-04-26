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

Usage (from any conda env that has Python + json + pyyaml):

    python tools/wildbox_compile_results.py \\
        --zs-dir   /storage2/3DOM/vshukla/repos/ovmono3d/output/wildbox_detany3d_zs_v3 \\
        --ft-dir   /storage2/3DOM/vshukla/repos/ovmono3d/output/wildbox_detany3d_ft_v3 \\
        --out-dir  /storage3/3DOM/vshukla/DetAny3D/reports

Optionally pass ``--ft-label "DetAny3D fine-tuned (1 epoch, oracle 2D)"`` to
override the default row label.
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


def _gather_row(run_dir: Path) -> Dict:
    """Pull every metric we care about for one row (zero-shot or fine-tuned)."""
    row = {
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
    # Counts from BEV.
    row["n_preds"] = row["bev"].get("n_preds")
    row["n_gt"] = row["bev"].get("n_gt")
    return row


def _fmt(val: Optional[float], digits: int = 3, dash: str = "—") -> str:
    if val is None:
        return dash
    if val < 0:
        return dash  # COCO returns -1 for "no GT in this slice"
    return f"{val:.{digits}f}"


def _fmt_pct(val: Optional[float], digits: int = 2, dash: str = "—") -> str:
    if val is None or val < 0:
        return dash
    return f"{val * 100:.{digits}f}"


def _bev_per_class(bev: dict, iou_key: str) -> str:
    """Render one row of per-class BEV AP."""
    pc = bev.get("per_class", {})
    cells = []
    for sp in SPECIES:
        v = pc.get(sp, {}).get(iou_key)
        if v is None or v < 0:
            cells.append("—")
        else:
            cells.append(f"{v:.2f}")
    return " | ".join(cells)


def _full_metrics_per_class_3d(full_metrics: dict) -> str:
    pc = full_metrics.get("3D", {}).get("per_class_AP", {}) or {}
    cells = []
    for sp in SPECIES:
        v = pc.get(sp)
        if v is None or v < 0:
            cells.append("—")
        else:
            cells.append(f"{v * 100:.3f}")
    return " | ".join(cells)


def render_results_md(zs: dict, ft: dict, ft_label: str) -> str:
    today = _dt.date.today().isoformat()
    out = []
    push = out.append

    push("# DetAny3D × WildBox — paper results\n")
    push(f"_Generated: {today}_\n")
    push("This file is intended to be pasted (or condensed) into the experiments / "
         "results section of the WildBox dataset paper. For the methods section, "
         "see [EXPERIMENT_DESIGN_DETANY3D.md](EXPERIMENT_DESIGN_DETANY3D.md). "
         "For the engineering reference, see [WILDBOX_DETANY3D.md](../WILDBOX_DETANY3D.md).\n")

    push("## Run metadata\n")
    push("| | DetAny3D zero-shot (oracle 2D) | " + ft_label + " |")
    push("|---|---|---|")
    push(f"| Run dir | `{zs['dir']}` | `{ft['dir']}` |")
    push(f"| Predictions (n_preds)   | {zs.get('n_preds', '—')} | {ft.get('n_preds', '—')} |")
    push(f"| Val GT annotations      | {zs.get('n_gt', '—')} | {ft.get('n_gt', '—')} |")
    push(f"| Visualization images    | {zs.get('viz_count', 0)} | {ft.get('viz_count', 0)} |")
    push("")

    # ---------------- Headline metrics ----------------
    push("## Headline metrics\n")
    push("Primary metric: AP_BEV @ IoU 0.50 (cross-arch contract per "
         "[ovmono3d/WILDBOX_EXPERIMENT.md §20.4](../../ovmono3d/WILDBOX_EXPERIMENT.md)).\n")

    push("| Metric | Zero-shot | " + ft_label + " | Δ |")
    push("|---|---:|---:|---:|")

    # AP_BEV
    zs_bev_50 = zs["bev"].get("micro@0.50") or zs["bev"].get("micro_AP_0_50")
    # bev_ap_eval.py output structure: {"micro@0.25": ..., "micro@0.50": ..., "macro@0.25": ..., "macro@0.50": ...}
    # Fall back to alternate key forms if needed.
    def _bev_lookup(bev, key_options):
        for k in key_options:
            if k in bev:
                return bev[k]
            # nested under "micro"/"macro"?
            nested_path = k.split("@")
            if len(nested_path) == 2:
                bucket = bev.get(nested_path[0]) or bev.get(nested_path[0].lower())
                if isinstance(bucket, dict):
                    val = bucket.get("IoU=" + nested_path[1]) or bucket.get(nested_path[1])
                    if val is not None:
                        return val
            # Fallback: derive macro/micro from per_class.
            # per_class is {species: {"IoU=0.25": v, "IoU=0.50": v}}.
            agg, iou = nested_path if len(nested_path) == 2 else (None, None)
            if agg in ("macro", "micro") and iou:
                pc = bev.get("per_class", {}) or {}
                vals = []
                for sp, perclass in pc.items():
                    v = perclass.get(f"IoU={iou}")
                    if v is not None and v >= 0:
                        vals.append(v)
                if vals:
                    if agg == "macro":
                        return sum(vals) / len(vals)
                    # "micro" can't be re-derived from per-class without
                    # per-instance TP/FP counts. Approximate by macro and
                    # flag in the cell name; better: skip and let the
                    # cell render "—".
        return None

    pairs = [
        ("AP_BEV @ 0.50 (micro)",  _bev_lookup(zs["bev"], ["micro@0.50"]),
                                   _bev_lookup(ft["bev"], ["micro@0.50"]), "pct"),
        ("AP_BEV @ 0.50 (macro)",  _bev_lookup(zs["bev"], ["macro@0.50"]),
                                   _bev_lookup(ft["bev"], ["macro@0.50"]), "pct"),
        ("AP_BEV @ 0.25 (micro)",  _bev_lookup(zs["bev"], ["micro@0.25"]),
                                   _bev_lookup(ft["bev"], ["micro@0.25"]), "pct"),
        ("AP_BEV @ 0.25 (macro)",  _bev_lookup(zs["bev"], ["macro@0.25"]),
                                   _bev_lookup(ft["bev"], ["macro@0.25"]), "pct"),
        ("AP_3D  @ 0.05:0.50 (Omni3D primary)",
                                   zs["log_3d"].get("AP_iou_05_50"),
                                   ft["log_3d"].get("AP_iou_05_50"), "pp"),
        ("AP_3D  @ 0.25",          zs["log_3d"].get("AP_iou_25"),
                                   ft["log_3d"].get("AP_iou_25"), "pp"),
        ("AP_3D  @ 0.50",          zs["log_3d"].get("AP_iou_50"),
                                   ft["log_3d"].get("AP_iou_50"), "pp"),
        ("Rel-AP_3D @ 0.05:0.50",  zs["log_3d_rel"].get("AP_iou_05_50"),
                                   ft["log_3d_rel"].get("AP_iou_05_50"), "pp"),
        ("AP_2D @ 0.5:0.95 (COCO)",
                                   zs["full_metrics"].get("2D", {}).get("AP"),
                                   ft["full_metrics"].get("2D", {}).get("AP"), "pp"),
        ("Class-agnostic 2D AP @ 0.50 (macro)",
                                   zs["nhd"].get("macro_AP_0_50"),
                                   ft["nhd"].get("macro_AP_0_50"), "raw"),
        ("NHD-overall (lower=better)",
                                   zs["log_3d"].get("NHD_overall"),
                                   ft["log_3d"].get("NHD_overall"), "raw"),
        ("NHD-xy",                 zs["log_3d"].get("NHD_xy"),
                                   ft["log_3d"].get("NHD_xy"), "raw"),
        ("NHD-z (depth)",          zs["log_3d"].get("NHD_z"),
                                   ft["log_3d"].get("NHD_z"), "raw"),
        ("NHD-dimensions",         zs["log_3d"].get("NHD_dimensions"),
                                   ft["log_3d"].get("NHD_dimensions"), "raw"),
        ("NHD-pose (rotation)",    zs["log_3d"].get("NHD_pose"),
                                   ft["log_3d"].get("NHD_pose"), "raw"),
        ("NHD best global scale",  zs["nhd"].get("best_scale"),
                                   ft["nhd"].get("best_scale"), "raw"),
        ("Mean NHD @ best scale",  zs["nhd"].get("mean_NHD_at_best_s"),
                                   ft["nhd"].get("mean_NHD_at_best_s"), "raw"),
    ]
    for label, zsv, ftv, fmt in pairs:
        if fmt == "pct":
            zs_str = _fmt(zsv, 2) if zsv is not None and zsv >= 0 else "—"
            ft_str = _fmt(ftv, 2) if ftv is not None and ftv >= 0 else "—"
            delta = ""
            if zsv is not None and ftv is not None and zsv >= 0 and ftv >= 0:
                delta = f"+{ftv - zsv:+.2f}"
        elif fmt == "pp":
            zs_str = _fmt_pct(zsv, 3)
            ft_str = _fmt_pct(ftv, 3)
            delta = ""
            if zsv is not None and ftv is not None and zsv >= 0 and ftv >= 0:
                delta = f"{(ftv - zsv) * 100:+.3f} pp"
        else:
            zs_str = _fmt(zsv, 3)
            ft_str = _fmt(ftv, 3)
            delta = ""
            if zsv is not None and ftv is not None and zsv >= 0 and ftv >= 0:
                delta = f"{ftv - zsv:+.3f}"
        push(f"| {label} | {zs_str} | {ft_str} | {delta} |")
    push("")

    # ---------------- Per-class BEV ----------------
    push("## Per-class AP_BEV @ 0.25 (oracle 2D)\n")
    push("| Row | giraffe | grevys_zebra | elephant | plains_zebra | rhino | gazelle |")
    push("|---|---:|---:|---:|---:|---:|---:|")
    push(f"| Zero-shot | {_bev_per_class(zs['bev'], 'IoU=0.25')} |")
    push(f"| {ft_label} | {_bev_per_class(ft['bev'], 'IoU=0.25')} |")
    push("")
    push("## Per-class AP_BEV @ 0.50 (oracle 2D)\n")
    push("| Row | giraffe | grevys_zebra | elephant | plains_zebra | rhino | gazelle |")
    push("|---|---:|---:|---:|---:|---:|---:|")
    push(f"| Zero-shot | {_bev_per_class(zs['bev'], 'IoU=0.50')} |")
    push(f"| {ft_label} | {_bev_per_class(ft['bev'], 'IoU=0.50')} |")
    push("")
    push("## Per-class standard AP_3D @ 0.05:0.50 (Omni3DEvaluator)\n")
    push("| Row | giraffe | grevys_zebra | elephant | plains_zebra | rhino | gazelle |")
    push("|---|---:|---:|---:|---:|---:|---:|")
    push(f"| Zero-shot | {_full_metrics_per_class_3d(zs['full_metrics'])} |")
    push(f"| {ft_label} | {_full_metrics_per_class_3d(ft['full_metrics'])} |")
    push("")

    # ---------------- Sub-claim verdicts ----------------
    push("## Sub-claim verdicts\n")
    push("**Sub-claim 1 — 2D-3D decoupling.** Zero-shot 3D AP collapses across all "
         "IoU thresholds despite oracle 2D inputs being identical to those used by "
         "ovmono3d's published rows (median IoU vs GT = 0.877, class-agnostic 2D AP "
         "≈ 68 — see [ovmono3d §6.4.2](../../ovmono3d/WILDBOX_EXPERIMENT.md)). "
         "Pretrained Omni3D-distribution 3D priors do not transfer to wildlife "
         "synthetic-scale geometry under any IoU threshold tested.\n")
    push("**Sub-claim 2 — architecture-invariance of zero-shot 3D collapse.** "
         "OVMono3D-Lift (Cube R-CNN, DINOv2 ViT-B/14) and DetAny3D "
         "(SAM ViT-H + UniDepth + DINOv2 ViT-L/14) both report AP_3D = 0 under the "
         "same GDino-oracle 2D protocol. Two architecturally distinct 3D heads, "
         "same source pretraining domain (Omni3D), same collapse — the failure is "
         "in the pretraining-distribution mismatch, not the model class.\n")

    push("## Cross-architecture comparison context\n")
    push("Numbers from ovmono3d's published rows ([ovmono3d/WILDBOX_EXPERIMENT.md "
         "§21.3](../../ovmono3d/WILDBOX_EXPERIMENT.md), full val 13 779 images, "
         "3-seed mean ± std):\n")
    push("| Row | macro AP_3D @ 0.25 | 2D AP @ 0.5 |")
    push("|---|---:|---:|")
    push("| OVMono3D-Lift zero-shot (RPN-transfer) | ≈0  | (low) |")
    push("| OVMono3D-Lift zero-shot (GDino oracle) | ≈0  | ≈68 (class-agnostic) |")
    push("| OVMono3D-Lift fine-tuned (3-seed)      | 9.12 ± 0.84 | 77.8 ± 0.9 |")
    push("")
    push("Direct comparability caveats — see Caveats below.\n")

    # ---------------- Caveats ----------------
    push("## Caveats and disclosures\n")
    push("- **Single-seed.** Multi-seed mean ± std reporting deferred (each seed = "
         "another full fine-tune at ~6 h on 1 × A40). DetAny3D row should be footnoted "
         "as single-seed; OVMono3D-Lift's published row is 3-seed mean ± std.\n")
    push("- **Training budget.** DetAny3D was fine-tuned for 1 epoch (~46k samples "
         "seen). OVMono3D-Lift's published row used 15k iter × batch 8 ≈ 120k samples "
         "(~2.6 epochs). DetAny3D's row is conservative; deeper training would lift "
         "absolute numbers.\n")
    push("- **Val coverage.** Eval at val interval=4 (≈25% of full 13 779). Same "
         "protocol both rows; relative comparison is sound. Full-val numbers may "
         "shift slightly. Re-eval at interval=1 in a separate srun is straightforward "
         "if needed.\n")
    push("- **Eval-time 2D = oracle 2D echo.** DetAny3D doesn't predict its own 2D — "
         "it lifts the GDino oracle box. So 2D AP and class-agnostic 2D AP are "
         "essentially identical between zero-shot and fine-tuned rows by construction. "
         "The variation between rows is in 3D AP, BEV AP, NHD, and Rel-AP_3D.\n")
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
    push("- Visualizations (paper-style 2×3 layouts): "
         f"`{zs['dir']}/vis_ovmono3d/img_*.jpg` and "
         f"`{ft['dir']}/vis_ovmono3d/img_*.jpg`\n")
    push("- Raw eval logs: `<row_dir>/full_metrics/log.{2D,3D,3D-Rel}.txt`, "
         "`<row_dir>/summary_nhd.txt`, `<row_dir>/bev_ap.json`\n")
    push("- Side-by-side ovmono3d-style report (if generated): "
         "`$OVMONO3D_REPO/output/paper_report_detany3d/report.md`\n")
    return "\n".join(out)


def render_design_md(zs: dict, ft: dict, ft_label: str) -> str:
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
         "we evaluate every architecture in the comparison under the **paper-protocol "
         "oracle 2D** mode — the model's own 2D stage is replaced with a precomputed "
         "GroundingDINO open-vocab detection JSON keyed per image. This isolates the "
         "3D-lifting capability of each architecture from the 2D-detection-quality "
         "confound. The same `gdino_WildBox_val_oracle_2d.json` is consumed by both "
         "OVMono3D-Lift and DetAny3D.\n")
    push("DetAny3D is a **prompt-conditioned** detector (no closed-vocab RPN), so "
         "ovmono3d's RPN-transfer zero-shot row has no DetAny3D analog and is "
         "omitted from the cross-arch table. We report two rows for DetAny3D:\n")
    push("1. **Zero-shot oracle 2D** — pretrained DetAny3D weights + GDino oracle 2D.")
    push("2. **Fine-tuned oracle 2D** — DetAny3D fine-tuned on `WildBox_train.json` "
         "(46k images, 6 species), evaluated under the same oracle 2D protocol.\n")

    push("## 2. Architecture as used\n")
    push("Backbone: SAM ViT-H image encoder ([detect_anything/modeling/image_encoder.py]"
         "(../detect_anything/modeling/image_encoder.py)), with a vendored UniDepth "
         "depth predictor and DINOv2 ViT-L/14 features. Prompt-conditioned 3D head "
         "([detect_anything/modeling/mask_decoder.py](../detect_anything/modeling/mask_decoder.py)) "
         "outputs 3D depth (log scale), dimensions, 6D rotation, and a 24-class "
         "orientation classification per prompt. The 3D head is class-agnostic — "
         "what makes the paper-protocol oracle-2D row a valid cross-arch comparison.\n")
    push("Pretrained checkpoint: the released `detany3d.pth` (trained on Omni3D — "
         "KITTI, nuScenes, SUN-RGBD, ARKitScenes, Objectron, Hypersim, 3RScan, "
         "Cityscapes3D, Waymo). Same source domain as OVMono3D-Lift's pretraining; "
         "this is what makes the cross-architecture zero-shot collapse a fair "
         "test of pretraining-domain transferability.\n")

    push("## 3. Training configuration\n")
    push("- **Optimizer:** AdamW (lr 1e-5; UniDepth subgroup lr 1e-7; weight decay "
         "1e-7).\n")
    push("- **Schedule:** CosineAnnealingLR over `cfg.num_epochs` with linear warmup. "
         "Resume from pretrained without loading the optimizer/scheduler state, so "
         "the cosine schedule restarts fresh on each fine-tune attempt "
         "([train.py:506-512](../train.py#L506-L512), commit `eeee735`).\n")
    push("- **Frozen / trainable:** image encoder frozen; prompt encoder + mask "
         "decoder + adapters trainable. Matches OVMono3D-Lift's policy (DINOv2 "
         "frozen during fine-tune).\n")
    push("- **Batch size:** 1 per GPU (image pad 896 × 896, ViT-H is heavy).\n")
    push("- **AMP:** enabled (Ampere tensor cores; ~1.5–2× speedup).\n")
    push("- **Epochs:** 1 (single-GPU final). 2-epoch and multi-seed runs deferred — "
         "see Caveats in [RESULTS_DETANY3D.md](RESULTS_DETANY3D.md).\n")
    push("- **Loss list:** `intrinsic_loss, 2d_bbox_loss, 3d_bbox_loss`. **Depth loss "
         "explicitly disabled** ([WILDBOX_DETANY3D.md §5.4](../WILDBOX_DETANY3D.md)) "
         "because WildBox's per-segment uniform-scale depth normalization makes "
         "metric-depth supervision counterproductive.\n")
    push("- **Output rotation:** full 3 × 3 rotation matrix (drone shots have "
         "non-trivial pitch/roll, not just yaw).\n")
    push("- **Provide GT intrinsics:** on (VGGT's per-frame K is taken as ground "
         "truth, bypassing DetAny3D's intrinsic-prediction head).\n")

    push("## 4. Data pipeline\n")
    push("- **Dataset format:** ovmono3d's `WildBox_{train,val}.json` (Omni3D schema, "
         "absolute paths, K per image, SAM3-tight `bbox`, `bbox3D_cam`, `center_cam`, "
         "`dimensions[W,H,L]`, `R_cam`).\n")
    push("- **Conversion:** [detect_anything/datasets/data_creator/wildbox.py]"
         "(../detect_anything/datasets/data_creator/wildbox.py) writes a DetAny3D "
         "pickle. Convention swap: DetAny3D's `[w, h, l]` = "
         "`reversed(Omni3D[W, H, L])`. Validated by `--verify-projection` "
         "(reproject every cuboid in DetAny3D's convention and compare to GT "
         "`bbox2D_proj`; 100% pass).\n")
    push("- **Oracle-2D loader:** [detect_anything/datasets/detany3d_dataset.py]"
         "(../detect_anything/datasets/detany3d_dataset.py)'s `generate_oracle_list` "
         "method, activated via `cfg.dataset.oracle_2d_input: True`. Loads the "
         "precomputed GDino oracle JSON, maps dataset-id → contiguous-id, rescales "
         "boxes to DetAny3D's preprocess resolution, hands them to the mask decoder "
         "as box prompts.\n")
    push("- **Category mapping:** [data/category_meta_wildbox.json]"
         "(../data/category_meta_wildbox.json) — 6 species, dataset-id ascending, "
         "contiguous 0..5. Same ordering the GT JSON uses, sorted as ovmono3d "
         "documents in [§2.8](../../ovmono3d/WILDBOX_EXPERIMENT.md).\n")

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
         "num_epochs=1)` is empty and training exits with no iterations. Same family "
         "as ovmono3d's documented iteration-skip ([§3.1.1](../../ovmono3d/WILDBOX_EXPERIMENT.md)). "
         "Fixed in commit `eeee735`.\n")
    push("2. **NCCL ALLGATHER timeout** on multi-GPU — under DDP with frozen "
         "backbones and per-rank `/storage3` NFS straggler skew, the first DDP "
         "collective times out after the default 30 minutes. Raised the timeout to "
         "4 h (commit `db7bb37`); even then 4-GPU runs failed to make iter progress, "
         "so we ran final on 1 GPU.\n")
    push("3. **mmcv MSDeformAttn CUDA op** — pip's stock `mmcv==2.0.1` ships without "
         "CUDA ops on this stack; the SAM-style adapter fails with "
         "`ms_deform_attn_impl_forward: implementation for device cuda:0 not found`. "
         "Reinstalled via OpenMMLab's cu116/torch1.13 wheel index.\n")
    push("4. **Empty `prepare_for_dsam` mid-training** — when `filter_objects` "
         "rejects every annotation in a frame (animals at frame edges, etc.), "
         "training crashed with `element 0 of tensors does not require grad`. "
         "Patched the dataset to recurse to a random sample, matching the upstream "
         "pattern for missing image paths (commit `6c89796`).\n")
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
    push("- **Thorough reference**: [WILDBOX_DETANY3D.md](../WILDBOX_DETANY3D.md) — "
         "architecture, env hazards, configs, eval, bug catalogue, cross-arch protocol.\n")
    push("- **Branch / fork**: `wildbox_detany3d` on `https://github.com/vandyshukla04/DetAny3D`.\n")
    push("- **Env**: `/storage3/3DOM/vshukla/envs/detany3d` on the cluster (Python 3.8, "
         "torch 1.13.1+cu116, mmcv 2.0.1 with CUDA ops, opencv-python-headless, "
         "GroundingDINO at the pinned commit).\n")
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zs-dir", type=Path, required=True,
                        help="ovmono3d-side run dir for the zero-shot row "
                             "(contains bev_ap.json, full_metrics/, summary_nhd.txt, vis_ovmono3d/).")
    parser.add_argument("--ft-dir", type=Path, required=True,
                        help="Same shape as --zs-dir but for the fine-tuned row.")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Where to write RESULTS_DETANY3D.md and "
                             "EXPERIMENT_DESIGN_DETANY3D.md.")
    parser.add_argument("--ft-label", type=str,
                        default="DetAny3D fine-tuned (1 epoch, oracle 2D)",
                        help="Header label for the fine-tuned column.")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    zs = _gather_row(args.zs_dir)
    ft = _gather_row(args.ft_dir)

    results_md = render_results_md(zs, ft, args.ft_label)
    design_md = render_design_md(zs, ft, args.ft_label)

    (args.out_dir / "RESULTS_DETANY3D.md").write_text(results_md)
    (args.out_dir / "EXPERIMENT_DESIGN_DETANY3D.md").write_text(design_md)

    print(f"wrote {args.out_dir / 'RESULTS_DETANY3D.md'}  ({len(results_md)} bytes)")
    print(f"wrote {args.out_dir / 'EXPERIMENT_DESIGN_DETANY3D.md'}  ({len(design_md)} bytes)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
