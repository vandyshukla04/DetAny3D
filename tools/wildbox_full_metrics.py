"""
Run ovmono3d's standard Omni3D evaluator on DetAny3D-exported predictions.

Adds the metrics that ``bev_ap_eval.py`` and ``class_agnostic_eval.py``
don't cover, so DetAny3D can fill every column of ovmono3d's main paper
table:

    - AP_3D @ 0.05:0.50 (standard 3D AP via pytorch3d.box3d_overlap, CPU)
    - AP_3D @ 0.25 / 0.50 / 0.75
    - 2D AP @ 0.5:0.95 (COCO via pycocotools)
    - per-class breakdowns
    - disentangled NHD per (xy, z, dimensions, pose)
    - Rel-AP_3D (LabelAny3D global-scalar grid search; opt-in via --eval-rel-ap3d)

Imports from ovmono3d's cubercnn package -- this script must run in
ovmono3d's conda env (which has pytorch3d-CPU built; see ovmono3d's
WILDBOX_EXPERIMENT.md §3.1.3).

Inputs:
  --predictions   instances_predictions.pth (detectron2-format) produced
                  by tools/wildbox_export_predictions.py
  --gt            WildBox_val.json (Omni3D schema)
  --ovmono3d-repo path to the ovmono3d clone (for sys.path injection)
  --out-dir       where to write log.{2D,3D,3D-Rel}.txt + summary.json

Output:
  summary.json with the same column structure as ovmono3d's
  paper_report/metrics.json, so make_report.py --compare can ingest it.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("wildbox_full_metrics")


def _normalise_instance(inst: dict) -> dict:
    """Coerce our exported instance dict to the keys ovmono3d's evaluator
    expects (image_id, category_id, bbox xywh, score, bbox3D 8x3,
    center_cam, center_2D, dimensions, pose). Handles 'bbox3D_cam' alias."""
    out = {
        "image_id": int(inst["image_id"]),
        "category_id": int(inst["category_id"]),
        "bbox": [float(v) for v in inst["bbox"]],
        "score": float(inst.get("score", inst.get("score_3d", 0.0))),
    }
    bbox3D = inst.get("bbox3D") if "bbox3D" in inst else inst.get("bbox3D_cam")
    if bbox3D is None:
        raise KeyError(f"prediction missing bbox3D / bbox3D_cam: keys={list(inst.keys())}")
    out["bbox3D"] = bbox3D
    out["center_cam"] = inst["center_cam"]
    # center_2D may be missing on some exports -- synthesise from bbox center.
    if "center_2D" in inst:
        out["center_2D"] = inst["center_2D"]
    else:
        x, y, w, h = inst["bbox"]
        out["center_2D"] = [x + w / 2.0, y + h / 2.0]
    out["dimensions"] = inst["dimensions"]
    out["pose"] = inst["pose"]
    if "depth" in inst:
        out["depth"] = float(inst["depth"])
    else:
        # mean Z over the 8 corners
        zs = [c[2] for c in bbox3D]
        out["depth"] = float(sum(zs) / len(zs))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="instances_predictions.pth path")
    parser.add_argument("--gt", type=Path, required=True,
                        help="WildBox_val.json (Omni3D schema)")
    parser.add_argument("--ovmono3d-repo", type=Path, required=True,
                        help="path to ovmono3d clone (for cubercnn imports)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--eval-rel-ap3d", action="store_true",
                        help="also run LabelAny3D Rel-AP_3D scale search "
                             "(slow on CPU; ~10-30 min on full val).")
    parser.add_argument("--rel-ap3d-search", type=str, default="0.05,3.0,32",
                        help="lo,hi,n for global-scalar grid (zero-shot needs "
                             "the wider 0.05 lower bound).")
    parser.add_argument("--score-min", type=float, default=0.0,
                        help="drop preds with score below this.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Inject ovmono3d into the import path.
    sys.path.insert(0, str(args.ovmono3d_repo.resolve()))
    try:
        from cubercnn.data.datasets import Omni3D
        from cubercnn.evaluation.omni3d_evaluation import _evaluate_predictions_on_omni
    except ImportError as e:
        print(f"ERROR: failed to import from ovmono3d at {args.ovmono3d_repo}", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        print("Run this script from ovmono3d's conda env, with --ovmono3d-repo set "
              "to the repo root.", file=sys.stderr)
        return 2

    logger.info(f"loading GT {args.gt}")
    omni_gt = Omni3D([str(args.gt)])

    # ovmono3d's prepare_wildbox_dataset.py emits 'bbox3D_cam' but the
    # evaluator reads 'bbox3D'. Mirror the key on every loaded annotation
    # so Omni3DevalWithNHD.computeIoU at line 1675 works.
    n_aliased = 0
    for ann in omni_gt.anns.values():
        if "bbox3D" not in ann and "bbox3D_cam" in ann:
            ann["bbox3D"] = ann["bbox3D_cam"]
            n_aliased += 1
    if n_aliased:
        logger.info(f"aliased bbox3D <- bbox3D_cam on {n_aliased} GT annotations")

    logger.info(f"loading predictions {args.predictions}")
    import torch
    preds_per_image = torch.load(str(args.predictions), weights_only=False, map_location="cpu")

    omni_results = []
    n_skipped_score = 0
    for entry in preds_per_image:
        for inst in entry.get("instances", []):
            if inst.get("score", inst.get("score_3d", 1.0)) < args.score_min:
                n_skipped_score += 1
                continue
            omni_results.append(_normalise_instance(inst))
    logger.info(f"loaded {len(omni_results)} predictions across "
                f"{len(preds_per_image)} images (skipped {n_skipped_score} "
                f"by score threshold)")

    if not omni_results:
        print("ERROR: no predictions remain after score threshold; aborting", file=sys.stderr)
        return 3

    lo, hi, n = (float(x) for x in args.rel_ap3d_search.split(","))
    evals, log_strs = _evaluate_predictions_on_omni(
        omni_gt=omni_gt,
        omni_results=omni_results,
        iou_type="bbox",
        eval_rel_ap3d=args.eval_rel_ap3d,
        rel_ap3d_search=(lo, hi, int(n)),
    )

    # Save the human-readable log per mode.
    for mode, log_str in log_strs.items():
        out_log = args.out_dir / f"log.{mode}.txt"
        out_log.write_text(log_str)
        logger.info(f"wrote {out_log}")

    # Pull the headline numbers into a machine-readable summary.
    summary = {}
    for mode, omni_eval in evals.items():
        stats = getattr(omni_eval, "stats", None)
        if stats is None:
            continue
        # COCOeval-style stats vector: [AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl]
        summary[mode] = {
            "AP":   float(stats[0]) if len(stats) > 0 else None,
            "AP50": float(stats[1]) if len(stats) > 1 else None,
            "AP75": float(stats[2]) if len(stats) > 2 else None,
        }
        # Per-class precision (if exposed by Omni3DevalWithNHD).
        # The structure is precision[T x R x K x A x M]; T iouThrs, K classes.
        try:
            import numpy as np
            precision = omni_eval.eval.get("precision", None)
            if precision is not None and isinstance(precision, np.ndarray):
                # precision shape: [T, R, K, A, M]; AP per class = mean over T,R at A=0,M=-1
                per_class_ap = precision[:, :, :, 0, -1]  # all T, all R, K classes
                # Mask -1 values (no GT)
                per_class_ap_masked = np.where(per_class_ap == -1, np.nan, per_class_ap)
                cat_ids = omni_gt.getCatIds()
                cat_names = [c["name"] for c in omni_gt.loadCats(cat_ids)]
                summary[mode]["per_class_AP"] = {
                    name: (float(np.nanmean(per_class_ap_masked[:, :, i]))
                           if not np.all(np.isnan(per_class_ap_masked[:, :, i])) else None)
                    for i, name in enumerate(cat_names)
                }
        except Exception as e:
            logger.warning(f"per-class AP extraction failed for mode {mode}: {e}")

    summary["meta"] = {
        "predictions": str(args.predictions),
        "gt": str(args.gt),
        "score_min": args.score_min,
        "n_predictions": len(omni_results),
        "n_images_in_gt": len(omni_gt.getImgIds()),
        "rel_ap3d_search": [lo, hi, int(n)] if args.eval_rel_ap3d else None,
    }

    out_summary = args.out_dir / "summary.json"
    out_summary.write_text(json.dumps(summary, indent=2))
    logger.info(f"wrote {out_summary}")

    # Print the key numbers.
    print("\n=== summary ===")
    for mode in ["2D", "3D", "3D-Rel"]:
        if mode in summary:
            s = summary[mode]
            print(f"  {mode}: AP={s.get('AP', None)}  AP50={s.get('AP50', None)}  AP75={s.get('AP75', None)}")
            if "per_class_AP" in s:
                pc = s["per_class_AP"]
                cells = "  ".join(f"{k}={v:.3f}" if v is not None else f"{k}=-"
                                  for k, v in pc.items())
                print(f"    per-class: {cells}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
