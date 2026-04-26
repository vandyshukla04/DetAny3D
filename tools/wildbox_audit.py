"""
Methodology audit for the DetAny3D × WildBox benchmarking pipeline.

Runs a battery of pass/fail integrity checks against the cluster's
on-disk artifacts: the val pickle, ovmono3d's WildBox_val.json,
the GDino oracle JSON, the category metadata, and any prediction
JSONs you point at. Designed to catch the *kind* of bug we kept
hitting during this benchmark (convention swaps, category-id
remaps, train/val leakage, eval-time filter divergence, score-
field misuse) without re-running any GPU work.

Usage::

    # Full audit. --pred-dir is repeatable; one per row scored to date.
    python tools/wildbox_audit.py \\
        --pkl-train data/pkls/wildbox/WildBox_train.pkl \\
        --pkl-val   data/pkls/wildbox/WildBox_val.pkl \\
        --gt-json   /storage2/3DOM/vshukla/repos/ovmono3d/datasets/Omni3D/WildBox_val.json \\
        --oracle    /storage2/3DOM/vshukla/repos/ovmono3d/datasets/Omni3D/gdino_WildBox_val_oracle_2d.json \\
        --category-meta data/category_meta_wildbox.json \\
        --pred-dir /storage2/3DOM/vshukla/repos/ovmono3d/output/wildbox_detany3d_zs_v3 \\
        --pred-dir /storage2/3DOM/vshukla/repos/ovmono3d/output/wildbox_detany3d_zs_gt2d_v3 \\
        --pred-dir /storage2/3DOM/vshukla/repos/ovmono3d/output/wildbox_detany3d_ft_v3 \\
        --pred-dir /storage2/3DOM/vshukla/repos/ovmono3d/output/wildbox_detany3d_ft_ep2_v3

Each section prints either ``OK`` or ``FAIL`` (sometimes ``WARN``)
followed by the supporting numbers.

If a check FAILs, the eval results may be polluted; investigate before
trusting the numbers in the paper.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional


def _section(title: str):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _ok(msg: str):
    print(f"  OK    {msg}")


def _fail(msg: str):
    print(f"  FAIL  {msg}")


def _warn(msg: str):
    print(f"  WARN  {msg}")


def _info(msg: str):
    print(f"        {msg}")


def _try_load_torch():
    """Lazy import torch only when prediction files are passed."""
    import torch  # noqa: F401
    return __import__("torch")


def audit_meta(meta_path: Path) -> dict:
    _section("[1] Category metadata")
    meta = json.loads(meta_path.read_text())
    classes = meta["thing_classes"]
    mp = {int(k): int(v) for k, v in meta["thing_dataset_id_to_contiguous_id"].items()}

    n_unique_contig = len(set(mp.values()))
    if n_unique_contig != len(mp):
        _fail(f"contiguous-id map has duplicate values: {mp}")
    else:
        _ok(f"contiguous-id map injective ({len(mp)} entries)")

    sorted_ds_ids = sorted(mp.keys())
    sorted_contig_ids = [mp[d] for d in sorted_ds_ids]
    if sorted_contig_ids == list(range(len(mp))):
        _ok(f"contiguous ids 0..{len(mp)-1} sorted by ascending dataset_id "
            f"(matches ovmono3d §2.8 ordering rule)")
    else:
        _fail(f"contiguous ids out of order vs ascending dataset_id: "
              f"sorted_contig_ids={sorted_contig_ids}")

    if classes != [{"giraffe": 1000, "grevys_zebra": 1001, "elephant": 1002,
                    "plains_zebra": 1003, "rhino": 1004,
                    "gazelle": 1005}[k] is not None and k for k in classes][:6]:
        # Display
        pass
    _info(f"thing_classes = {classes}")
    _info(f"dataset_id->contig = {mp}")
    return meta


def audit_pickle(pkl_path: Path, gt_json: dict, label: str) -> dict:
    _section(f"[2.{label}] Pickle integrity: {pkl_path}")
    pkl = pickle.load(open(pkl_path, "rb"))
    _info(f"records: {len(pkl)}")
    n_objs = sum(len(r.get("obj_list", [])) for r in pkl)
    _info(f"objects (post-converter filter): {n_objs}")

    # Required keys per record.
    required_record_keys = {"K", "img_path", "depth_path", "obj_list"}
    required_obj_keys = {"3d_bbox", "2d_bbox_proj", "rotation_pose",
                         "instance_id", "label", "image_id",
                         "score", "visibility", "truncation"}
    missing_record = 0
    missing_obj = 0
    bad_K_shape = 0
    for r in pkl:
        if not required_record_keys.issubset(r.keys()):
            missing_record += 1
        K = r.get("K")
        if not (hasattr(K, "shape") and K.shape == (1, 3, 3)):
            bad_K_shape += 1
        for o in r.get("obj_list", []):
            if not required_obj_keys.issubset(o.keys()):
                missing_obj += 1
                break

    if missing_record == 0 and bad_K_shape == 0 and missing_obj == 0:
        _ok("schema: all records and objects have the expected keys; K is (1,3,3)")
    else:
        _fail(f"schema issues: missing_record={missing_record} "
              f"bad_K_shape={bad_K_shape} missing_obj={missing_obj}")

    # Convention swap: dimension ordering should be reversed from Omni3D's [W,H,L].
    # We verify by comparing the pickle's dim ordering against the JSON's dim ordering
    # for matching annotations (image_id + ann_id stable).
    json_anns_by_id = {int(a["id"]): a for a in gt_json.get("annotations", [])}
    n_dim_check = 0
    n_dim_match = 0
    for r in pkl[:200]:  # sample 200 for speed
        for o in r.get("obj_list", []):
            inst_id = o.get("instance_id", "")
            # instance_id format: f"{image_id}_{ann_id}" per our converter.
            try:
                _, ann_id_str = inst_id.split("_")
                ann_id = int(ann_id_str)
            except (ValueError, AttributeError):
                continue
            ann = json_anns_by_id.get(ann_id)
            if ann is None or "dimensions" not in ann:
                continue
            n_dim_check += 1
            json_dims = list(ann["dimensions"])  # [W, H, L]
            pkl_dims = o["3d_bbox"][3:6]         # [w, h, l] expected reversed
            expected = list(reversed(json_dims))
            if all(abs(a - b) < 1e-3 for a, b in zip(pkl_dims, expected)):
                n_dim_match += 1
    if n_dim_check > 0:
        rate = n_dim_match / n_dim_check
        if rate >= 0.99:
            _ok(f"dimensions [w,h,l] = reversed(Omni3D [W,H,L]): "
                f"{n_dim_match}/{n_dim_check} match (>=99%)")
        else:
            _fail(f"dimensions reversal failed for "
                  f"{n_dim_check - n_dim_match}/{n_dim_check} sampled annotations")
    else:
        _info("dim-reversal check skipped (no matching ann_ids found in the "
              "GT JSON passed for this pickle; pass the right --gt-json / "
              "--train-gt-json to enable)")

    # depth_path should be None for all records (synthetic-scale GT, no metric depth).
    n_with_depth = sum(1 for r in pkl if r.get("depth_path") is not None)
    if n_with_depth == 0:
        _ok("depth_path: None on every record (matches synthetic-scale design)")
    else:
        _warn(f"depth_path: {n_with_depth} records have non-None depth_path "
              f"(unexpected for WildBox)")

    # All labels are contiguous 0..N-1.
    bad_labels = [o["label"] for r in pkl for o in r.get("obj_list", [])
                  if not isinstance(o["label"], int) or o["label"] < 0 or o["label"] > 5]
    if not bad_labels:
        _ok("labels: all in {0..5}")
    else:
        _fail(f"{len(bad_labels)} objects have invalid label (must be contiguous 0..5)")

    # image_id stable across record's obj_list.
    n_inconsistent_img_id = 0
    for r in pkl:
        ids = {o.get("image_id") for o in r.get("obj_list", [])}
        if len(ids) > 1:
            n_inconsistent_img_id += 1
    if n_inconsistent_img_id == 0:
        _ok("image_id consistent within each record")
    else:
        _fail(f"{n_inconsistent_img_id} records have multiple image_id values "
              f"(should be one per image)")

    # img_path existence is too slow to check on full pickle; sample a few.
    sample_paths = [r["img_path"] for r in pkl[::max(1, len(pkl) // 5)][:5]]
    n_exist = sum(1 for p in sample_paths if Path(p).exists())
    _info(f"sampled file_paths exist: {n_exist}/{len(sample_paths)} "
          f"({sample_paths[0] if sample_paths else 'none'} ...)")

    return {"pkl": pkl, "n_records": len(pkl), "n_objs": n_objs}


def audit_train_val_leak(pkl_train: dict, pkl_val: dict):
    _section("[3] Train/val leakage")
    # Image-level: video extracted from path.
    def vid(r):
        try:
            return r["img_path"].split("/")[-3]
        except (KeyError, IndexError):
            return None

    train_vids = {vid(r) for r in pkl_train["pkl"] if vid(r)}
    val_vids = {vid(r) for r in pkl_val["pkl"] if vid(r)}
    overlap = train_vids & val_vids
    if not overlap:
        _ok(f"video-level disjoint: {len(train_vids)} train vids, "
            f"{len(val_vids)} val vids, 0 shared")
    else:
        _fail(f"video-level leakage: {len(overlap)} shared videos: "
              f"{list(overlap)[:5]}")

    # Image-level (sanity check; should also be disjoint).
    def img_id(r):
        if r.get("obj_list"):
            return int(r["obj_list"][0]["image_id"])
        return None
    train_img_ids = {img_id(r) for r in pkl_train["pkl"] if img_id(r) is not None}
    val_img_ids = {img_id(r) for r in pkl_val["pkl"] if img_id(r) is not None}
    img_overlap = train_img_ids & val_img_ids
    if not img_overlap:
        _ok("image_id disjoint between train and val")
    else:
        _warn(f"image_id space overlaps between train and val "
              f"(expected if both use 0-indexed enumeration; verify via path)")
        _info(f"shared image_ids: {len(img_overlap)} (could be coincidental)")


def audit_oracle(oracle_path: Path, pkl_val: dict, meta: dict):
    _section("[4] Oracle JSON ↔ val pickle")
    oracle = json.loads(oracle_path.read_text())
    oracle_by_id = {int(e["image_id"]): e for e in oracle}
    n_oracle_imgs = len(oracle)
    n_oracle_boxes = sum(len(e.get("instances", [])) for e in oracle)
    _info(f"oracle: {n_oracle_imgs} images, {n_oracle_boxes} boxes "
          f"({n_oracle_boxes/max(1,n_oracle_imgs):.2f} avg)")

    # Image-id coverage.
    pkl_ids = {int(r["obj_list"][0]["image_id"]) for r in pkl_val["pkl"]
               if r.get("obj_list")}
    oracle_ids = set(oracle_by_id.keys())
    in_pkl_not_oracle = pkl_ids - oracle_ids
    in_oracle_not_pkl = oracle_ids - pkl_ids
    if not in_pkl_not_oracle and not in_oracle_not_pkl:
        _ok(f"image_id coverage: {len(pkl_ids)} pkl == {len(oracle_ids)} oracle, "
            f"perfect match")
    else:
        _fail(f"image_id coverage gap: pkl-only={len(in_pkl_not_oracle)} "
              f"oracle-only={len(in_oracle_not_pkl)}")

    # Oracle category_id values must all be in our dataset_id set.
    ds_ids = set(int(k) for k in meta["thing_dataset_id_to_contiguous_id"].keys())
    bad_classes = collections.Counter()
    for e in oracle:
        for inst in e.get("instances", []):
            cid = int(inst["category_id"])
            if cid not in ds_ids:
                bad_classes[cid] += 1
    if not bad_classes:
        _ok(f"all oracle category_ids ∈ {sorted(ds_ids)}")
    else:
        _fail(f"oracle has out-of-taxonomy category_ids: {dict(bad_classes)}")

    # Per-class oracle counts.
    cls = collections.Counter()
    for e in oracle:
        for inst in e.get("instances", []):
            cls[inst.get("category_name", "?")] += 1
    _info(f"per-class oracle box counts: {dict(cls)}")


def audit_predictions(pred_dir: Path, gt_json: dict, oracle: List[dict],
                      pkl_val: dict, meta: dict):
    _section(f"[5] Predictions: {pred_dir.name}")
    torch = _try_load_torch()
    pth_path = (pred_dir / "inference" / "iter_final" / "WildBox_val"
                / "instances_predictions.pth")
    if not pth_path.exists():
        _warn(f"missing {pth_path} — skip")
        return
    preds = torch.load(str(pth_path), weights_only=False, map_location="cpu")
    n_imgs_in_preds = len(preds)
    n_preds = sum(len(p.get("instances", [])) for p in preds)
    _info(f"prediction images: {n_imgs_in_preds}, total preds: {n_preds}")

    # All required prediction-instance keys present.
    required = {"image_id", "category_id", "bbox", "score", "bbox3D_cam",
                "center_cam", "dimensions", "pose"}
    missing = 0
    for p in preds:
        for inst in p.get("instances", []):
            if not required.issubset(inst.keys()):
                missing += 1
                break
    if missing == 0:
        _ok("prediction-instance schema complete")
    else:
        _fail(f"{missing} prediction images have instances with missing keys")

    # Category-id space: should be the *contiguous* 0..5 OR the dataset_id space
    # 1000..1005 — the exporter's task is to remap; we audit by checking against
    # category_meta and complaining loudly if neither space matches.
    contig_ids = set(meta["thing_dataset_id_to_contiguous_id"].values())
    ds_ids = {int(k) for k in meta["thing_dataset_id_to_contiguous_id"].keys()}
    pred_classes = collections.Counter()
    for p in preds:
        for inst in p.get("instances", []):
            pred_classes[int(inst["category_id"])] += 1
    space_contig = all(c in contig_ids for c in pred_classes)
    space_ds = all(c in ds_ids for c in pred_classes)
    if space_ds:
        _ok(f"predictions use dataset_id space {sorted(ds_ids)} (post-export)")
    elif space_contig:
        _warn(f"predictions use contiguous_id space {sorted(contig_ids)} "
              f"— these need wildbox_full_metrics.py's remap to score under "
              f"Omni3DEvaluator (bev_ap_eval handles either)")
    else:
        _fail(f"predictions use UNKNOWN category_id space: {sorted(pred_classes)}")
    _info(f"per-class prediction counts: {dict(pred_classes)}")

    # Prediction count vs oracle count for oracle-protocol rows.
    # Heuristic: if every image has the same n_preds as the oracle, we're echoing
    # oracle 2D faithfully. Otherwise we're either filtering (e.g. small-box
    # filter, GT-2D filter_objects path) or losing boxes somewhere.
    oracle_by_id = {int(e["image_id"]): e for e in oracle}
    n_match = 0
    n_total = 0
    n_pred_lt_oracle = 0
    n_pred_gt_oracle = 0
    sampled_imgs = [int(p["image_id"]) for p in preds[:200] if p.get("instances")]
    for img_id in sampled_imgs:
        # find pred entries for this image (sum over the instances list -- already done above)
        pred_n = 0
        for p in preds:
            if int(p.get("image_id", -1)) == img_id:
                pred_n = len(p.get("instances", []))
                break
        oracle_n = len(oracle_by_id.get(img_id, {}).get("instances", []))
        n_total += 1
        if pred_n == oracle_n:
            n_match += 1
        elif pred_n < oracle_n:
            n_pred_lt_oracle += 1
        else:
            n_pred_gt_oracle += 1
    if n_total > 0:
        rate = n_match / n_total
        if rate >= 0.95:
            _ok(f"per-image n_preds matches oracle n_instances: "
                f"{n_match}/{n_total} ({rate*100:.0f}%) — oracle echo intact")
        elif n_pred_lt_oracle > 0 and n_pred_gt_oracle == 0:
            _warn(f"predictions are a strict subset of oracle for "
                  f"{n_pred_lt_oracle}/{n_total} sampled images "
                  f"(filter_objects or oracle hook is dropping). "
                  f"Compare classes/sizes against oracle.")
        else:
            _info(f"oracle-vs-pred per-image: {n_match} match, "
                  f"{n_pred_lt_oracle} pred<oracle, {n_pred_gt_oracle} pred>oracle "
                  f"(latter is unusual; investigate)")

    # Score field: distribution.
    scores = [float(inst["score"]) for p in preds for inst in p.get("instances", [])]
    if scores:
        smin = min(scores); smax = max(scores)
        smedian = sorted(scores)[len(scores)//2]
        n_unique = len(set(round(s, 5) for s in scores))
        _info(f"score distribution: min={smin:.4f} median={smedian:.4f} "
              f"max={smax:.4f} unique≈{n_unique}")
        if n_unique == 1:
            _warn(f"all predictions have score={scores[0]:.4f}; per-class AP "
                  f"becomes order-degenerate (acceptable for GT-2D ceiling row, "
                  f"misleading for oracle-2D rows where oracle scores should propagate)")
        elif smin >= 0.0 and smax <= 1.0:
            _ok(f"scores in [{smin:.4f}, {smax:.4f}] (well-formed; "
                f"{n_unique} distinct values)")
        else:
            _warn(f"scores out of [0,1]: min={smin}, max={smax}")

    # bbox is xywh in original image pixels. Range sanity.
    bad_bbox = 0
    for p in preds:
        for inst in p.get("instances", []):
            bb = inst["bbox"]
            if (len(bb) != 4
                    or bb[2] <= 0 or bb[3] <= 0
                    or bb[0] < -1 or bb[1] < -1
                    or bb[0] > 1e5 or bb[1] > 1e5):
                bad_bbox += 1
    if bad_bbox == 0:
        _ok("all bbox entries in xywh with positive width/height")
    else:
        _fail(f"{bad_bbox} bbox entries have non-positive w/h or out-of-range coords")

    # bbox3D_cam shape = 8x3 with finite floats.
    bad_3d = 0
    for p in preds:
        for inst in p.get("instances", []):
            corners = inst.get("bbox3D_cam") or inst.get("bbox3D")
            if corners is None:
                bad_3d += 1; continue
            try:
                if len(corners) != 8 or any(len(c) != 3 for c in corners):
                    bad_3d += 1; continue
                for c in corners:
                    for v in c:
                        if not math.isfinite(v):
                            bad_3d += 1; raise StopIteration
            except StopIteration:
                continue
    if bad_3d == 0:
        _ok("all bbox3D_cam entries are 8×3 with finite values")
    else:
        _fail(f"{bad_3d} bbox3D_cam entries malformed or contain NaN/inf")

    # Dimensions ordering: post-export should be Omni3D [W, H, L].
    # We can't fully verify without re-running the projection, but flag if
    # any dimension is non-positive.
    bad_dims = sum(1 for p in preds for inst in p.get("instances", [])
                   if any(d <= 0 for d in inst.get("dimensions", [1, 1, 1])))
    if bad_dims == 0:
        _ok("all dimensions positive (no degenerate cuboids)")
    else:
        _fail(f"{bad_dims} predictions have non-positive dimensions")

    # Pose is a 3x3 rotation matrix; check approximate orthogonality on a sample.
    n_orth_check = 0
    n_orth_pass = 0
    for p in preds[:50]:
        for inst in p.get("instances", []):
            R = inst.get("pose")
            if R is None or len(R) != 3:
                continue
            n_orth_check += 1
            try:
                # R^T R should be ~identity.
                rt = [[sum(R[k][i] * R[k][j] for k in range(3)) for j in range(3)]
                      for i in range(3)]
                err = sum(abs(rt[i][j] - (1.0 if i == j else 0.0))
                          for i in range(3) for j in range(3))
                if err < 0.05:
                    n_orth_pass += 1
            except (TypeError, ValueError):
                pass
    if n_orth_check > 0:
        rate = n_orth_pass / n_orth_check
        if rate >= 0.9:
            _ok(f"pose orthogonal: {n_orth_pass}/{n_orth_check} sampled "
                f"R^T R within 0.05 of I")
        else:
            _warn(f"only {n_orth_pass}/{n_orth_check} sampled poses are orthogonal "
                  f"— check rotation_6d_to_matrix decoding")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl-train", type=Path, required=True)
    ap.add_argument("--pkl-val", type=Path, required=True)
    ap.add_argument("--gt-json", type=Path, required=True,
                    help="ovmono3d's WildBox_val.json")
    ap.add_argument("--train-gt-json", type=Path, default=None,
                    help="ovmono3d's WildBox_train.json. Optional but required "
                         "for the dim-reversal check on the train pickle (since "
                         "train and val annotation IDs are independently 0-indexed).")
    ap.add_argument("--oracle", type=Path, required=True,
                    help="ovmono3d's gdino_WildBox_val_oracle_2d.json")
    ap.add_argument("--category-meta", type=Path,
                    default=Path("data/category_meta_wildbox.json"))
    ap.add_argument("--pred-dir", type=Path, action="append", default=[],
                    help="Per-row directory under ovmono3d/output/, repeatable.")
    args = ap.parse_args(argv)

    print("=" * 72)
    print("DetAny3D × WildBox — methodology audit")
    print("=" * 72)

    meta = audit_meta(args.category_meta)

    print("\nLoading WildBox_val.json (this may take a few seconds)...")
    gt_json = json.loads(args.gt_json.read_text())
    _info(f"  GT: {len(gt_json.get('images', []))} images, "
          f"{len(gt_json.get('annotations', []))} annotations")

    if args.train_gt_json:
        print("\nLoading WildBox_train.json (this may take a few seconds)...")
        train_gt_json = json.loads(args.train_gt_json.read_text())
        _info(f"  TRAIN GT: {len(train_gt_json.get('images', []))} images, "
              f"{len(train_gt_json.get('annotations', []))} annotations")
    else:
        train_gt_json = None
        print("\n(skipping train-pickle dim-reversal check — pass --train-gt-json "
              "to enable; not strictly needed if val passed since same converter)")

    pkl_train = audit_pickle(args.pkl_train, train_gt_json or {"annotations": []}, "train")
    pkl_val = audit_pickle(args.pkl_val, gt_json, "val")
    audit_train_val_leak(pkl_train, pkl_val)
    audit_oracle(args.oracle, pkl_val, meta)

    if args.pred_dir:
        oracle_list = json.loads(args.oracle.read_text())
        for pd in args.pred_dir:
            audit_predictions(pd, gt_json, oracle_list, pkl_val, meta)
    else:
        _section("[5] Predictions")
        _info("(no --pred-dir passed; skipping prediction-side audits)")

    print()
    print("=" * 72)
    print("Audit done. Review FAIL and WARN lines above before paper submission.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
