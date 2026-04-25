# DetAny3D × WildBox — replication guide

This branch (`wildbox_detany3d`) ports ovmono3d's WildBox 3D wildlife
detection experiments onto the DetAny3D architecture. Goal: a fair
zero-shot vs fine-tuned cross-architecture benchmark for the WildBox
dataset paper.

The companion documentation in ovmono3d ([WILDBOX_EXPERIMENT.md][1],
[FINAL_RUN.md][2], [QUICK_START_SMOKE_TEST.md][3]) is the source of truth
for everything dataset- and protocol-related. **Do not duplicate that
content here** — only the DetAny3D-specific deltas live in this file.

[1]: ../ovmono3d/WILDBOX_EXPERIMENT.md
[2]: ../ovmono3d/FINAL_RUN.md
[3]: ../ovmono3d/QUICK_START_SMOKE_TEST.md

---

## 1. What we report (protocol per [§20.5][1])

Two rows in the main paper table for DetAny3D:

| Row | What | Where the numbers come from |
|---|---|---|
| **Zero-shot, paper protocol (oracle 2D)** | Pretrained DetAny3D + ovmono3d's `gdino_WildBox_val_oracle_2d.json` for 2D prompts. The 3D head lifts those boxes to 3D. | `output/wildbox_detany3d_zeroshot_oracle/` (in ovmono3d repo) |
| **Fine-tuned (single seed)** | DetAny3D fine-tuned on `WildBox_train.json`, evaluated against the same val + same oracle 2D. | `output/wildbox_detany3d_finetuned_oracle/` |

The third row in ovmono3d's protocol — **zero-shot RPN-transfer** — has
no analog in DetAny3D, which is prompt-conditioned (no closed-vocab RPN
to transfer). Footnote in the paper. Cross-architecture comparison stays
apples-to-apples on rows 2 + 3.

Multi-seed mean±std is **deferred** in this branch — a single deterministic
fine-tune for now. Add seeds 1, 2 later by varying the random seed in
`wildbox_final.yaml` and rerunning step [4/6] of `tools/wildbox_final.sh`.

---

## 2. What's added to DetAny3D on this branch

| File | Purpose |
|---|---|
| `data/category_meta_wildbox.json` | 6-species mapping (dataset-id 1000–1005 → contiguous 0–5), sorted ascending per [§2.8][1]. |
| `detect_anything/datasets/data_creator/wildbox.py` | Converter: ovmono3d's Omni3D-schema `WildBox_*.json` → DetAny3D pickle. |
| `detect_anything/datasets/detany3d_dataset.py` | Added `generate_oracle_list` method + `cfg.dataset.oracle_2d_input` hook. |
| `detect_anything/configs/wildbox/wildbox_smoke.yaml` | 3-epoch smoke training config. |
| `detect_anything/configs/wildbox/wildbox_final.yaml` | Final fine-tune config (single seed). |
| `detect_anything/configs/wildbox/wildbox_eval_oracle2d.yaml` | Eval-only config for paper-protocol zero-shot AND fine-tuned eval (same oracle JSON for both). |
| `tools/wildbox_export_predictions.py` | DetAny3D eval JSON → `instances_predictions.pth` for ovmono3d's eval stack. |
| `tools/wildbox_smoke.sh` | End-to-end smoke pipeline (data convert → zero-shot eval → fine-tune → fine-tuned eval → export → ovmono3d eval). |
| `tools/wildbox_final.sh` | Same shape, full data, single seed. |

**No model changes.** The 3D heads, mask decoder, image encoder are all
unchanged. Only data loading, configs, eval glue.

---

## 3. Convention swaps (worth knowing before debugging numbers)

### 3.1 Cuboid dimensions

ovmono3d / Omni3D stores `dimensions = [W, H, L]` with the axis assignment
`X = L, Y = H, Z = W` (see [§2.4][1]).

DetAny3D's `compute_3d_bbox_vertices` in
`detect_anything/datasets/utils.py` takes `[w, h, l]` where `w` is the
X-extent, `h` is the Y-extent, `l` is the Z-extent.

So the converter writes `DetAny3D[w, h, l] = reversed(Omni3D.dimensions)`.
The exporter (`tools/wildbox_export_predictions.py`) reverses again on
the way out so ovmono3d's evaluators see Omni3D ordering.

If post-training BEV / 3D AP looks anomalous, run the converter with
`--verify-projection` — it projects each cuboid back to 2D using
DetAny3D's convention and compares against the GT `bbox2D_proj` already
in the WildBox JSON. >5% failures means the dimension or rotation
convention swap is wrong.

### 3.2 Rotation matrix

`R_cam` is passed through unchanged. Both Omni3D and DetAny3D treat it
as "rotation from local cuboid frame to camera frame." Local-frame axis
ordering should match the dimension ordering above; if it doesn't, you'd
see the projection-verify check fail systematically.

### 3.3 Yaw

`output_rotation_matrix: True` is required for WildBox — drone shots
have non-trivial pitch/roll, not just yaw. The yaw value in the pickle's
`3d_bbox` is a placeholder (`0.0`); DetAny3D reads `rotation_pose`
directly when `output_rotation_matrix` is set.

### 3.4 Depth

WildBox depth is **synthetic per-segment scale** (median |Z|=1, not
metric). To handle this honestly:

1. `tune_with_depth: False` in the training configs.
2. `depth_loss` removed from `loss.loss_list`.
3. `depth_path: None` in every pickle entry → loaded as zeros, depth
   mask becomes zero, no depth supervision either way.

Depth predictions still come out at inference (the head still runs, it
just isn't supervised). They're passed through to `instances_predictions.pth`
unchanged; ovmono3d's BEV AP ignores Z, and Rel-AP3D handles the global
scale offset.

This mirrors ovmono3d's [§5.4][1] rationale (downweight depth for
synthetic-scale GT) but goes further: full disable rather than 0.5×.
Cross-architecture comparison is apples-to-apples because both
architectures effectively treat WildBox as 2D + dimensions + pose,
ignoring the Z-axis prior.

---

## 4. Smoke pipeline (~ run on cluster after clone)

Before running, ensure these exist on the cluster:

- ovmono3d cloned alongside DetAny3D (any path)
- ovmono3d's smoke pipeline already run, producing `output/smoke/WildBox_{train,val}.json`
- ovmono3d's GDino oracle precomputed as `datasets/Omni3D/gdino_WildBox_val_oracle_2d.json`
- DetAny3D's checkpoints under `checkpoints/{sam_ckpts,unidepth_ckpts,dino_ckpts,detany3d_ckpts}/`

Then:

```bash
cd <DetAny3D clone>
git checkout wildbox_detany3d
git pull

export OVMONO3D_REPO=/path/to/ovmono3d
export WILDBOX_TRAIN_JSON=$OVMONO3D_REPO/output/smoke/WildBox_train.json
export WILDBOX_VAL_JSON=$OVMONO3D_REPO/output/smoke/WildBox_val.json
export GDINO_ORACLE_JSON=$OVMONO3D_REPO/datasets/Omni3D/gdino_WildBox_val_oracle_2d.json
export NUM_GPUS=1   # bump to 8 for DDP

bash tools/wildbox_smoke.sh
```

Six steps, ~1–3 hours wall-clock depending on hardware:

1. Convert WildBox JSONs → DetAny3D pickles, with projection verify.
2. Zero-shot oracle-2D eval (no training; pretrained DetAny3D + GDino prompts).
3. Smoke fine-tune (3 epochs).
4. Fine-tuned oracle-2D eval (same GDino prompts).
5. Export both JSONs to ovmono3d `instances_predictions.pth`.
6. Run ovmono3d's `bev_ap_eval.py` + `class_agnostic_eval.py` on each.

Smoke passes when both `output/smoke_detany3d_*/bev_ap.json` files exist
with non-degenerate per-class entries for all 6 species.

Match against ovmono3d's [QUICK_START_SMOKE_TEST.md][3] §12 inventory —
artifacts equivalent for DetAny3D should all exist.

---

## 5. Final pipeline

Same shape, full 15-zip 6-species data, single seed:

```bash
export WILDBOX_TRAIN_JSON=$OVMONO3D_REPO/datasets/Omni3D/WildBox_train.json
export WILDBOX_VAL_JSON=$OVMONO3D_REPO/datasets/Omni3D/WildBox_val.json
# GDINO_ORACLE_JSON unchanged
export NUM_GPUS=8

bash tools/wildbox_final.sh
```

Outputs land in:

- `$OVMONO3D_REPO/output/wildbox_detany3d_zeroshot_oracle/`
- `$OVMONO3D_REPO/output/wildbox_detany3d_finetuned_oracle/`
- `$OVMONO3D_REPO/output/paper_report_detany3d/report.md` (side-by-side)

---

## 6. Hazards from ovmono3d's [§3.1, §21.2][1] worth re-checking on this side

| ovmono3d hazard | DetAny3D analog |
|---|---|
| **Silent-skip-training** (pretrained iter field) | DetAny3D loads via `cfg.resume` (sets `start_epoch = checkpoint['epoch']`); the orchestrator `tools/wildbox_final.sh` greps the log for ≥50 iter lines as a canary. |
| **Category-meta ordering** (per-class metrics collapse) | We use `data/category_meta_wildbox.json` for both training (label assignment) and oracle hook (dataset-id → contig); same file for both -> ordering can't drift. |
| **Path remapping after data move** | `tools/wildbox_smoke.sh` and `tools/wildbox_final.sh` accept `PATH_REMAP` env var; the converter rewrites image paths before pickling. |
| **GDino preprocessing collapse** | We don't run GDino in-process — we consume ovmono3d's precomputed JSON. Hazard not applicable. |

---

## 7. What we DON'T plan to verify in this branch (future work)

- Multi-seed mean±std (rare classes' wide error bars). Run after single
  seed numbers are sane.
- 5-species → 6-species regression analysis ([§10][1]). DetAny3D's
  per-class numbers will tell their own story.
- Closed-vocab RPN-transfer baseline. DetAny3D has no analog.
- GDino-prompted vs GT-2D ceiling. Out of scope per current decision (a)
  in our planning thread.
