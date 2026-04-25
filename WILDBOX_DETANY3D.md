# DetAny3D × WildBox — experiment documentation

The thorough reference for reproducing the WildBox monocular 3D wildlife
detection experiments on the **DetAny3D** architecture, as a parallel
row in ovmono3d's cross-architecture comparison.

**Companion docs (don't duplicate; reference)**:
- [FINAL_RUN_DETANY3D.md](FINAL_RUN_DETANY3D.md) — linear ops pipeline (copy-paste steps).
- [QUICK_START_SMOKE_TEST_DETANY3D.md](QUICK_START_SMOKE_TEST_DETANY3D.md) — ~1.5h smoke validation.

**ovmono3d cross-references** (the dataset is documented authoritatively
on the ovmono3d side; we don't re-document it here):
- [ovmono3d/WILDBOX_EXPERIMENT.md](../ovmono3d/WILDBOX_EXPERIMENT.md) §2 (dataset), §6 (eval protocol), §20 (cross-arch contract).
- [ovmono3d/FINAL_RUN.md](../ovmono3d/FINAL_RUN.md) (ovmono3d's own ops doc).
- [ovmono3d/QUICK_START_SMOKE_TEST.md](../ovmono3d/QUICK_START_SMOKE_TEST.md) (ovmono3d's smoke).

---

## 1. TL;DR

- **Task**: monocular 3D detection of African wildlife from drone footage, on the WildBox dataset (6 species, 15 zips, 64 vids / 60k frames / 237k 3D-cuboid instances; see ovmono3d's WILDBOX_EXPERIMENT.md §2).
- **Architecture**: DetAny3D — promptable SAM-ViT-H backbone + UniDepth metric-depth head + DINOv2 features, with a 3D cuboid head conditioned on a 2D prompt. Vendored components live under [detect_anything/modeling/](detect_anything/modeling/); only GroundingDINO is an external runtime dependency.
- **Cross-arch protocol** (per [ovmono3d §20.4](../ovmono3d/WILDBOX_EXPERIMENT.md)): we report DetAny3D in the **paper-protocol oracle-2D** mode using ovmono3d's precomputed `gdino_WildBox_val_oracle_2d.json` — same 2D source for every architecture in the comparison.
- **Reporting rows for DetAny3D** (single-seed; multi-seed deferred):
  1. **Zero-shot oracle 2D** — pretrained DetAny3D + GDino prompts.
  2. **Fine-tuned (single seed)** — same prompts, fine-tuned weights.
  3. *(Row 1 of ovmono3d's protocol — RPN-transfer — has no analog: DetAny3D is prompt-conditioned, no closed-vocab RPN to transfer. Footnote in the paper.)*
- **Metrics reported** (matches ovmono3d's main + supplementary tables):
  - **Primary**: AP_BEV @ 0.50 (cross-arch contract).
  - **Supplementary**: AP_BEV @ 0.25, AP_3D @ 0.25 / 0.50 / 0.75, 2D AP @ 0.5:0.95, class-agnostic 2D AP, disentangled NHD per (xy, z, dimensions, pose), Rel-AP_3D (LabelAny3D scale-aligned).
- **Where to start**:
  - **Ops**: [FINAL_RUN_DETANY3D.md](FINAL_RUN_DETANY3D.md).
  - **Smoke**: [QUICK_START_SMOKE_TEST_DETANY3D.md](QUICK_START_SMOKE_TEST_DETANY3D.md).
  - **Resume after compaction**: §9 (current state).

---

## 2. Architecture

### 2.1 What DetAny3D is

A promptable 3D detection foundation model. Forward call takes an image plus per-object 2D prompts (boxes or points) and outputs 3D cuboids. The 3D cube head is class-agnostic — what makes the paper-protocol oracle-2D row a valid cross-arch comparison.

Vendored components in this repo:
- **SAM ViT-H image encoder** ([detect_anything/modeling/image_encoder.py](detect_anything/modeling/image_encoder.py))
- **UniDepth depth predictor** (vendored as `pixel_encoder` / `pixel_decoder` in image_encoder.py + [backbones/](detect_anything/modeling/backbones/))
- **DINOv2 features** (loaded from external `dinov2_vitl14_pretrain.pth` checkpoint)
- **Prompt encoder** ([detect_anything/modeling/prompt_encoder.py](detect_anything/modeling/prompt_encoder.py))
- **Mask decoder + 3D heads** ([detect_anything/modeling/mask_decoder.py](detect_anything/modeling/mask_decoder.py)) — outputs 2D mask, 3D depth (log scale), 3D dimensions, 3D rotation (6D representation), 3D alpha-class.

External runtime dependencies:
- **GroundingDINO** at the pinned commit `856dde20...` from `requirements.txt`. Only its Python API is needed at module-import time; we *don't* call `load_model` in our pipeline (we consume ovmono3d's precomputed oracle JSON instead). ms_deform_attn CUDA-op build failures are non-fatal for our use case (CPU fallback warnings are harmless when GDino is never invoked).
- **mmcv 2.0.1 with CUDA ops** for the MultiScaleDeformableAttention used inside the SAM-style adapter at [detect_anything/modeling/ops/](detect_anything/modeling/ops/) — `from mmcv.ops.multi_scale_deform_attn import *`. Pip's stock mmcv wheel may ship without CUDA ops; install via OpenMMLab's wheel index (see §3.4).

### 2.2 Why Row 1 (RPN-transfer) has no DetAny3D analog

ovmono3d reports three rows per architecture:
1. Zero-shot RPN-transfer (closed-vocab, model's own 2D classifier)
2. Zero-shot GDino oracle (paper protocol, open-vocab 2D from GroundingDINO)
3. Fine-tuned

Row 1 measures whether the architecture's *closed-vocab 2D classifier* generalises to wildlife species not in its pretraining taxonomy. DetAny3D has no closed-vocab 2D classifier — it's a prompt-conditioned model with no internal RPN. Substituting GDino-as-its-own-2D-source makes Row 1 = Row 2, collapsing the distinction.

We omit Row 1 in the main paper table for DetAny3D and footnote the asymmetry. Cross-arch comparison stays apples-to-apples on Rows 2 and 3.

### 2.3 The four checkpoints

| Slot | File | Source | Notes |
|---|---|---|---|
| `checkpoints/sam_ckpts/sam_vit_h_4b8939.pth` | 2.4 GB | Meta SAM public release | publicly downloadable |
| `checkpoints/dino_ckpts/dinov2_vitl14_pretrain.pth` | 1.2 GB | Meta DINOv2 model zoo | publicly downloadable |
| `checkpoints/unidepth_ckpts/model.pth` | 4.1 GB | DetAny3D authors' Google Drive | only loaded when **not** resuming from a DetAny3D checkpoint (train.py line 513) |
| `checkpoints/detany3d_ckpts/detany3d.pth` (alias of `other_exp_ckpt.pth`) | 4.1 GB | DetAny3D authors' Google Drive | the one our wildbox configs `resume:` from |

For our wildbox pipeline the UniDepth checkpoint isn't actually loaded — the DetAny3D checkpoint already contains the (UniDepth-initialized + author-fine-tuned) full model state. Keep it on disk anyway in case you want to train from scratch.

---

## 3. Environment setup hazards (read before debugging)

The full step-by-step is in [FINAL_RUN_DETANY3D.md §1](FINAL_RUN_DETANY3D.md). The four most painful gotchas:

### 3.1 `requirements.txt` is unusable as-is with `pip install -r`

Has ~10 `@ file:///croot/...` lines (Brotli, certifi, idna, mkl-fft, etc.) — those are conda-build artifact paths from the upstream authors' machine and don't resolve anywhere else. Use the cleaned [requirements_clean.txt](requirements_clean.txt) which strips them, plus the torch family (installed separately via PyTorch's cu116 wheel index) and the GroundingDINO `-e git+` (cloned at the pinned commit separately).

Even with the cleaned file, run with `--no-deps` to bypass pip's resolver: `nuscenes-devkit==1.1.11` requires `matplotlib<3.6` while the rest of the stack pins `matplotlib==3.7.5`. Upstream clearly installed via `pip freeze` of a hand-resolved env; we mirror that with `--no-deps`. Inspecting transitive constraints is unnecessary because every transitive dep is already explicitly pinned in the file.

### 3.2 `opencv-python` and `opencv-python-headless` collide on the same `cv2/` package directory

`requirements.txt` pins **both** versions. Whichever installs second wins; uninstalling that one wipes the entire `cv2/` package. Plus, on a headless GPU node `import cv2` from the non-headless wheel raises `libGL.so.1: cannot open shared object file`.

Fix: keep only the headless one. After the bulk install, run:

```bash
pip uninstall -y opencv-python
pip install --force-reinstall --no-deps opencv-python-headless==4.10.0.84
```

GroundingDINO's `setup.py develop` (next step) silently re-pulls non-headless `opencv-python` as a dependency — repeat the uninstall/force-reinstall after `pip install -e .` on GroundingDINO.

Same hazard ovmono3d documents in [WILDBOX_EXPERIMENT.md §3.1.4](../ovmono3d/WILDBOX_EXPERIMENT.md).

### 3.3 mmcv from pip ships without CUDA ops; the SAM adapter dies at runtime

The SAM-style adapter ([detect_anything/modeling/adaper.py](detect_anything/modeling/adaper.py)) uses MSDeformAttn whose `ext_module` is wildcard-imported from `mmcv.ops.multi_scale_deform_attn`. Stock pip-installed `mmcv==2.0.1` may not include the compiled CUDA op, surfacing as:

```
RuntimeError: ms_deform_attn_impl_forward: implementation for device cuda:0 not found.
```

Fix: install from OpenMMLab's CUDA-ops wheel index for our torch+CUDA combo:

```bash
pip uninstall -y mmcv
pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html
```

Verify:

```bash
python -c "from mmcv.ops.multi_scale_deform_attn import ext_module; print([a for a in dir(ext_module) if 'deform' in a])"
# expect: ['ms_deform_attn_backward', 'ms_deform_attn_forward']
```

### 3.4 Released DetAny3D checkpoint stores `epoch=93` — silent-skip-training (CRITICAL)

The released checkpoint's `epoch=93` field gets read into `start_epoch` when `cfg.resume:` is set. With `cfg.num_epochs<=93` the outer loop `range(start_epoch, num_epochs)` becomes empty and the run exits without doing anything. Affects both inference (`inference_only=True`) and fresh fine-tunes from the pretrained init.

Same family of bug as ovmono3d's iteration-skip ([ovmono3d §3.1.1](../ovmono3d/WILDBOX_EXPERIMENT.md)). Fixed in [train.py:499-512](train.py#L499-L512):

```python
is_continuation = cfg.resume_scheduler and not cfg.inference_only
start_epoch = checkpoint['epoch'] if is_continuation else 0
```

Canary check after any training run: `grep -cE "iter[er]?[: ][0-9]+" exps/<run>/log.txt` should be > 50. The orchestrator does this automatically and exits non-zero if it isn't.

### 3.5 Other gotchas worth flagging

- **`libGL.so.1` on compute nodes** — same as ovmono3d's [§3.1.4](../ovmono3d/WILDBOX_EXPERIMENT.md). Use `opencv-python-headless` only (see §3.2).
- **`KeyError: 'gradient_checkpointing'` style errors from GDino** during model load — only matter if we actually call GDino at runtime, which we don't.
- **DDP with all parameters frozen** — eval configs that set `freeze.{image_encoder, prompt_encoder, mask_decoder}: True` break PyTorch's DDP wrap with `RuntimeError: DistributedDataParallel is not needed when a module doesn't have any parameter that requires a gradient`. Set all three to `False` in eval configs (the model still runs in eval mode via `model.eval()`).
- **Empty `prepare_for_dsam`** — when `filter_objects` rejects every annotation in a frame (e.g. animals at frame edges with off-screen 3D centers), training crashes with `element 0 of tensors does not require grad`. Patched in [detany3d_dataset.py:236](detect_anything/datasets/detany3d_dataset.py#L236) to recurse to a random sample.

---

## 4. Pipeline

Five additive pieces on this branch — no model changes, no upstream-evaluator changes:

| Component | Path |
|---|---|
| WildBox JSON → DetAny3D pickle converter | [detect_anything/datasets/data_creator/wildbox.py](detect_anything/datasets/data_creator/wildbox.py) |
| Oracle-2D loader hook | [detect_anything/datasets/detany3d_dataset.py](detect_anything/datasets/detany3d_dataset.py) — `generate_oracle_list` method, activated by `cfg.dataset.oracle_2d_input: True` |
| 6-species category metadata | [data/category_meta_wildbox.json](data/category_meta_wildbox.json) |
| Smoke / final / eval configs | [detect_anything/configs/wildbox/](detect_anything/configs/wildbox/) |
| DetAny3D JSON → ovmono3d `instances_predictions.pth` exporter | [tools/wildbox_export_predictions.py](tools/wildbox_export_predictions.py) |
| Standard Omni3D AP_3D / 2D AP / Rel-AP_3D evaluator (wraps ovmono3d's `_evaluate_predictions_on_omni`) | [tools/wildbox_full_metrics.py](tools/wildbox_full_metrics.py) |
| Smoke + final orchestrators | [tools/wildbox_smoke.sh](tools/wildbox_smoke.sh), [tools/wildbox_final.sh](tools/wildbox_final.sh) |

Step-level rationale:

1. **Convert** — read ovmono3d's `WildBox_{train,val}.json` (Omni3D schema, absolute paths, K per image, SAM3-tight `bbox`, `bbox3D_cam`, `center_cam`, `dimensions[W,H,L]`, `R_cam`); write a pickle list with the per-image dict shape DetAny3D's `DetAny3DDataset` expects. The dimension ordering swap and a sanity-check projection-verify pass are documented in §5.1 below.
2. **Zero-shot oracle eval** — load pretrained DetAny3D weights, eval on full val pickle, prompts come from the precomputed GDino oracle JSON (loaded by `generate_oracle_list`). Saves a JSON of per-image predictions.
3. **Fine-tune** — same data, run the model with all losses except `depth_loss` (synthetic-scale GT, see §5.4 below). Multi-GPU DDP via `torchrun --nproc_per_node=N`.
4. **Fine-tuned eval** — re-run [2] with the fine-tuned checkpoint as `resume:`.
5. **Export** — convert each DetAny3D JSON (`exps/.../wildbox_*.json`) to ovmono3d's `instances_predictions.pth` schema. Reverses the dimension ordering back to Omni3D's `[W, H, L]`.
6. **Score** — run ovmono3d's eval scripts on the exported predictions:
   - `bev_ap_eval.py` for AP_BEV @ {0.25, 0.50}.
   - `class_agnostic_eval.py --nhd` for class-agnostic 2D AP + disentangled NHD.
   - `tools/wildbox_full_metrics.py` (this repo, runs from ovmono3d's env) for standard AP_3D / 2D AP / Rel-AP_3D.
   - `visualize_class_agnostic.py` for the 2×3 paper-style figures.

---

## 5. Configs

All under [detect_anything/configs/wildbox/](detect_anything/configs/wildbox/).

### 5.1 Convention swaps (read before changing dim/rotation handling)

ovmono3d / Omni3D stores `dimensions = [W, H, L]` with axis assignment `X = L, Y = H, Z = W` (per [ovmono3d §2.4](../ovmono3d/WILDBOX_EXPERIMENT.md)). DetAny3D's `compute_3d_bbox_vertices` ([utils.py:148](detect_anything/datasets/utils.py#L148)) takes `[w, h, l]` where `w` is X-extent, `h` is Y-extent, `l` is Z-extent.

Therefore:
```
DetAny3D[w, h, l] = [Omni3D.L, Omni3D.H, Omni3D.W] = reversed(WildBox.dimensions)
```

The converter applies this swap when writing pickles; the exporter applies the reverse when emitting `instances_predictions.pth`. `R_cam` is passed through unchanged (both conventions treat it as "rotation from local cuboid frame to camera frame"). If post-training BEV / 3D AP looks anomalous, run the converter with `--verify-projection` — it reprojects every cuboid via DetAny3D's convention and compares to the GT `bbox2D_proj`. >5% failures = the convention swap is wrong.

### 5.2 Training: `wildbox_smoke.yaml` and `wildbox_final.yaml`

Both inherit identical structure. Differences:

| | smoke | final |
|---|---|---|
| `num_epochs` | 3 | 3 |
| Train pickle | `WildBox_smoke_train.pkl` | `WildBox_train.pkl` |
| Val pickle | `WildBox_smoke_val.pkl` | `WildBox_val.pkl` |
| Subsample interval | typically 50 (train) / 20 (val) | 1 |
| `batch_size` per GPU | 1 (constant; image size 896×896, ViT-H is heavy) | 1 |
| GPUs | 1 (smoke is fast enough single-GPU) | 4–8 (DDP) |

Common settings:
- `resume:` → `./checkpoints/detany3d_ckpts/detany3d.pth`
- `resume_scheduler: False` (fresh fine-tune from pretrained; combined with the §3.4 patch this gives `start_epoch=0`).
- `inference_only: False`.
- `freeze.image_encoder: True`, `prompt_encoder: False`, `mask_decoder: False` (fine-tune the 3D heads + adapters; backbone frozen).
- `tune_with_depth: False`, `loss.loss_list: ['intrinsic_loss', '2d_bbox_loss', '3d_bbox_loss']` (depth excluded — WildBox depth is synthetic per-segment scale, not metric).
- `output_rotation_matrix: True` (drone shots have non-trivial pitch/roll).
- `provide_gt_intrinsics: True` (use VGGT's per-frame K from the JSON).
- `add_cubercnn_for_ap_inference: True` (so the validation loop dumps the JSON we'll export).
- `use_amp: True` (~1.5–2× throughput on Ampere; A40 has tensor cores).
- `dataset.zero_shot: True` — the `filter_objects` taxonomy gate at [detany3d_dataset.py:593](detect_anything/datasets/detany3d_dataset.py#L593) is bypassed for unknown dataset names ('wildbox' isn't in the kitti/sunrgbd/arkitscenes/etc. allowlist), so all 6 wildlife classes pass through.

### 5.3 Eval: `wildbox_eval_oracle2d.yaml`

Same backbone as training, plus:
- `inference_only: True`, `eval_interval: 1`.
- `freeze.{image_encoder, prompt_encoder, mask_decoder}: False` (DDP wrap requires at least one trainable param even though we never backward — see §3.5).
- `dataset.oracle_2d_input: True`, `dataset.oracle_2d_path: ./datasets/Omni3D/gdino_WildBox_val_oracle_2d.json`. Activates `generate_oracle_list` which replaces GT/in-process-GDino prompts with the loaded oracle boxes.
- `loss.loss_list: []` (eval-only).
- `dataset.perturbation_box_prompt: False` (don't jitter the oracle boxes).

Used twice in the orchestrator: once for the zero-shot row (`resume` = pretrained DetAny3D) and once for the fine-tuned row (`resume` = checkpoint from `wildbox_final.yaml`'s output).

### 5.4 Why depth loss is excluded

WildBox's 3D GT comes from VGGT pseudo-labels with per-segment uniform scale normalization (median |Z|=1, not metric). DetAny3D's depth head was pretrained on metric depth (KITTI, nuScenes, SUN-RGBD, ARKitScenes, Hypersim). Supervising the depth head against synthetic-scale GT pulls it in inconsistent directions.

Three layers of defense:
1. `tune_with_depth: False` (top-level cfg flag).
2. Drop `depth_loss` from `loss.loss_list`.
3. `depth_path: None` in every pickle entry → loaded as zeros, depth mask zeroed out by `process_depth`'s `min_distance` filter, no pixels contribute.

Depth predictions still come out at inference (the head still runs because subsequent 3D-cuboid heads condition on depth features); they're passed through to `instances_predictions.pth` unchanged. ovmono3d's BEV AP ignores Z; Rel-AP_3D handles the global scale offset.

This matches ovmono3d's [§5.4](../ovmono3d/WILDBOX_EXPERIMENT.md) rationale (downweight depth for synthetic-scale GT) but goes further — full disable rather than 0.5×. Cross-architecture comparison stays apples-to-apples because both architectures effectively treat WildBox as 2D + dimensions + pose, ignoring the Z-axis prior.

---

## 6. Evaluation

### 6.1 Metric coverage

Same set of metrics ovmono3d reports in their main + supplementary paper tables. Tools that produce them, and which env they run in:

| Metric | Tool | Env | Notes |
|---|---|---|---|
| AP_BEV @ 0.25 / 0.50 | `ovmono3d/tools/bev_ap_eval.py` | ovmono3d | **primary cross-arch metric** ([§20.4](../ovmono3d/WILDBOX_EXPERIMENT.md)) |
| Class-agnostic 2D AP @ 0.25 / 0.50 / 0.75 | `ovmono3d/tools/class_agnostic_eval.py --nhd` | ovmono3d | also outputs disentangled NHD per (xy, z, dimensions, pose) |
| Standard AP_3D @ 0.25 / 0.50 / 0.75 | [tools/wildbox_full_metrics.py](tools/wildbox_full_metrics.py) | ovmono3d | wraps ovmono3d's `_evaluate_predictions_on_omni`; needs pytorch3d-CPU |
| 2D AP @ 0.5:0.95 (COCO) | same | ovmono3d | computed alongside 3D in the same call |
| Rel-AP_3D | same, with `--eval-rel-ap3d --rel-ap3d-search 0.05,3.0,32` | ovmono3d | LabelAny3D scale-aligned grid; widened lower bound for zero-shot scale |
| 2×3 paper visualizations | `ovmono3d/tools/visualize_class_agnostic.py` | ovmono3d | `gt_only` + `pred_only` + `combined`, with/without ground grid |

Side-by-side compare table: `ovmono3d/tools/make_report.py --compare` ingests `bev_ap.json` + `summary_nhd.txt` + `full_metrics/summary.json` from each row's directory.

### 6.2 Aggregations

Per [ovmono3d §6.2](../ovmono3d/WILDBOX_EXPERIMENT.md): every metric reported three ways — micro (over all preds), macro (mean per-class), and per-class. The new `wildbox_full_metrics.py` produces all three; `bev_ap_eval.py` and `class_agnostic_eval.py` already do.

### 6.3 What our eval glue normalises

Two key conversions in the bridge from DetAny3D's exported JSON to ovmono3d's evaluator inputs:

1. **Dimension ordering**: DetAny3D `[w, h, l]` → Omni3D `[W, H, L]` (reversed). Done in [tools/wildbox_export_predictions.py](tools/wildbox_export_predictions.py).
2. **Category ID space**: DetAny3D's predictions carry contiguous `category_id` 0..5 (its training-time label space); ovmono3d's COCOEvaluator subclass expects dataset_ids 1000..1005. Done at the predictions-side in [tools/wildbox_full_metrics.py](tools/wildbox_full_metrics.py). `bev_ap_eval.py` normalises both directions internally so it doesn't need this remap; `_evaluate_predictions_on_omni` does.

If per-class AP comes out 0 for every class, suspect the category-ID remap. If specific classes look wrong (e.g. only one non-zero), suspect the contiguous-ID *ordering* — verify the ordering in `category_meta_wildbox.json` is sorted by dataset-id ascending (same rule ovmono3d documents in [§2.8](../ovmono3d/WILDBOX_EXPERIMENT.md)).

### 6.4 Why the 2D AP is identical between zero-shot and fine-tuned rows

DetAny3D doesn't predict 2D boxes — it lifts a 2D prompt to 3D. With oracle 2D as the prompt, the exporter echoes the oracle box back as the prediction's 2D `bbox`. Both rows therefore have identical 2D inputs and outputs — the only difference is the 3D lift. **2D AP being equal across rows is correct.** The variation between rows is in 3D AP, BEV AP, NHD, and Rel-AP_3D.

---

## 7. Adapting to new species or zips

For new zips (same species), or new species, follow ovmono3d's [§8](../ovmono3d/WILDBOX_EXPERIMENT.md). The DetAny3D side requires three adaptations on top:

1. Edit [data/category_meta_wildbox.json](data/category_meta_wildbox.json) to add the new species — keep `thing_classes` sorted by ascending dataset-id (same rule as ovmono3d).
2. Re-run the converter — pickles auto-pick up the new categories from the WildBox JSON.
3. No code change needed in DetAny3D's training loop — `dataset.zero_shot: True` means the taxonomy gate is bypassed for any unknown dataset name.

---

## 8. Bugs caught and fixed (don't re-introduce)

| # | Symptom | Root cause | Fix | Commit |
|---|---|---|---|---|
| 1 | Per-class AP all 0 in `wildbox_full_metrics` | Predictions carry contiguous `category_id`; evaluator expects dataset_id | Build contiguous→dataset_id map from GT, remap before eval | `1b0ab2a` |
| 2 | `KeyError: 'depth'` on GT during 3D area-range filter | WildBox JSON has `center_cam` but no top-level `depth` field | Synthesise `depth = float(ann["center_cam"][2])` after loading GT | `8c75089` |
| 3 | `KeyError: 'bbox3D'` on GT during 3D IoU computation | WildBox JSON has `bbox3D_cam` but evaluator reads `bbox3D` | Alias `bbox3D <- bbox3D_cam` after loading GT | `0eca64c` |
| 4 | `RuntimeError: ms_deform_attn_impl_forward: implementation for device cuda:0 not found` | pip's stock mmcv lacks CUDA ops | Reinstall via OpenMMLab cu116 wheel index | env fix |
| 5 | `RuntimeError: element 0 of tensors does not require grad` mid-training | Empty `prepare_for_dsam` when `filter_objects` rejects every annotation | Recurse to a random sample instead of returning `[]` | `6c89796` |
| 6 | `RuntimeError: DistributedDataParallel is not needed when a module doesn't have any parameter that requires a gradient` | Eval config froze every component | Set all `freeze.*: False` in eval config; `model.eval()` still applies | config |
| 7 | **Silent-skip-training**: train run exits without iterating | Released checkpoint has `epoch=93`; resume sets `start_epoch=93` > `num_epochs` | `start_epoch=0` when not continuing a prior training run | `eeee735` |
| 8 | `cv2.error: libGL.so.1: cannot open shared object file` | `opencv-python` (non-headless) needs system graphics libs not present on compute nodes | Use `opencv-python-headless` only | env fix |
| 9 | `requirements.txt` has `@ file:///croot/...` lines that fail with `pip install -r` | Conda-build artifact paths from upstream authors' machine | Use cleaned [requirements_clean.txt](requirements_clean.txt) with `--no-deps` | branch artifact |

Bug #7 is the silent killer; canary check on every training run is `grep -cE "iter[er]?[: ][0-9]+" exps/<run>/log.txt` > 50.

---

## 9. Cross-architecture protocol — DetAny3D's row

Per [ovmono3d §20.5](../ovmono3d/WILDBOX_EXPERIMENT.md). What we hold fixed:

- Same val set: `WildBox_val.json` (13 779 images, 6 species)
- Same precomputed GDino oracle: `gdino_WildBox_val_oracle_2d.json`
- Same `category_meta` ordering (dataset-id ascending, contiguous 0..5)
- Same eval scripts: `bev_ap_eval.py`, `class_agnostic_eval.py`, `make_report.py`

What we vary (relative to ovmono3d):

- Architecture (DetAny3D, prompt-conditioned, vs OVMono3D-Lift, RPN-based)
- Backbone pretraining (SAM ViT-H + DINOv2 + UniDepth, vs DINOv2 ViT-B/14)
- Loss weights specific to the architecture (DetAny3D has separate intrinsic_loss, 2d_bbox_loss, 3d_bbox_loss; OVMono3D-Lift has Cube R-CNN's loss decomposition)
- Optimizer (DetAny3D: AdamW lr=1e-5; OVMono3D: SGD lr=2e-3)
- Schedule (DetAny3D: cosine; OVMono3D: multistep)
- Multi-GPU policy (DetAny3D: torchrun DDP; OVMono3D: detectron2's launcher)

Reportable rows (matches [ovmono3d §20.5](../ovmono3d/WILDBOX_EXPERIMENT.md)):

| Row | Arch | Backbone | 3D head | 2D source | Numbers |
|---|---|---|---|---|---|
| Z-OV | OVMono3D-Lift | DINOv2 ViT-B/14 | Cube R-CNN | RPN-transfer | from ovmono3d's `wl6_zeroshot_rpn` |
| Z-OV-O | OVMono3D-Lift | DINOv2 ViT-B/14 | Cube R-CNN | **GDino oracle** | from ovmono3d's `wl6_zeroshot_oracle2d` |
| F-OV | OVMono3D-Lift | DINOv2 ViT-B/14 | Cube R-CNN | own RPN (FT) | from ovmono3d's `wl6_rt0.5_multiseed/seed*` |
| Z-DA-O | DetAny3D | SAM-H + DINOv2-L + UniDepth | DetAny3D 3D heads | **GDino oracle** | from `output/wildbox_detany3d_zeroshot_oracle/` |
| F-DA-O | DetAny3D | SAM-H + DINOv2-L + UniDepth | DetAny3D 3D heads | GDino oracle (FT) | from `output/wildbox_detany3d_finetuned_oracle/` |

DetAny3D rows are produced by `tools/wildbox_final.sh`. Each row's directory has `bev_ap.json`, `summary_nhd.txt`, `full_metrics/summary.json`, `full_metrics/log.{2D,3D,3D-Rel}.txt`, and `vis_ovmono3d/img_*.jpg`.

`make_report.py --compare` ingests these directories the same way it does ovmono3d's own rows.

---

## 10. Current state (as of branch tip)

- **Branch**: `wildbox_detany3d` on `https://github.com/vandyshukla04/DetAny3D`
- **Env**: `/storage3/3DOM/vshukla/envs/detany3d` on the cluster (Python 3.8, torch 1.13.1+cu116, mmcv 2.0.1 with CUDA ops, opencv-python-headless, GroundingDINO at the pinned commit)
- **Smoke run**: green end-to-end. Numbers (1/50 train subsample × 3 epochs at this scale, expect 5-10× higher on full data):
  - Zero-shot: AP_3D@0.25=0.0, BEV macro@0.25=0.0, NHD best-scale=0.10 (severely under-scaled)
  - Fine-tuned: AP_3D@0.25=0.0018, BEV macro@0.25=2.61, NHD best-scale=0.94 (recovered to ≈unit scale)
  - Pattern matches ovmono3d's documented expectations: pretrained 3D priors fail for wildlife synthetic scale, fine-tuning recovers it.
- **Final run**: ready to launch (see [FINAL_RUN_DETANY3D.md](FINAL_RUN_DETANY3D.md))
- **Multi-seed**: deferred. Re-run [tools/wildbox_final.sh](tools/wildbox_final.sh) with `SEED=1`, `SEED=2` (after adding seed handling to the config) for rare-class mean±std reporting per [ovmono3d §6.5](../ovmono3d/WILDBOX_EXPERIMENT.md).

Future work in priority order:
1. Multi-seed (3 × full final).
2. RPN-transfer-equivalent row for DetAny3D — would need a learned classification head on the prompt encoder; out of scope for v1.
3. Higher input resolution for small-class bottleneck (gazelle).
4. Ensemble across seeds.
