# DetAny3D × WildBox — final paper run

Linear ops pipeline for the DetAny3D-side of the WildBox cross-architecture
comparison. Every step is a single copy-paste block. For the "why" behind any
command, see [WILDBOX_DETANY3D.md](WILDBOX_DETANY3D.md). For dataset-level
documentation, see [ovmono3d/WILDBOX_EXPERIMENT.md](../ovmono3d/WILDBOX_EXPERIMENT.md).

**Cluster:** `node81` or `node82` (gpu-A40 partition, 8× A40 each), env at
`/storage3/3DOM/vshukla/envs/detany3d`, repo at `/storage3/3DOM/vshukla/DetAny3D`.

**Important — multi-GPU is currently broken on this cluster.** Two 4-GPU
attempts hung at NCCL ALLGATHER `SeqNum=1` (see [WILDBOX_DETANY3D.md §3.5
and bug #10](WILDBOX_DETANY3D.md)). The headline run is **single-GPU**;
multi-GPU debugging is deferred to follow-up work.

**Run time budget (1× A40, 1 epoch full train, val at interval=4):**
- Data prep: ~2 min (cached after first run)
- Zero-shot oracle eval (3 445 val images at interval=4): ~50 min on 1 GPU
- Fine-tune (46k × 1 epoch): ~6 h on 1 GPU
- Fine-tuned oracle eval: ~50 min on 1 GPU
- Full eval suite (BEV + class_agnostic + standard 3D AP + Rel-AP_3D): ~30 min
- Visualizations + report: ~10 min
- **Total: ~8.5 h** — fits in a `--time=10:00:00` srun

Caveats:
- Eval at `interval=4` (3 445 val images) gives 25%-of-full-val numbers. Same
  protocol still produces directly-rankable metrics, but absolute values are
  noisier than ovmono3d's full-val eval. Re-eval at `interval=1` later in a
  separate srun if you want directly-comparable absolute numbers.
- 1 epoch training (46k samples seen) is less than ovmono3d's effective
  training budget (15k iter × batch 8 = 120k samples ≈ 2.6 epochs). Defensible
  for "fine-tuning from a strong pretrained head"; document this in the paper.

---

## 0. Prereqs (one-time per cluster; skip if env already works)

If the env doesn't exist yet, follow [WILDBOX_DETANY3D.md §3](WILDBOX_DETANY3D.md) for the build steps. Quick verification:

```bash
ssh frontendnew
conda activate /storage3/3DOM/vshukla/envs/detany3d
cd /storage3/3DOM/vshukla/DetAny3D
git pull   # always start from latest

# All four must succeed.
python -c "
import torch, xformers, mmcv, transformers
from mmcv.ops.multi_scale_deform_attn import ext_module
import groundingdino
from detect_anything.datasets.detany3d_dataset import DetAny3DDataset
from detect_anything.datasets.data_creator.wildbox import convert
print('imports ok; mmcv ext fns:', [a for a in dir(ext_module) if 'deform' in a])
"

# All four checkpoints must exist.
ls -la checkpoints/sam_ckpts/sam_vit_h_4b8939.pth \
       checkpoints/dino_ckpts/dinov2_vitl14_pretrain.pth \
       checkpoints/unidepth_ckpts/model.pth \
       checkpoints/detany3d_ckpts/detany3d.pth
```

If any check fails, jump to [WILDBOX_DETANY3D.md §3](WILDBOX_DETANY3D.md).

---

## 1. Locate ovmono3d artifacts (cluster-side)

```bash
export OVMONO3D_REPO=/storage2/3DOM/vshukla/repos/ovmono3d
export WILDBOX_TRAIN_JSON=$OVMONO3D_REPO/datasets/Omni3D/WildBox_train.json
export WILDBOX_VAL_JSON=$OVMONO3D_REPO/datasets/Omni3D/WildBox_val.json
export GDINO_ORACLE_JSON=$OVMONO3D_REPO/datasets/Omni3D/gdino_WildBox_val_oracle_2d.json

ls -la $WILDBOX_TRAIN_JSON $WILDBOX_VAL_JSON $GDINO_ORACLE_JSON
```

All three must exist (they're produced by ovmono3d's own pipeline; the GDino oracle is the ~67-min one-time precompute documented in [ovmono3d §6.4.2](../ovmono3d/WILDBOX_EXPERIMENT.md)).

---

## 2. Claim a 1-GPU srun on gpu-A40

```bash
# Check who's using node81/82's GPUs first.
sinfo -p gpu-A40 -o "%n %T %G %C %m"
ssh node81 "nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv" 2>/dev/null

# 1 GPU + 200G mem (image-decode workers want a lot of CPU RAM; <128G has
# OOM-killed `import torch` itself at job start).
srun --partition=gpu-A40 --gres=gpu:1 --mem=200G --cpus-per-task=16 \
     --time=10:00:00 --pty bash
```

**Don't request 4 GPUs.** Two prior 4-GPU attempts hung at NCCL ALLGATHER
SeqNum=1 with no recovery (see [WILDBOX_DETANY3D.md §3.5](WILDBOX_DETANY3D.md)).
Even with the 4-hour timeout patch in commit `db7bb37`, training never
completes a single iteration in DDP mode on this cluster.

Inside the srun shell:

```bash
conda activate /storage3/3DOM/vshukla/envs/detany3d
cd /storage3/3DOM/vshukla/DetAny3D
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv
python -c "import torch; print('cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())"
```

Must print `cuda True gpus 1`. If `Killed` appears, raise `--mem=256G`.

---

## 3. Pre-run config tweaks (before launching)

Two edits the orchestrator doesn't make automatically — apply once per session:

```bash
cd /storage3/3DOM/vshukla/DetAny3D

# (a) num_epochs: 1 in wildbox_final.yaml. With ~46k samples × 1 epoch on
#     1×A40 at ~2 it/s, training is ~6 h. num_epochs: 2 → ~12 h, busts srun.
sed -i 's/^num_epochs: [0-9]\+/num_epochs: 1/' \
    detect_anything/configs/wildbox/wildbox_final.yaml
grep "^num_epochs:" detect_anything/configs/wildbox/wildbox_final.yaml

# (b) Reset wildbox_eval_oracle2d.yaml's val interval. If a prior smoke
#     debug session set interval: 20, the final eval would silently
#     subsample to 689 of 13779 val images and report duplicate-of-smoke
#     numbers. interval: 4 fits two evals (zero-shot + fine-tuned) into
#     ~50 min each on 1 GPU. Set 1 instead if you have time-budget for
#     full-val eval (~3 h × 2).
python <<'PY'
import re, yaml
p = 'detect_anything/configs/wildbox/wildbox_eval_oracle2d.yaml'
s = open(p).read()
s = re.sub(r'(range:\s*\{[^}]*?)interval:\s*\d+', r'\1interval: 4', s)
# Normalize pkl paths to WildBox_val.pkl (smoke pickle is byte-identical anyway).
s = re.sub(r"pkl_path:\s*'\./data/pkls/wildbox/WildBox_smoke_val\.pkl'",
           "pkl_path: './data/pkls/wildbox/WildBox_val.pkl'", s)
open(p, 'w').write(s)
cfg = yaml.safe_load(open(p))
print('eval val:', cfg['dataset']['val']['wildbox'])
PY
```

---

## 4. Run the orchestrator

```bash
export OVMONO3D_REPO=/storage2/3DOM/vshukla/repos/ovmono3d
export WILDBOX_TRAIN_JSON=$OVMONO3D_REPO/datasets/Omni3D/WildBox_train.json
export WILDBOX_VAL_JSON=$OVMONO3D_REPO/datasets/Omni3D/WildBox_val.json
export GDINO_ORACLE_JSON=$OVMONO3D_REPO/datasets/Omni3D/gdino_WildBox_val_oracle_2d.json
export NUM_GPUS=1
export PYTHONUNBUFFERED=1

mkdir -p logs
bash tools/wildbox_final.sh 2>&1 | tee logs/final_run.log
```

The orchestrator runs seven stages — each prints its own banner. What to expect:

### 4.1 Stage `[1/7]` — Convert WildBox JSONs → DetAny3D pickles

```
[1/7] Convert WildBox JSONs -> DetAny3D pickles
============================================================
INFO wildbox_converter | wrote 45979 samples (170554 obj) -> data/pkls/wildbox/WildBox_train.pkl
INFO wildbox_converter | projection verify: 170554/170554 ok (tol=4 px), worst_dist=0.00 px
INFO wildbox_converter | wrote 13779 samples (66951 obj) -> data/pkls/wildbox/WildBox_val.pkl
INFO wildbox_converter | projection verify: 66951/66951 ok (tol=4 px), worst_dist=0.00 px
  train: 45979 images, 170554 objects
  val: 13779 images, 66951 objects
```

`projection verify` must report 100% OK on both splits — that's the convention-swap sanity check ([WILDBOX_DETANY3D.md §5.1](WILDBOX_DETANY3D.md)).

### 4.2 Stage `[2/7]` — Zero-shot oracle eval

`exps/wildbox_final_zeroshot_oracle/<ts>/` fills with predictions. At interval=4 (3,445 val images), ~1.3 it/s on A40 = **~50 min**. If you see `Number of samples: 13778` (full val) instead of `~3445`, the interval=4 edit didn't apply — Ctrl-C, fix step 3(b), restart.

### 4.3 Stage `[3/7]` — Fine-tune (1-GPU)

```
[3/7] Final fine-tune (NUM_GPUS=1)
```

Watch for `iter:` lines via `PYTHONUNBUFFERED=1`. Throughput ≈ 2 it/s on 1× A40. 46k iter (1 epoch) → **~6 h**.

If `iter:` lines never appear and the run exits in seconds, **that's the silent-skip bug #7** — see [WILDBOX_DETANY3D.md §3.4](WILDBOX_DETANY3D.md). The orchestrator's `[3/7]` step exits non-zero if `< 50 iter:` lines are logged.

If `iter:` lines never appear and the run **doesn't** exit but GPU stays at 100% with no progress for >30 min, that's the **NCCL hang bug #10** — but it shouldn't happen on 1 GPU since no NCCL collectives fire. If it does, [WILDBOX_DETANY3D.md §3.5](WILDBOX_DETANY3D.md) covers the deeper investigation.

### 3.4 Stage `[4/7]` — Fine-tuned oracle eval

Same shape as `[2/7]`, with the fine-tuned checkpoint substituted into `wildbox_eval_oracle2d.yaml`'s `resume:` field.

### 3.5 Stage `[5/7]` — Export to ovmono3d format

```
[5/7] Export both runs to ovmono3d instances_predictions.pth
```

Two `instances_predictions.pth` files written under `$OVMONO3D_REPO/output/wildbox_detany3d_{zeroshot,finetuned}_oracle/inference/iter_final/WildBox_val/`.

### 3.6 Stage `[6/7]` — Full eval suite

The orchestrator switches to ovmono3d's conda env (shapely 2.x + pytorch3d-CPU) and runs four scoring tools per row:

```
[6/7] Full eval suite on both rows (BEV + class_agnostic +
      standard AP_3D/2D AP via Omni3DEvaluator + Rel-AP_3D)
```

Output per row directory: `bev_ap.json`, `summary_nhd.txt`, `full_metrics/summary.json`, `full_metrics/log.{2D,3D,3D-Rel}.txt`.

### 4.4 Stages `[4/7]` – `[7/7]` — fine-tuned eval, export, full eval suite, visualizations

Same as the smoke-tested versions:
- `[4/7]` runs `wildbox_eval_oracle2d.yaml` with the FT checkpoint patched into `resume:` via `sed -i.bak` (auto-restored after).
- `[5/7]` exports both prediction JSONs to `instances_predictions.pth` under `$OVMONO3D_REPO/output/wildbox_detany3d_{zeroshot,finetuned}_oracle/inference/iter_final/WildBox_val/`.
- `[6/7]` switches to ovmono3d's conda env (shapely 2.x + pytorch3d-CPU) and runs `bev_ap_eval.py`, `class_agnostic_eval.py --nhd`, [`tools/wildbox_full_metrics.py`](tools/wildbox_full_metrics.py) (with `--eval-rel-ap3d`), per row.
- `[7/7]` runs `visualize_class_agnostic.py` with `--every 100 --limit 40` per row, producing ~40 paper-style 2×3 images.

Optional `make_report.py --compare` writes `$OVMONO3D_REPO/output/paper_report_detany3d/report.md`.

### 4.5 Manual recovery if the orchestrator fails mid-run

If `[2/7]` succeeded but `[3/7]` crashed (e.g. NCCL hang on a multi-GPU attempt), you don't need to redo the zero-shot eval. Phase A scores the existing zero-shot prediction; Phase B runs only training + FT eval.

**Phase A — score the existing zero-shot row:**

```bash
conda activate /storage3/3DOM/vshukla/envs/ovmono3d
cd /storage2/3DOM/vshukla/repos/ovmono3d
DA3D_REPO=/storage3/3DOM/vshukla/DetAny3D
GT=$OVMONO3D_REPO/datasets/Omni3D/WildBox_val.json
ZS_PRED_JSON=$(ls -t $DA3D_REPO/exps/wildbox_final_zeroshot_oracle/*/wildbox_*.json | head -1)
OV_ZS_DIR=$OVMONO3D_REPO/output/wildbox_detany3d_zeroshot_oracle
mkdir -p "$OV_ZS_DIR/inference/iter_final/WildBox_val"

python $DA3D_REPO/tools/wildbox_export_predictions.py \
    --da3d-json "$ZS_PRED_JSON" --gt-json "$GT" \
    --out-pth "$OV_ZS_DIR/inference/iter_final/WildBox_val/instances_predictions.pth"
PREDS="$OV_ZS_DIR/inference/iter_final/WildBox_val/instances_predictions.pth"
python tools/bev_ap_eval.py --gt "$GT" --preds "$PREDS" --out "$OV_ZS_DIR/bev_ap.json"
python tools/class_agnostic_eval.py --gt "$GT" --preds "$PREDS" --nhd > "$OV_ZS_DIR/summary_nhd.txt" 2>&1
python $DA3D_REPO/tools/wildbox_full_metrics.py \
    --predictions "$PREDS" --gt "$GT" --ovmono3d-repo $OVMONO3D_REPO \
    --out-dir "$OV_ZS_DIR/full_metrics"
python tools/visualize_class_agnostic.py \
    --preds "$PREDS" --gt "$GT" --out "$OV_ZS_DIR/vis_ovmono3d" \
    --top-k 5 --every 100 --limit 40
```

**Phase B — run only training + FT eval:**

```bash
# In the 1-GPU srun, detany3d env.
conda activate /storage3/3DOM/vshukla/envs/detany3d
cd /storage3/3DOM/vshukla/DetAny3D

# Training only.
rm -rf exps/wildbox_final_ft
torchrun --nproc_per_node=1 train.py \
    --config_path detect_anything/configs/wildbox/wildbox_final.yaml \
    --exp_dir exps/wildbox_final_ft 2>&1 | tee logs/final_ft_1gpu.log

# FT eval only.
FT_RUN_DIR=$(ls -dt exps/wildbox_final_ft/*/ | head -1)
FT_CKPT=$(ls -t "$FT_RUN_DIR"/checkpoint_*.pth | head -1)
sed -i.bak -e "s|\./checkpoints/detany3d_ckpts/detany3d\.pth|$FT_CKPT|g" \
    detect_anything/configs/wildbox/wildbox_eval_oracle2d.yaml
rm -rf exps/wildbox_final_ft_eval
torchrun --nproc_per_node=1 train.py \
    --config_path detect_anything/configs/wildbox/wildbox_eval_oracle2d.yaml \
    --exp_dir exps/wildbox_final_ft_eval 2>&1 | tee logs/final_ft_eval_1gpu.log
mv detect_anything/configs/wildbox/wildbox_eval_oracle2d.yaml.bak \
   detect_anything/configs/wildbox/wildbox_eval_oracle2d.yaml
```

Then export + score the FT row exactly like Phase A above, but with `FT_PRED_JSON` and `OV_FT_DIR` in place of the ZS variants.

---

## 5. Verify the output inventory

```bash
OV_ZS_DIR=$OVMONO3D_REPO/output/wildbox_detany3d_zeroshot_oracle
OV_FT_DIR=$OVMONO3D_REPO/output/wildbox_detany3d_finetuned_oracle

# Both rows: required artifacts must all exist.
for run_dir in "$OV_ZS_DIR" "$OV_FT_DIR"; do
    label=$(basename "$run_dir")
    echo "=== $label ==="
    for f in \
        inference/iter_final/WildBox_val/instances_predictions.pth \
        bev_ap.json \
        summary_nhd.txt \
        full_metrics/summary.json \
        full_metrics/log.2D.txt \
        full_metrics/log.3D.txt \
        full_metrics/log.3D-Rel.txt \
        vis_ovmono3d
    do
        if [[ -e "$run_dir/$f" ]]; then
            echo "  ok  $f"
        else
            echo "  MISSING  $f"
        fi
    done
done
```

If anything's missing, check the corresponding stage's log under `logs/`.

---

## 6. Read the headline numbers

```bash
echo "=== zero-shot ==="
python -m json.tool $OV_ZS_DIR/bev_ap.json | grep -A1 IoU
python -m json.tool $OV_ZS_DIR/full_metrics/summary.json | head -40

echo "=== fine-tuned ==="
python -m json.tool $OV_FT_DIR/bev_ap.json | grep -A1 IoU
python -m json.tool $OV_FT_DIR/full_metrics/summary.json | head -40

# Side-by-side (if make_report.py succeeded):
cat $OVMONO3D_REPO/output/paper_report_detany3d/report.md 2>/dev/null | head -60
```

Expected pattern (mirroring smoke, but with order-of-magnitude higher absolute values):
- Zero-shot 3D AP / BEV AP near zero across all 6 species.
- Fine-tuned 3D AP / BEV AP non-zero across all classes, with elephant strongest, Grévy's zebra weakest (sparse predictions due to GDino's 0.035× under-prediction bias documented in [ovmono3d §6.4.2](../ovmono3d/WILDBOX_EXPERIMENT.md)).

---

## 7. Drop a row into the paper table

For each architecture-row in the cross-arch comparison, paste:

| Field | DetAny3D zero-shot | DetAny3D fine-tuned |
|---|---|---|
| AP_BEV @ 0.50 (macro) | from `$OV_ZS_DIR/bev_ap.json` | from `$OV_FT_DIR/bev_ap.json` |
| AP_3D @ 0.25 | from `$OV_ZS_DIR/full_metrics/summary.json` `[3D][AP50]` | same field, FT |
| 2D AP @ 0.5:0.95 | `summary.json [2D][AP]` | same |
| Class-agnostic 2D AP @ 0.5 | `summary_nhd.txt` | same |
| NHD best global scale | `summary_nhd.txt "best global scale"` | same |
| Per-class AP (giraffe, grevys_zebra, elephant, plains_zebra, rhino, gazelle) | `summary.json [3D][per_class_AP]` | same |

ovmono3d's [§20.5 reporting table](../ovmono3d/WILDBOX_EXPERIMENT.md) shows where DetAny3D's two rows slot in alongside ovmono3d's three rows.

---

## 8. Tear down

When the srun shell exits, the GPU allocation is released. The conda env, checkpoints, pickles, and output directories all persist on `/storage*` for re-use.

To re-run with a different seed:
1. Add `SEED:` field to `wildbox_final.yaml` and pipe to `set_seed()` calls in `train.py` (currently hardcoded; minor patch).
2. Re-run from §3 with a different `OUTPUT_DIR`.

---

## Troubleshooting

In order most likely to bite:

1. **`Killed` on `python -c "import torch; ..."`** — SLURM CPU-memory cap. Re-`srun` with `--mem=200G`.
2. **Multi-GPU `[3/7]` hangs at NCCL Init COMPLETE with 100% GPU util but no iter** — bug #10, see [WILDBOX_DETANY3D.md §3.5](WILDBOX_DETANY3D.md). **Use 1 GPU.** The 4h timeout patch in `db7bb37` only buys you a longer wait, not a fix.
3. **Stage `[3/7]` finishes in seconds with no `iter:` lines** — silent-skip bug #7, see [WILDBOX_DETANY3D.md §3.4](WILDBOX_DETANY3D.md). Verify `train.py:506-512` has the `is_continuation` guard.
4. **Final eval reports the same numbers as smoke (n_preds=3167, etc.)** — bug #11, see [WILDBOX_DETANY3D.md §3.6](WILDBOX_DETANY3D.md). The eval config still has `interval: 20` from a smoke debug session; re-run step §3(b).
5. **NaN training loss** — AMP instability. Set `use_amp: False` in `wildbox_final.yaml` (~1.5× slower but stable).
6. **`RuntimeError: ms_deform_attn_impl_forward: implementation for device cuda:0 not found`** — mmcv from pip lacks CUDA ops. Reinstall via OpenMMLab wheel index ([WILDBOX_DETANY3D.md §3.3](WILDBOX_DETANY3D.md)).
7. **Per-class AP all 0 in `full_metrics`** — category-ID remap bug; `git pull` to ensure `1b0ab2a` is present.
8. **`KeyError: 'bbox3D'` or `'depth'` from the evaluator** — GT-side aliasing wasn't applied; `git pull` to ensure `0eca64c` and `8c75089` are present.
9. **`libGL.so.1: cannot open shared object file`** — `opencv-python` was reinstalled by some transitive dep. `pip uninstall -y opencv-python && pip install --force-reinstall --no-deps opencv-python-headless==4.10.0.84`.
10. **`RuntimeError: element 0 of tensors does not require grad`** mid-training — empty `prepare_for_dsam` bug; ensure `6c89796` is present.
11. **Disk space** — pickle outputs ~3-5 GB total, exp dirs ~10 GB per run, vis dirs ~50 MB each. Check `df -h /storage3` before launching.

For deeper debugging see [WILDBOX_DETANY3D.md §3](WILDBOX_DETANY3D.md) (env hazards) and [§8](WILDBOX_DETANY3D.md) (full bug-fix table with commit refs).
