# DetAny3D × WildBox — final paper run

Linear ops pipeline for the DetAny3D-side of the WildBox cross-architecture
comparison. Every step is a single copy-paste block. For the "why" behind any
command, see [WILDBOX_DETANY3D.md](WILDBOX_DETANY3D.md). For dataset-level
documentation, see [ovmono3d/WILDBOX_EXPERIMENT.md](../ovmono3d/WILDBOX_EXPERIMENT.md).

**Cluster:** `node81` or `node82` (gpu-A40 partition, 8× A40 each), env at
`/storage3/3DOM/vshukla/envs/detany3d`, repo at `/storage3/3DOM/vshukla/DetAny3D`.

**Run time budget (4× A40 DDP, single seed):**
- Data prep: ~2 min (cached after first run)
- Zero-shot oracle eval (full 13 779 val): ~3 h on 1 GPU
- Fine-tune (46k × 3 epochs): ~5 h on 4 GPUs
- Fine-tuned oracle eval: ~3 h on 1 GPU
- Full eval suite (BEV + class_agnostic + standard 3D AP + Rel-AP_3D): ~30 min
- Visualizations + report: ~10 min
- **Total: ~12 h** — fits in a single `--time=12:00:00` srun

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

## 2. Claim a multi-GPU srun on gpu-A40

```bash
# Check who's using node81/82's GPUs first.
sinfo -p gpu-A40 -o "%n %T %G %C %m"
ssh node81 "nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv" 2>/dev/null

# Allocate 4 GPUs + enough CPU memory for image decoding + multi-worker loaders.
srun --partition=gpu-A40 --gres=gpu:4 --mem=128G --cpus-per-task=16 \
     --time=12:00:00 --pty bash
```

Inside the srun shell:

```bash
conda activate /storage3/3DOM/vshukla/envs/detany3d
cd /storage3/3DOM/vshukla/DetAny3D
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv
python -c "import torch; print('cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())"
```

Must print `cuda True gpus 4`. If `Killed` appears, raise `--mem=256G`.

---

## 3. Run the orchestrator

```bash
export OVMONO3D_REPO=/storage2/3DOM/vshukla/repos/ovmono3d
export WILDBOX_TRAIN_JSON=$OVMONO3D_REPO/datasets/Omni3D/WildBox_train.json
export WILDBOX_VAL_JSON=$OVMONO3D_REPO/datasets/Omni3D/WildBox_val.json
export GDINO_ORACLE_JSON=$OVMONO3D_REPO/datasets/Omni3D/gdino_WildBox_val_oracle_2d.json
export NUM_GPUS=4

mkdir -p logs
bash tools/wildbox_final.sh 2>&1 | tee logs/final_run.log
```

The orchestrator runs seven stages — each prints its own banner. What to expect:

### 3.1 Stage `[1/7]` — Convert WildBox JSONs → DetAny3D pickles

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

### 3.2 Stage `[2/7]` — Zero-shot oracle eval

`exps/wildbox_final_zeroshot_oracle/<ts>/` will fill with predictions. Watch the tqdm bar; rate ~1.3 it/s on A40 = ~3h for full val.

### 3.3 Stage `[3/7]` — Fine-tune (multi-GPU)

```
[3/7] Final fine-tune (multi-GPU DDP, NUM_GPUS=4)
```

Watch for `iter:` lines. Throughput on 4× A40 ≈ 8 iter/s aggregate. 138k iter total → ~5 h. If `iter:` lines never appear and the run exits in seconds, **that's the silent-skip bug** — see [WILDBOX_DETANY3D.md §3.4](WILDBOX_DETANY3D.md). The orchestrator's `[3/7]` step exits non-zero if `< 50 iter:` lines are logged.

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

### 3.7 Stage `[7/7]` — Visualizations + side-by-side report

```
[7/7] ovmono3d-style 2x3 visualizations on both rows
```

Per row: `vis_ovmono3d/img_*.jpg` (paper-style 2×3 layout, gt-only / pred-only / combined, with and without ground grid). `--every 100 --limit 40` produces ~40 images per row.

Optional `make_report.py --compare` writes `$OVMONO3D_REPO/output/paper_report_detany3d/report.md` — side-by-side metrics table for both rows.

---

## 4. Verify the output inventory

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

## 5. Read the headline numbers

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

## 6. Drop a row into the paper table

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

## 7. Tear down

When the srun shell exits, the GPU allocation is released. The conda env, checkpoints, pickles, and output directories all persist on `/storage*` for re-use.

To re-run with a different seed:
1. Add `SEED:` field to `wildbox_final.yaml` and pipe to `set_seed()` calls in `train.py` (currently hardcoded; minor patch).
2. Re-run from §3 with a different `OUTPUT_DIR`.

---

## Troubleshooting

In order most likely to bite:

1. **`Killed` on `python -c "import torch; ..."`** — SLURM CPU-memory cap is too tight. Re-`srun` with `--mem=128G` (or higher).
2. **Stage `[3/7]` finishes in seconds with no `iter:` lines** — silent-skip bug, see [WILDBOX_DETANY3D.md §3.4](WILDBOX_DETANY3D.md). Verify `train.py:506-512` has the `is_continuation` guard. If it doesn't, `git pull` and confirm.
3. **NaN training loss** — AMP instability on edge images. Set `use_amp: False` in `wildbox_final.yaml` (~1.5× slower but stable).
4. **`RuntimeError: ms_deform_attn_impl_forward: implementation for device cuda:0 not found`** — mmcv from pip lacks CUDA ops. Reinstall via OpenMMLab wheel index (see [WILDBOX_DETANY3D.md §3.3](WILDBOX_DETANY3D.md)).
5. **Per-class AP all 0 in `full_metrics`** — category-ID remap bug; `git pull` to ensure `1b0ab2a` is present.
6. **`KeyError: 'bbox3D'` or `'depth'` from the evaluator** — GT-side aliasing wasn't applied; `git pull` to ensure `0eca64c` and `8c75089` are present.
7. **`libGL.so.1: cannot open shared object file`** — `opencv-python` was reinstalled by some transitive dep. `pip uninstall -y opencv-python && pip install --force-reinstall --no-deps opencv-python-headless==4.10.0.84`.
8. **Disk space** — pickle outputs are ~3-5 GB total, exp dirs ~10 GB per run, vis dirs ~50 MB each. Check `df -h /storage3` before launching.

For deeper debugging see [WILDBOX_DETANY3D.md §3](WILDBOX_DETANY3D.md) (env hazards) and [§8](WILDBOX_DETANY3D.md) (full bug-fix table with commit refs).
