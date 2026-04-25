# DetAny3D × WildBox — quick-start smoke test (~1.5 h)

**Purpose**: prove the WildBox pipeline works end-to-end on the DetAny3D
side before committing to the multi-hour final run. Catches breakage
after pulling new code, after re-installing the env, or after porting
to a new cluster.

**Pipeline this validates**:
data conversion → oracle-2D loader → zero-shot eval JSON write → fine-tune
training (multi-iter) → fine-tuned eval JSON write → export to ovmono3d
format → BEV AP + class-agnostic + standard 3D AP via Omni3DEvaluator +
visualizations + side-by-side metrics file.

**Cluster**: `node81` or `node82` (gpu-A40 partition, A40), env at
`/storage3/3DOM/vshukla/envs/detany3d`.

The smoke uses a coarse subsample (1/50 train, 1/20 val) of the **full**
WildBox JSONs that ovmono3d's pipeline already produced. It is **not**
re-running ovmono3d's smoke prep — we read directly from
`$OVMONO3D_REPO/datasets/Omni3D/WildBox_{train,val}.json`. Numbers will
be low (~50× under-trained vs final); the goal is exercising every
plumbing step.

For dataset-level documentation see
[ovmono3d/WILDBOX_EXPERIMENT.md §2](../ovmono3d/WILDBOX_EXPERIMENT.md).
For the rationale behind any DetAny3D-side step see
[WILDBOX_DETANY3D.md](WILDBOX_DETANY3D.md).

---

## 0. Setup

```bash
# Frontend or any node with conda + git.
ssh frontendnew
cd /storage3/3DOM/vshukla/DetAny3D
git pull
conda activate /storage3/3DOM/vshukla/envs/detany3d

# Quick env health check.
python -c "
import torch, mmcv
from mmcv.ops.multi_scale_deform_attn import ext_module
import groundingdino
from detect_anything.datasets.detany3d_dataset import DetAny3DDataset
print('imports ok; mmcv ext fns:', [a for a in dir(ext_module) if 'deform' in a])
"
```

If any of those fail, jump to [WILDBOX_DETANY3D.md §3](WILDBOX_DETANY3D.md) (env hazards).

---

## 1. Locate ovmono3d artifacts

```bash
export OVMONO3D_REPO=/storage2/3DOM/vshukla/repos/ovmono3d
export WILDBOX_TRAIN_JSON=$OVMONO3D_REPO/datasets/Omni3D/WildBox_train.json
export WILDBOX_VAL_JSON=$OVMONO3D_REPO/datasets/Omni3D/WildBox_val.json
export GDINO_ORACLE_JSON=$OVMONO3D_REPO/datasets/Omni3D/gdino_WildBox_val_oracle_2d.json

ls -la "$WILDBOX_TRAIN_JSON" "$WILDBOX_VAL_JSON" "$GDINO_ORACLE_JSON"
```

All three must exist. If they don't, run ovmono3d's pipeline first ([ovmono3d/FINAL_RUN.md](../ovmono3d/FINAL_RUN.md) §2 + the GDino oracle precompute in §9).

---

## 2. Configure smoke subsampling

The smoke configs are committed with `interval: 1`. For quick validation, lower the val interval (eval is the expensive single-GPU step) and the train interval (so 3 epochs finishes in ~20 min).

```bash
cd /storage3/3DOM/vshukla/DetAny3D

# wildbox_smoke.yaml: train interval=50, val interval=20.
python <<'PY'
import re, yaml
p = 'detect_anything/configs/wildbox/wildbox_smoke.yaml'
s = open(p).read()
new_iv = iter([50, 20])
def sub(m):
    return m.group(1) + f'interval: {next(new_iv)}'
s = re.sub(r'(range:\s*\n\s+begin:\s*0\s*\n\s+end:\s*-1\s*\n\s+)interval:\s*1\b', sub, s, count=2)
open(p, 'w').write(s)
cfg = yaml.safe_load(open(p))
print("smoke train range:", cfg['dataset']['train']['wildbox']['range'])
print("smoke val   range:", cfg['dataset']['val']['wildbox']['range'])
PY

# wildbox_eval_oracle2d.yaml: val interval=20 (single-line range form).
python <<'PY'
import re, yaml
p = 'detect_anything/configs/wildbox/wildbox_eval_oracle2d.yaml'
s = open(p).read()
s = re.sub(r'(range:\s*\{[^}]*?)interval:\s*1', r'\1interval: 20', s)
open(p, 'w').write(s)
cfg = yaml.safe_load(open(p))
print("eval val range:", cfg['dataset']['val']['wildbox']['range'])
PY
```

Expected: `interval: 50` for smoke train, `interval: 20` for both smoke val and eval val. **Do not commit these edits** — for the final run we want `interval: 1`.

---

## 3. Claim a single A40 (smoke is fast enough)

```bash
srun --partition=gpu-A40 --gres=gpu:1 --mem=64G --cpus-per-task=8 \
     --time=04:00:00 --pty bash
```

Inside the srun:

```bash
conda activate /storage3/3DOM/vshukla/envs/detany3d
cd /storage3/3DOM/vshukla/DetAny3D
python -c "import torch; print('cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())"
```

Must print `cuda True gpus 1`. If `Killed` appears, raise `--mem=128G`.

---

## 4. Run the smoke orchestrator

```bash
export OVMONO3D_REPO=/storage2/3DOM/vshukla/repos/ovmono3d
export WILDBOX_TRAIN_JSON=$OVMONO3D_REPO/datasets/Omni3D/WildBox_train.json
export WILDBOX_VAL_JSON=$OVMONO3D_REPO/datasets/Omni3D/WildBox_val.json
export GDINO_ORACLE_JSON=$OVMONO3D_REPO/datasets/Omni3D/gdino_WildBox_val_oracle_2d.json
export NUM_GPUS=1

mkdir -p logs
bash tools/wildbox_smoke.sh 2>&1 | tee logs/smoke_run.log
```

Six stages — each prints its own banner. Expected timings:

| Stage | What | Time |
|---|---|---|
| `[1/6]` | Convert WildBox JSONs → DetAny3D pickles | ~10 sec |
| `[2/6]` | Zero-shot oracle eval (689 val samples at interval=20) | ~9 min |
| `[3/6]` | Smoke fine-tune (920 train samples × 3 epochs) | ~22 min |
| `[4/6]` | Fine-tuned oracle eval (689 val samples) | ~9 min |
| `[5/6]` | Export both runs to ovmono3d format | ~30 sec |
| `[6/6]` | BEV AP + class-agnostic + standard AP_3D + visualizations | ~10 min |

**Total: ~50 min**. Plus ~10 min if running the full eval suite (Tests 1–3 in §5 below).

---

## 5. Sanity-check the smoke outputs

```bash
OV_ZS_DIR=$OVMONO3D_REPO/output/smoke_detany3d_zeroshot_oracle
OV_FT_DIR=$OVMONO3D_REPO/output/smoke_detany3d_finetuned_oracle
GT=$WILDBOX_VAL_JSON
DA3D_REPO=/storage3/3DOM/vshukla/DetAny3D
```

### 5.1 BEV AP (already produced by stage [6/6])

```bash
echo "=== zero-shot BEV AP ==="
python -m json.tool $OV_ZS_DIR/bev_ap.json | head -30
echo "=== fine-tuned BEV AP ==="
python -m json.tool $OV_FT_DIR/bev_ap.json | head -30
```

Expected pattern at smoke scale (1/50 train, 3 epochs, ~50× under-trained vs final):
- Zero-shot AP_BEV macro@0.25: 0.00
- Fine-tuned AP_BEV macro@0.25: ~2–4%
- Per-class: elephant strongest, Grévy's zebra weakest (sparse preds — GDino's known bias).

### 5.2 Class-agnostic + NHD

If stage [6/6] of `wildbox_smoke.sh` ran the class-agnostic step it'll already exist. Otherwise run manually (in ovmono3d's env for shapely 2.x):

```bash
conda deactivate
conda activate /storage3/3DOM/vshukla/envs/ovmono3d
cd /storage2/3DOM/vshukla/repos/ovmono3d
for run_dir in "$OV_ZS_DIR" "$OV_FT_DIR"; do
    label=$(basename "$run_dir")
    echo "=== class_agnostic: $label ==="
    python tools/class_agnostic_eval.py --gt "$GT" \
        --preds "$run_dir/inference/iter_final/WildBox_val/instances_predictions.pth" \
        --nhd > "$run_dir/summary_nhd.txt" 2>&1
    grep -E "AP@|NHD|best global scale" "$run_dir/summary_nhd.txt" | head -15
    echo
done
```

The headline signal of the dataset paper:
- Zero-shot **best global scale ~0.10** (severe scale mismatch — pretrained 3D priors say "metric scale" but WildBox is synthetic)
- Fine-tuned **best global scale ~0.94** (recovered to near-unit, matching WildBox's median |Z|=1)

### 5.3 Standard AP_3D + 2D AP via Omni3DEvaluator

```bash
for run_dir in "$OV_ZS_DIR" "$OV_FT_DIR"; do
    label=$(basename "$run_dir")
    echo "=== full_metrics: $label ==="
    python "$DA3D_REPO/tools/wildbox_full_metrics.py" \
        --predictions "$run_dir/inference/iter_final/WildBox_val/instances_predictions.pth" \
        --gt "$GT" --ovmono3d-repo /storage2/3DOM/vshukla/repos/ovmono3d \
        --out-dir "$run_dir/full_metrics" 2>&1 | tail -10
done

echo "=== zero-shot summary ==="; python -m json.tool $OV_ZS_DIR/full_metrics/summary.json | head -40
echo "=== fine-tuned summary ==="; python -m json.tool $OV_FT_DIR/full_metrics/summary.json | head -40
```

Expected:
- Zero-shot 3D AP: 0.000 across all classes (3D coordinate scale is wildly off)
- Fine-tuned 3D AP: small but **non-zero** across most classes, especially elephant
- Both rows show identical 2D AP (~0.028) — DetAny3D doesn't predict 2D, it lifts oracle 2D inputs unchanged. This is correct ([WILDBOX_DETANY3D.md §6.4](WILDBOX_DETANY3D.md)).

### 5.4 Visualizations

Stage [6/6] of the orchestrator runs `visualize_class_agnostic.py` automatically. Verify:

```bash
for run_dir in "$OV_ZS_DIR" "$OV_FT_DIR"; do
    label=$(basename "$run_dir")
    n=$(ls "$run_dir/vis_ovmono3d" 2>/dev/null | wc -l)
    echo "$label: $n viz files"
done
```

Expected: ~30 `img_*.jpg` files per row, each a 2×3 layout (gt-only / pred-only / combined, with and without ground grid).

---

## 6. What this smoke proves on a green run

Tick boxes. If anything is missing, the smoke FAILS — debug before launching final.

- [ ] WildBox JSON → DetAny3D pickle conversion (with `--verify-projection` 100% OK on both splits — convention swap is correct)
- [ ] Oracle-2D loader hook activates (`loaded oracle-2D entries for 13779 images` line in eval log)
- [ ] Zero-shot eval JSON written under `exps/wildbox_smoke_zeroshot_oracle/<ts>/`
- [ ] Fine-tune actually iterates (>50 `iter:` lines in `exps/wildbox_smoke_ft/<ts>/log.txt` — silent-skip canary)
- [ ] Fine-tuned eval JSON written
- [ ] Export to `instances_predictions.pth` succeeds (3-file existence under `$OV_*_DIR/inference/iter_final/WildBox_val/`)
- [ ] BEV AP shows zero-shot=0 / fine-tuned=non-zero macro@0.25
- [ ] Class-agnostic shows fine-tuned best-scale recovered toward 1.0 (was ~0.10 zero-shot)
- [ ] Standard AP_3D shows fine-tuned non-zero 3D AP across classes
- [ ] ovmono3d-style 2×3 visualizations produced

---

## 7. If anything fails

In order most likely to be wrong (mirrors [WILDBOX_DETANY3D.md §8](WILDBOX_DETANY3D.md)):

1. **Stage `[3/6]` finishes immediately, no `iter:` lines logged** — silent-skip bug. `grep -A2 is_continuation train.py` to confirm the patch is applied. `git pull` to be safe.
2. **`KeyError: 'bbox3D'` or `'depth'` in stage `[6/6]`** — GT-side aliasing missing. `git pull` to ensure `0eca64c` and `8c75089` are at branch tip.
3. **All-zero per-class AP from stage `[6/6]`** — category-ID remap bug. `git pull` to ensure `1b0ab2a` is present.
4. **`RuntimeError: ms_deform_attn_impl_forward: ... not found`** — mmcv lacks CUDA ops. `pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html`.
5. **`libGL.so.1` import error** — `pip install --force-reinstall --no-deps opencv-python-headless==4.10.0.84`.
6. **`Killed` on torch import inside srun** — raise `--mem=128G`.
7. **`RuntimeError: element 0 of tensors does not require grad`** mid-training — empty-`prepare_for_dsam` bug. `git pull` to ensure the recursion fix at [detany3d_dataset.py:236](detect_anything/datasets/detany3d_dataset.py#L236) is present.
8. **NaN training loss** — AMP edge case. Set `use_amp: False` in `wildbox_smoke.yaml`.

For deeper debugging see [WILDBOX_DETANY3D.md §3 / §8](WILDBOX_DETANY3D.md).

Once green, scale up to the full run via [FINAL_RUN_DETANY3D.md](FINAL_RUN_DETANY3D.md).
