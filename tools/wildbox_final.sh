#!/usr/bin/env bash
# Wildbox final pipeline for DetAny3D -- single-seed paper run.
#
# Multi-GPU DDP for fine-tuning (configurable NUM_GPUS); single-GPU for
# eval rows (eval at batch_size=1 doesn't benefit much from DDP).
#
# Required env vars:
#   WILDBOX_TRAIN_JSON  - ovmono3d's full WildBox_train.json
#   WILDBOX_VAL_JSON    - ovmono3d's full WildBox_val.json
#   GDINO_ORACLE_JSON   - ovmono3d's gdino_WildBox_val_oracle_2d.json
#                         (the SAME file every architecture in the
#                         cross-arch comparison consumes; §20.4 contract)
#   OVMONO3D_REPO       - path to the ovmono3d clone (for eval stack +
#                         output-dir convention)
#
# Optional:
#   NUM_GPUS              - default 4 (set higher if more A40s are free)
#   DETANY3D_REPO         - default $(pwd)
#   PATH_REMAP            - "/old=/new" prefix rewrite for image paths
#   OVMONO3D_ENV_PREFIX   - conda env path with shapely 2.x + pytorch3d-CPU
#                           (REQUIRED — no default)
#   DETANY3D_ENV_PREFIX   - conda env path used for inference + training
#                           (REQUIRED — no default)

set -eo pipefail

: "${WILDBOX_TRAIN_JSON:?required}"
: "${WILDBOX_VAL_JSON:?required}"
: "${GDINO_ORACLE_JSON:?required}"
: "${OVMONO3D_REPO:?required}"

NUM_GPUS=${NUM_GPUS:-4}
DETANY3D_REPO=${DETANY3D_REPO:-$(pwd)}
PATH_REMAP=${PATH_REMAP:-}
: "${OVMONO3D_ENV_PREFIX:?required — set to your local OVMono3D conda env prefix}"
: "${DETANY3D_ENV_PREFIX:?required — set to your local DetAny3D  conda env prefix}"

# Source conda so `conda activate` works inside this non-interactive shell.
CONDA_BASE=$(conda info --base 2>/dev/null || echo /opt/miniforge3)
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

cd "$DETANY3D_REPO"
mkdir -p data/pkls/wildbox datasets/Omni3D logs

run_da3d() { conda activate "$DETANY3D_ENV_PREFIX"; }
run_ov()   { conda activate "$OVMONO3D_ENV_PREFIX"; }

# --------------------------------------------------------------------
echo "============================================================"
echo "[1/7] Convert WildBox JSONs -> DetAny3D pickles"
echo "============================================================"
run_da3d
REMAP_ARG=""; [[ -n "$PATH_REMAP" ]] && REMAP_ARG="--path-remap $PATH_REMAP"

python -m detect_anything.datasets.data_creator.wildbox \
    --wildbox-json "$WILDBOX_TRAIN_JSON" \
    --category-meta data/category_meta_wildbox.json \
    --out-pkl data/pkls/wildbox/WildBox_train.pkl \
    $REMAP_ARG --verify-projection

python -m detect_anything.datasets.data_creator.wildbox \
    --wildbox-json "$WILDBOX_VAL_JSON" \
    --category-meta data/category_meta_wildbox.json \
    --out-pkl data/pkls/wildbox/WildBox_val.pkl \
    $REMAP_ARG --verify-projection

ln -sf "$(realpath "$GDINO_ORACLE_JSON")" datasets/Omni3D/gdino_WildBox_val_oracle_2d.json
python -c "
import pickle
for split in ['train', 'val']:
    p = pickle.load(open(f'data/pkls/wildbox/WildBox_{split}.pkl', 'rb'))
    n_obj = sum(len(r['obj_list']) for r in p)
    print(f'  {split}: {len(p)} images, {n_obj} objects')
    assert len(p) > 0
"

# --------------------------------------------------------------------
echo "============================================================"
echo "[2/7] Zero-shot oracle-2D eval (paper-protocol row 2)"
echo "============================================================"
ZS_EXP_DIR=exps/wildbox_final_zeroshot_oracle
rm -rf "$ZS_EXP_DIR"
torchrun --nproc_per_node=1 train.py \
    --config_path detect_anything/configs/wildbox/wildbox_eval_oracle2d.yaml \
    --exp_dir "$ZS_EXP_DIR" 2>&1 | tee logs/final_zeroshot_oracle.log

ZS_RUN_DIR=$(ls -dt "$ZS_EXP_DIR"/*/ | head -1)
ZS_PRED_JSON=$(ls "$ZS_RUN_DIR"/wildbox_*.json | head -1)

# --------------------------------------------------------------------
echo "============================================================"
echo "[3/7] Final fine-tune (multi-GPU DDP, NUM_GPUS=$NUM_GPUS)"
echo "============================================================"
FT_EXP_DIR=exps/wildbox_final_ft
rm -rf "$FT_EXP_DIR"
torchrun --nproc_per_node=$NUM_GPUS train.py \
    --config_path detect_anything/configs/wildbox/wildbox_final.yaml \
    --exp_dir "$FT_EXP_DIR" 2>&1 | tee logs/final_ft.log

FT_RUN_DIR=$(ls -dt "$FT_EXP_DIR"/*/ | head -1)
FT_CKPT=$(ls -t "$FT_RUN_DIR"/checkpoint_*.pth | head -1)

# Verify training actually iterated (silent-skip canary, ovmono3d §3.1.1).
N_ITER_LINES=$(grep -cE "iter[er]?[: ][0-9]+" "$FT_RUN_DIR"/log.txt 2>/dev/null || echo 0)
if (( N_ITER_LINES < 50 )); then
    echo "ERROR: only $N_ITER_LINES iter log lines -- training may have silently skipped" >&2
    exit 3
fi
echo "fine-tune iter log lines: $N_ITER_LINES (OK)"

# --------------------------------------------------------------------
echo "============================================================"
echo "[4/7] Fine-tuned oracle-2D eval (paper-protocol row 3)"
echo "============================================================"
sed -i.bak \
    -e "s|\./checkpoints/detany3d_ckpts/detany3d\.pth|$FT_CKPT|g" \
    detect_anything/configs/wildbox/wildbox_eval_oracle2d.yaml

FT_EVAL_EXP_DIR=exps/wildbox_final_ft_eval
rm -rf "$FT_EVAL_EXP_DIR"
torchrun --nproc_per_node=1 train.py \
    --config_path detect_anything/configs/wildbox/wildbox_eval_oracle2d.yaml \
    --exp_dir "$FT_EVAL_EXP_DIR" 2>&1 | tee logs/final_ft_eval.log

mv detect_anything/configs/wildbox/wildbox_eval_oracle2d.yaml.bak \
   detect_anything/configs/wildbox/wildbox_eval_oracle2d.yaml

FT_EVAL_RUN_DIR=$(ls -dt "$FT_EVAL_EXP_DIR"/*/ | head -1)
FT_PRED_JSON=$(ls "$FT_EVAL_RUN_DIR"/wildbox_*.json | head -1)

# --------------------------------------------------------------------
echo "============================================================"
echo "[5/7] Export both runs to ovmono3d instances_predictions.pth"
echo "============================================================"
OV_ZS_DIR="$OVMONO3D_REPO/output/wildbox_detany3d_zeroshot_oracle"
OV_FT_DIR="$OVMONO3D_REPO/output/wildbox_detany3d_finetuned_oracle"
mkdir -p "$OV_ZS_DIR/inference/iter_final/WildBox_val" \
         "$OV_FT_DIR/inference/iter_final/WildBox_val"

python tools/wildbox_export_predictions.py \
    --da3d-json "$ZS_PRED_JSON" --gt-json "$WILDBOX_VAL_JSON" \
    --out-pth "$OV_ZS_DIR/inference/iter_final/WildBox_val/instances_predictions.pth"

python tools/wildbox_export_predictions.py \
    --da3d-json "$FT_PRED_JSON" --gt-json "$WILDBOX_VAL_JSON" \
    --out-pth "$OV_FT_DIR/inference/iter_final/WildBox_val/instances_predictions.pth"

# --------------------------------------------------------------------
echo "============================================================"
echo "[6/7] Full eval suite on both rows (BEV + class_agnostic +"
echo "      standard AP_3D/2D AP via Omni3DEvaluator + Rel-AP_3D)"
echo "============================================================"
run_ov

for run_dir in "$OV_ZS_DIR" "$OV_FT_DIR"; do
    label=$(basename "$run_dir")
    PREDS="$run_dir/inference/iter_final/WildBox_val/instances_predictions.pth"

    echo "--- BEV AP: $label ---"
    python "$OVMONO3D_REPO/tools/bev_ap_eval.py" \
        --gt "$WILDBOX_VAL_JSON" --preds "$PREDS" --out "$run_dir/bev_ap.json"

    echo "--- class_agnostic + NHD: $label ---"
    python "$OVMONO3D_REPO/tools/class_agnostic_eval.py" \
        --gt "$WILDBOX_VAL_JSON" --preds "$PREDS" --nhd \
        > "$run_dir/summary_nhd.txt" 2>&1
    grep -E "AP@|NHD" "$run_dir/summary_nhd.txt" | head -20

    echo "--- standard Omni3D AP_3D + 2D AP + Rel-AP_3D: $label ---"
    python "$DETANY3D_REPO/tools/wildbox_full_metrics.py" \
        --predictions "$PREDS" \
        --gt "$WILDBOX_VAL_JSON" \
        --ovmono3d-repo "$OVMONO3D_REPO" \
        --out-dir "$run_dir/full_metrics" \
        --eval-rel-ap3d \
        --rel-ap3d-search "0.05,3.0,32" 2>&1 | tail -10
done

# --------------------------------------------------------------------
echo "============================================================"
echo "[7/7] ovmono3d-style 2x3 visualizations on both rows"
echo "============================================================"
for run_dir in "$OV_ZS_DIR" "$OV_FT_DIR"; do
    label=$(basename "$run_dir")
    PREDS="$run_dir/inference/iter_final/WildBox_val/instances_predictions.pth"
    echo "--- visualize: $label ---"
    python "$OVMONO3D_REPO/tools/visualize_class_agnostic.py" \
        --preds "$PREDS" --gt "$WILDBOX_VAL_JSON" \
        --out "$run_dir/vis_ovmono3d" \
        --top-k 5 --every 100 --limit 40 2>&1 | tail -5
done

# Optional: side-by-side report (skip if make_report.py needs additional setup).
if [[ -f "$OVMONO3D_REPO/tools/make_report.py" ]]; then
    python "$OVMONO3D_REPO/tools/make_report.py" --compare \
        --run-dir "$OV_ZS_DIR" --label "DetAny3D zero-shot (oracle 2D)" \
        --run-dir "$OV_FT_DIR" --label "DetAny3D fine-tuned (oracle 2D)" \
        --gt "$WILDBOX_VAL_JSON" \
        --out "$OVMONO3D_REPO/output/paper_report_detany3d" \
        2>&1 | tail -15 || echo "make_report.py errored; check eval outputs manually."
fi

echo ""
echo "============================================================"
echo "FINAL DONE. Outputs:"
echo "  zero-shot:        $OV_ZS_DIR"
echo "  fine-tuned:       $OV_FT_DIR"
echo "  side-by-side:     $OVMONO3D_REPO/output/paper_report_detany3d/report.md (if make_report.py ran)"
echo "============================================================"
