#!/usr/bin/env bash
# Wildbox final pipeline for DetAny3D -- single-seed paper run.
#
# Mirrors ovmono3d/FINAL_RUN.md (15-zip, 6-species, single-seed for now).
# Multi-seed is deferred per current scope.
#
# Required env vars (same as smoke):
#   WILDBOX_TRAIN_JSON  - ovmono3d's full WildBox_train.json
#                         (e.g. /storage2/.../ovmono3d/datasets/Omni3D/WildBox_train.json)
#   WILDBOX_VAL_JSON    - ovmono3d's full WildBox_val.json
#   GDINO_ORACLE_JSON   - ovmono3d's gdino_WildBox_val_oracle_2d.json (portable contract, §20.4)
#   OVMONO3D_REPO       - clone path of ovmono3d
#
# Optional:
#   NUM_GPUS            - default 1
#   DETANY3D_REPO       - default $(pwd)
#   PATH_REMAP          - e.g. "/old/prefix=/new/prefix"

set -eo pipefail

: "${WILDBOX_TRAIN_JSON:?required: ovmono3d WildBox_train.json}"
: "${WILDBOX_VAL_JSON:?required: ovmono3d WildBox_val.json}"
: "${GDINO_ORACLE_JSON:?required: ovmono3d gdino_WildBox_val_oracle_2d.json}"
: "${OVMONO3D_REPO:?required: ovmono3d repo clone path}"

NUM_GPUS=${NUM_GPUS:-1}
DETANY3D_REPO=${DETANY3D_REPO:-$(pwd)}
PATH_REMAP=${PATH_REMAP:-}

cd "$DETANY3D_REPO"
mkdir -p data/pkls/wildbox datasets/Omni3D logs

echo "[1/6] Convert WildBox JSONs -> DetAny3D pickles (15-zip, 6-species)"

REMAP_ARG=""
if [[ -n "$PATH_REMAP" ]]; then
    REMAP_ARG="--path-remap $PATH_REMAP"
fi

python -m detect_anything.datasets.data_creator.wildbox \
    --wildbox-json "$WILDBOX_TRAIN_JSON" \
    --category-meta data/category_meta_wildbox.json \
    --out-pkl data/pkls/wildbox/WildBox_train.pkl \
    $REMAP_ARG \
    --verify-projection

python -m detect_anything.datasets.data_creator.wildbox \
    --wildbox-json "$WILDBOX_VAL_JSON" \
    --category-meta data/category_meta_wildbox.json \
    --out-pkl data/pkls/wildbox/WildBox_val.pkl \
    $REMAP_ARG \
    --verify-projection

ln -sf "$(realpath "$GDINO_ORACLE_JSON")" datasets/Omni3D/gdino_WildBox_val_oracle_2d.json

python -c "
import pickle
for split in ['train', 'val']:
    p = pickle.load(open(f'data/pkls/wildbox/WildBox_{split}.pkl', 'rb'))
    n_obj = sum(len(r['obj_list']) for r in p)
    print(f'  {split}: {len(p)} images, {n_obj} objects')
    assert len(p) > 0, f'empty pickle: {split}'
"

echo "[2/6] Zero-shot oracle-2D eval (paper-protocol row 2)"
ZS_EXP_DIR=exps/wildbox_final_zeroshot_oracle
rm -rf "$ZS_EXP_DIR"
torchrun --nproc_per_node=$NUM_GPUS train.py \
    --config_path detect_anything/configs/wildbox/wildbox_eval_oracle2d.yaml \
    --exp_dir "$ZS_EXP_DIR" 2>&1 | tee logs/final_zeroshot_oracle.log

ZS_RUN_DIR=$(ls -dt "$ZS_EXP_DIR"/*/ | head -1)
ZS_PRED_JSON=$(ls "$ZS_RUN_DIR"/wildbox_*.json | head -1)

echo "[3/6] Final fine-tune (single seed)"
FT_EXP_DIR=exps/wildbox_final_ft
rm -rf "$FT_EXP_DIR"
torchrun --nproc_per_node=$NUM_GPUS train.py \
    --config_path detect_anything/configs/wildbox/wildbox_final.yaml \
    --exp_dir "$FT_EXP_DIR" 2>&1 | tee logs/final_ft.log

FT_RUN_DIR=$(ls -dt "$FT_EXP_DIR"/*/ | head -1)
FT_CKPT=$(ls -t "$FT_RUN_DIR"/checkpoint_*.pth | head -1)

# Verify training iterated (silent-skip canary, ovmono3d §3.1.1 analog).
N_ITER_LINES=$(grep -cE "iter[er]?[: ][0-9]+" "$FT_RUN_DIR"/log.txt 2>/dev/null || echo 0)
if (( N_ITER_LINES < 50 )); then
    echo "ERROR: only $N_ITER_LINES iter log lines -- training may have silently skipped." >&2
    exit 3
fi
echo "fine-tune iter log lines: $N_ITER_LINES (OK)"

echo "[4/6] Fine-tuned oracle-2D eval (paper-protocol row 3)"
sed -i.bak \
    -e "s|\./checkpoints/detany3d_ckpts/detany3d\.pth|$FT_CKPT|g" \
    detect_anything/configs/wildbox/wildbox_eval_oracle2d.yaml

FT_EVAL_EXP_DIR=exps/wildbox_final_ft_eval
rm -rf "$FT_EVAL_EXP_DIR"
torchrun --nproc_per_node=$NUM_GPUS train.py \
    --config_path detect_anything/configs/wildbox/wildbox_eval_oracle2d.yaml \
    --exp_dir "$FT_EVAL_EXP_DIR" 2>&1 | tee logs/final_ft_eval.log

mv detect_anything/configs/wildbox/wildbox_eval_oracle2d.yaml.bak \
   detect_anything/configs/wildbox/wildbox_eval_oracle2d.yaml

FT_EVAL_RUN_DIR=$(ls -dt "$FT_EVAL_EXP_DIR"/*/ | head -1)
FT_PRED_JSON=$(ls "$FT_EVAL_RUN_DIR"/wildbox_*.json | head -1)

echo "[5/6] Export both runs to ovmono3d instances_predictions.pth"
OV_ZS_DIR="$OVMONO3D_REPO/output/wildbox_detany3d_zeroshot_oracle"
OV_FT_DIR="$OVMONO3D_REPO/output/wildbox_detany3d_finetuned_oracle"
mkdir -p "$OV_ZS_DIR/inference/iter_final/WildBox_val" \
         "$OV_FT_DIR/inference/iter_final/WildBox_val"

python tools/wildbox_export_predictions.py \
    --da3d-json "$ZS_PRED_JSON" \
    --gt-json "$WILDBOX_VAL_JSON" \
    --out-pth "$OV_ZS_DIR/inference/iter_final/WildBox_val/instances_predictions.pth"

python tools/wildbox_export_predictions.py \
    --da3d-json "$FT_PRED_JSON" \
    --gt-json "$WILDBOX_VAL_JSON" \
    --out-pth "$OV_FT_DIR/inference/iter_final/WildBox_val/instances_predictions.pth"

echo "[6/6] Run ovmono3d eval stack + paper report"
cd "$OVMONO3D_REPO"

for run_dir in "$OV_ZS_DIR" "$OV_FT_DIR"; do
    label=$(basename "$run_dir")
    echo "--- $label ---"
    python tools/bev_ap_eval.py \
        --gt "$WILDBOX_VAL_JSON" \
        --predictions "$run_dir/inference/iter_final/WildBox_val/instances_predictions.pth" \
        --out "$run_dir/bev_ap.json" \
        --classes giraffe grevys_zebra elephant plains_zebra rhino gazelle

    python tools/class_agnostic_eval.py \
        --gt "$WILDBOX_VAL_JSON" \
        --predictions "$run_dir/inference/iter_final/WildBox_val/instances_predictions.pth" \
        --out "$run_dir/summary_nhd.txt" \
        --classes giraffe grevys_zebra elephant plains_zebra rhino gazelle \
        --nhd
done

# Side-by-side report (uses ovmono3d/tools/make_report.py if present).
if [[ -x "$OVMONO3D_REPO/tools/make_report.py" ]] || [[ -f "$OVMONO3D_REPO/tools/make_report.py" ]]; then
    python tools/make_report.py --compare \
        --run-dir "$OV_ZS_DIR" --label "DetAny3D zero-shot (oracle 2D)" \
        --run-dir "$OV_FT_DIR" --label "DetAny3D fine-tuned (oracle 2D)" \
        --gt "$WILDBOX_VAL_JSON" \
        --out "$OVMONO3D_REPO/output/paper_report_detany3d" \
        || echo "WARNING: make_report.py failed; check eval outputs manually."
fi

echo ""
echo "DONE."
echo "  zero-shot oracle:   $OV_ZS_DIR"
echo "  fine-tuned oracle:  $OV_FT_DIR"
echo "  side-by-side:       $OVMONO3D_REPO/output/paper_report_detany3d/report.md"
