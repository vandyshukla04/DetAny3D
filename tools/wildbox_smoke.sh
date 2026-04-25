#!/usr/bin/env bash
# Wildbox smoke pipeline for DetAny3D -- end-to-end pipeline validation.
#
# Mirrors ovmono3d/QUICK_START_SMOKE_TEST.md but on the DetAny3D
# architecture. Numbers will be low; the goal is to prove every step
# (data load -> train -> infer -> export -> ovmono3d eval) runs green.
#
# Required env vars:
#   WILDBOX_TRAIN_JSON  - ovmono3d's smoke WildBox_train.json
#                         (e.g. /storage2/.../ovmono3d/output/smoke/WildBox_train.json)
#   WILDBOX_VAL_JSON    - ovmono3d's smoke WildBox_val.json
#   GDINO_ORACLE_JSON   - ovmono3d's gdino_WildBox_val_oracle_2d.json
#                         (the SAME file ovmono3d's row-2 eval consumes;
#                         §20.4 portable contract)
#   OVMONO3D_REPO       - clone path of ovmono3d (for the eval stack)
#
# Optional:
#   NUM_GPUS            - default 1
#   DETANY3D_REPO       - default $(pwd)
#   PATH_REMAP          - e.g. "/old/prefix=/new/prefix" if image absolute
#                         paths in the WildBox JSONs need rewriting

set -eo pipefail

: "${WILDBOX_TRAIN_JSON:?required: ovmono3d smoke WildBox_train.json path}"
: "${WILDBOX_VAL_JSON:?required: ovmono3d smoke WildBox_val.json path}"
: "${GDINO_ORACLE_JSON:?required: ovmono3d gdino_WildBox_val_oracle_2d.json path}"
: "${OVMONO3D_REPO:?required: ovmono3d repo clone path (for eval stack)}"

NUM_GPUS=${NUM_GPUS:-1}
DETANY3D_REPO=${DETANY3D_REPO:-$(pwd)}
PATH_REMAP=${PATH_REMAP:-}

cd "$DETANY3D_REPO"
mkdir -p data/pkls/wildbox datasets/Omni3D logs

echo "============================================================"
echo "[1/6] Convert WildBox JSONs -> DetAny3D pickles"
echo "============================================================"

REMAP_ARG=""
if [[ -n "$PATH_REMAP" ]]; then
    REMAP_ARG="--path-remap $PATH_REMAP"
fi

python -m detect_anything.datasets.data_creator.wildbox \
    --wildbox-json "$WILDBOX_TRAIN_JSON" \
    --category-meta data/category_meta_wildbox.json \
    --out-pkl data/pkls/wildbox/WildBox_smoke_train.pkl \
    $REMAP_ARG \
    --verify-projection

python -m detect_anything.datasets.data_creator.wildbox \
    --wildbox-json "$WILDBOX_VAL_JSON" \
    --category-meta data/category_meta_wildbox.json \
    --out-pkl data/pkls/wildbox/WildBox_smoke_val.pkl \
    $REMAP_ARG \
    --verify-projection

# Make the GDino oracle JSON discoverable at the path the config expects.
mkdir -p datasets/Omni3D
ln -sf "$(realpath "$GDINO_ORACLE_JSON")" datasets/Omni3D/gdino_WildBox_val_oracle_2d.json

# Sanity: pickle records non-empty.
python -c "
import pickle
for split in ['smoke_train', 'smoke_val']:
    p = pickle.load(open(f'data/pkls/wildbox/WildBox_{split}.pkl', 'rb'))
    n_obj = sum(len(r['obj_list']) for r in p)
    print(f'  {split}: {len(p)} images, {n_obj} objects')
    assert len(p) > 0, f'empty pickle: {split}'
"

echo ""
echo "============================================================"
echo "[2/6] Zero-shot oracle-2D eval (paper-protocol row 2)"
echo "============================================================"

# IMPORTANT: the oracle eval uses the val pickle. Override pkl_path via env
# isn't supported by Box configs, so we patch the smoke val path inside the
# eval config -- same val pickle is fine since we wrote it just above.
sed -i.bak \
    -e "s|\./data/pkls/wildbox/WildBox_val\.pkl|./data/pkls/wildbox/WildBox_smoke_val.pkl|g" \
    detect_anything/configs/wildbox/wildbox_eval_oracle2d.yaml

ZS_EXP_DIR=exps/wildbox_smoke_zeroshot_oracle
rm -rf "$ZS_EXP_DIR"
torchrun --nproc_per_node=$NUM_GPUS train.py \
    --config_path detect_anything/configs/wildbox/wildbox_eval_oracle2d.yaml \
    --exp_dir "$ZS_EXP_DIR" 2>&1 | tee logs/smoke_zeroshot_oracle.log

# DetAny3D adds a timestamp subdir; resolve it.
ZS_RUN_DIR=$(ls -dt "$ZS_EXP_DIR"/*/ | head -1)
ZS_PRED_JSON=$(ls "$ZS_RUN_DIR"/wildbox_*.json 2>/dev/null | head -1 || true)
if [[ -z "$ZS_PRED_JSON" ]]; then
    echo "ERROR: no prediction JSON in $ZS_RUN_DIR" >&2
    exit 2
fi
echo "zero-shot preds: $ZS_PRED_JSON"

echo ""
echo "============================================================"
echo "[3/6] Smoke fine-tune (~few minutes-hours depending on GPU)"
echo "============================================================"

# Restore original eval config (smoke fine-tune output uses WildBox_smoke_val).
mv detect_anything/configs/wildbox/wildbox_eval_oracle2d.yaml.bak \
   detect_anything/configs/wildbox/wildbox_eval_oracle2d.yaml

FT_EXP_DIR=exps/wildbox_smoke_ft
rm -rf "$FT_EXP_DIR"
torchrun --nproc_per_node=$NUM_GPUS train.py \
    --config_path detect_anything/configs/wildbox/wildbox_smoke.yaml \
    --exp_dir "$FT_EXP_DIR" 2>&1 | tee logs/smoke_ft.log

FT_RUN_DIR=$(ls -dt "$FT_EXP_DIR"/*/ | head -1)
FT_CKPT=$(ls -t "$FT_RUN_DIR"/checkpoint_*.pth 2>/dev/null | head -1 || true)
if [[ -z "$FT_CKPT" ]]; then
    echo "ERROR: no fine-tune checkpoint in $FT_RUN_DIR" >&2
    exit 3
fi
echo "fine-tune checkpoint: $FT_CKPT"

# Bug-hunt analog of ovmono3d §3.1.1: confirm training actually iterated.
N_ITER_LINES=$(grep -cE "iter[er]?[: ][0-9]+" "$FT_RUN_DIR"/log.txt 2>/dev/null || echo 0)
echo "iter log lines: $N_ITER_LINES"
if (( N_ITER_LINES < 5 )); then
    echo "WARNING: very few iter lines logged. Check $FT_RUN_DIR/log.txt for silent-skip."
fi

echo ""
echo "============================================================"
echo "[4/6] Fine-tuned oracle-2D eval (paper-protocol row 3)"
echo "============================================================"

sed -i.bak \
    -e "s|\./data/pkls/wildbox/WildBox_val\.pkl|./data/pkls/wildbox/WildBox_smoke_val.pkl|g" \
    -e "s|\./checkpoints/detany3d_ckpts/detany3d\.pth|$FT_CKPT|g" \
    detect_anything/configs/wildbox/wildbox_eval_oracle2d.yaml

FT_EVAL_EXP_DIR=exps/wildbox_smoke_ft_eval
rm -rf "$FT_EVAL_EXP_DIR"
torchrun --nproc_per_node=$NUM_GPUS train.py \
    --config_path detect_anything/configs/wildbox/wildbox_eval_oracle2d.yaml \
    --exp_dir "$FT_EVAL_EXP_DIR" 2>&1 | tee logs/smoke_ft_eval.log

mv detect_anything/configs/wildbox/wildbox_eval_oracle2d.yaml.bak \
   detect_anything/configs/wildbox/wildbox_eval_oracle2d.yaml

FT_EVAL_RUN_DIR=$(ls -dt "$FT_EVAL_EXP_DIR"/*/ | head -1)
FT_PRED_JSON=$(ls "$FT_EVAL_RUN_DIR"/wildbox_*.json 2>/dev/null | head -1 || true)
if [[ -z "$FT_PRED_JSON" ]]; then
    echo "ERROR: no prediction JSON in $FT_EVAL_RUN_DIR" >&2
    exit 4
fi

echo ""
echo "============================================================"
echo "[5/6] Export both runs to ovmono3d instances_predictions.pth"
echo "============================================================"

OV_ZS_DIR="$OVMONO3D_REPO/output/smoke_detany3d_zeroshot_oracle"
OV_FT_DIR="$OVMONO3D_REPO/output/smoke_detany3d_finetuned_oracle"
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

echo ""
echo "============================================================"
echo "[6/6] Run ovmono3d eval stack on both prediction files"
echo "============================================================"

cd "$OVMONO3D_REPO"

for run_dir in "$OV_ZS_DIR" "$OV_FT_DIR"; do
    label=$(basename "$run_dir")
    echo "--- $label ---"
    python tools/bev_ap_eval.py \
        --gt "$WILDBOX_VAL_JSON" \
        --predictions "$run_dir/inference/iter_final/WildBox_val/instances_predictions.pth" \
        --out "$run_dir/bev_ap.json" \
        --classes giraffe grevys_zebra elephant plains_zebra rhino gazelle \
        || echo "WARNING: bev_ap_eval failed for $label"

    python tools/class_agnostic_eval.py \
        --gt "$WILDBOX_VAL_JSON" \
        --predictions "$run_dir/inference/iter_final/WildBox_val/instances_predictions.pth" \
        --out "$run_dir/summary_nhd.txt" \
        --classes giraffe grevys_zebra elephant plains_zebra rhino gazelle \
        --nhd \
        || echo "WARNING: class_agnostic_eval failed for $label"
done

echo ""
echo "============================================================"
echo "SMOKE COMPLETE"
echo "  zero-shot oracle:   $OV_ZS_DIR"
echo "  fine-tuned oracle:  $OV_FT_DIR"
echo "Use ovmono3d/tools/make_report.py --compare for the side-by-side."
echo "============================================================"
