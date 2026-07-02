#!/usr/bin/env bash
# Per-seed WildBox fine-tune + eval for DetAny3D's multi-seed variance study.
#
# Runs ONE seed end-to-end, SINGLE-GPU. Multi-GPU DDP hangs on this cluster
# (NCCL ALLGATHER SeqNum=1, bug #10 -- see FINAL_RUN_DETANY3D.md), which is
# exactly why seed1 was left pending; single-GPU is the fix.
#
# Designed to run as one sbatch array task per seed, so it performs NO shared
# writes that could race a sibling task:
#   * the fine-tuned checkpoint is handed to the eval via train.py's --resume
#     CLI override -- NOT a `sed` on the shared eval yaml (which the final.sh
#     single-seed path uses and which two concurrent tasks would clobber);
#   * every exp_dir / output path is namespaced by ${SEED}.
# Data-prep (pkls + oracle symlink) is a one-time shared step: run
# tools/wildbox_multiseed_prep.sh ONCE before the array (this script errors if
# the pkls are missing).
#
# Output (matches the existing seed0/seed2 dirs byte-for-byte in layout, so the
# local summarize_runs.py mean+-std aggregation picks up all 5 seeds unchanged):
#   $OVMONO3D_REPO/output/wildbox_detany3d_ft_ep2_seed${SEED}_int1_v3/
#     inference/iter_final/WildBox_val/instances_predictions.pth
#     bev_ap.json  summary_nhd.txt
#     full_metrics/{summary.json,log.2D.txt,log.3D.txt,log.3D-Rel.txt}
#
# Required env (defaults match FINAL_RUN_DETANY3D.md):
#   SEED                - integer seed (e.g. 1, 3, 4)
#   OVMONO3D_REPO       - ovmono3d clone (eval stack + output-dir convention)
#   WILDBOX_VAL_JSON    - ovmono3d's WildBox_val.json (GT for export + scoring)
#   DETANY3D_ENV_PREFIX - conda env prefix for train/eval/export (detany3d)
#   OVMONO3D_ENV_PREFIX - conda env prefix for scoring (shapely2 + pytorch3d-CPU)
# Optional:
#   DETANY3D_REPO       - default $(pwd)
#   CONFIG_FT           - default detect_anything/configs/wildbox/wildbox_final.yaml
#   CONFIG_EVAL         - default detect_anything/configs/wildbox/wildbox_eval_oracle2d.yaml
#   EXP_ROOT            - default exps/multiseed
#   SKIP_SCORE=1        - train+eval+export only; score later in the ovmono3d env
set -eo pipefail

: "${SEED:?required -- integer seed, e.g. SEED=1}"
OVMONO3D_REPO="${OVMONO3D_REPO:-/storage2/3DOM/vshukla/repos/ovmono3d}"
WILDBOX_VAL_JSON="${WILDBOX_VAL_JSON:-$OVMONO3D_REPO/datasets/Omni3D/WildBox_val.json}"
DETANY3D_REPO="${DETANY3D_REPO:-$(pwd)}"
CONFIG_FT="${CONFIG_FT:-detect_anything/configs/wildbox/wildbox_final.yaml}"
CONFIG_EVAL="${CONFIG_EVAL:-detect_anything/configs/wildbox/wildbox_eval_oracle2d.yaml}"
EXP_ROOT="${EXP_ROOT:-exps/multiseed}"
: "${DETANY3D_ENV_PREFIX:?required -- DetAny3D conda env prefix}"
: "${OVMONO3D_ENV_PREFIX:?required -- OVMono3D conda env prefix}"

CONDA_BASE=$(conda info --base 2>/dev/null || echo /opt/miniforge3)
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
run_da3d() { conda activate "$DETANY3D_ENV_PREFIX"; }
run_ov()   { conda activate "$OVMONO3D_ENV_PREFIX"; }

cd "$DETANY3D_REPO"
mkdir -p logs "$EXP_ROOT"

# -- Pre-flight ------------------------------------------------------------
[[ -f data/pkls/wildbox/WildBox_train.pkl && -f data/pkls/wildbox/WildBox_val.pkl ]] || {
    echo "ERROR: WildBox pkls missing. Run tools/wildbox_multiseed_prep.sh once first." >&2
    exit 1
}
[[ -e datasets/Omni3D/gdino_WildBox_val_oracle_2d.json ]] || {
    echo "ERROR: gdino oracle json symlink missing. Run tools/wildbox_multiseed_prep.sh." >&2
    exit 1
}
# Guard the eval stride: int1 (full 13,779-frame val) is what seed0/seed2 used.
# A leftover interval:4/20 from a smoke session would silently subsample and
# make this seed non-comparable (FINAL_RUN_DETANY3D.md bug #11).
run_da3d
python - "$CONFIG_EVAL" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
iv = cfg['dataset']['val']['wildbox']['range'].get('interval', 1)
assert iv == 1, f"eval val interval={iv}, expected 1 (int1 full-val). Fix {sys.argv[1]} before running."
print(f"[preflight] eval val interval={iv} (int1 full-val) OK")
PY

echo "============================================================"
echo "SEED $SEED  |  single-GPU  |  started $(date)"
echo "  FT config:   $CONFIG_FT   (num_epochs from config = ep2)"
echo "  eval config: $CONFIG_EVAL (--resume override, no yaml edit)"
echo "============================================================"

# -- [1/4] Fine-tune (single-GPU, seeded) ---------------------------------
FT_EXP_DIR="$EXP_ROOT/ft_seed${SEED}"
EXISTING_CKPT=$(ls -t "$FT_EXP_DIR"/*/checkpoint_*.pth 2>/dev/null | head -1 || true)
if [[ -n "$EXISTING_CKPT" ]]; then
    echo "[1/4] checkpoint exists ($EXISTING_CKPT) -- skipping training"
else
    echo "[1/4] fine-tune seed=$SEED -> $FT_EXP_DIR"
    PYTHONUNBUFFERED=1 torchrun --nproc_per_node=1 train.py \
        --config_path "$CONFIG_FT" \
        --seed "$SEED" \
        --exp_dir "$FT_EXP_DIR" 2>&1 | tee "logs/multiseed_ft_seed${SEED}.log"
fi

FT_RUN_DIR=$(ls -dt "$FT_EXP_DIR"/*/ | head -1)
FT_CKPT=$(ls -t "$FT_RUN_DIR"/checkpoint_*.pth | head -1)
[[ -f "$FT_CKPT" ]] || { echo "ERROR: no checkpoint under $FT_RUN_DIR" >&2; exit 2; }

# Silent-skip canary: training must have actually iterated (bug #7).
N_ITER=$(grep -cE "iter[er]?[: ][0-9]+" "$FT_RUN_DIR"/log.txt 2>/dev/null || echo 0)
if (( N_ITER < 50 )); then
    echo "ERROR: only $N_ITER iter log lines under $FT_RUN_DIR -- training silently skipped?" >&2
    exit 3
fi
echo "[1/4] done. checkpoint=$FT_CKPT  (iter log lines: $N_ITER)"

# -- [2/4] Fine-tuned oracle-2D eval (--resume override; parallel-safe) ----
FT_EVAL_EXP_DIR="$EXP_ROOT/ft_eval_seed${SEED}"
rm -rf "$FT_EVAL_EXP_DIR"
echo "[2/4] fine-tuned eval seed=$SEED -> $FT_EVAL_EXP_DIR"
PYTHONUNBUFFERED=1 torchrun --nproc_per_node=1 train.py \
    --config_path "$CONFIG_EVAL" \
    --resume "$FT_CKPT" \
    --exp_dir "$FT_EVAL_EXP_DIR" 2>&1 | tee "logs/multiseed_ft_eval_seed${SEED}.log"

FT_EVAL_RUN_DIR=$(ls -dt "$FT_EVAL_EXP_DIR"/*/ | head -1)
FT_PRED_JSON=$(ls "$FT_EVAL_RUN_DIR"/wildbox_*.json | head -1)
[[ -f "$FT_PRED_JSON" ]] || { echo "ERROR: no prediction json under $FT_EVAL_RUN_DIR" >&2; exit 4; }
echo "[2/4] done. predictions=$FT_PRED_JSON"

# -- [3/4] Export to ovmono3d instances_predictions.pth --------------------
OUT_DIR="$OVMONO3D_REPO/output/wildbox_detany3d_ft_ep2_seed${SEED}_int1_v3"
PREDS="$OUT_DIR/inference/iter_final/WildBox_val/instances_predictions.pth"
mkdir -p "$(dirname "$PREDS")"
echo "[3/4] export -> $PREDS"
python tools/wildbox_export_predictions.py \
    --da3d-json "$FT_PRED_JSON" --gt-json "$WILDBOX_VAL_JSON" \
    --out-pth "$PREDS"

if [[ "${SKIP_SCORE:-0}" == "1" ]]; then
    echo "[4/4] SKIP_SCORE=1 -- stopping after export. Score later in the ovmono3d env."
    echo "SEED $SEED export done at $(date). preds=$PREDS"
    exit 0
fi

# -- [4/4] Score: BEV AP + class-agnostic/NHD + standard AP_3D/2D + Rel-AP_3D
echo "[4/4] score seed=$SEED (ovmono3d env)"
run_ov
python "$OVMONO3D_REPO/tools/bev_ap_eval.py" \
    --gt "$WILDBOX_VAL_JSON" --preds "$PREDS" --out "$OUT_DIR/bev_ap.json"
python "$OVMONO3D_REPO/tools/class_agnostic_eval.py" \
    --gt "$WILDBOX_VAL_JSON" --preds "$PREDS" --nhd \
    > "$OUT_DIR/summary_nhd.txt" 2>&1
python "$DETANY3D_REPO/tools/wildbox_full_metrics.py" \
    --predictions "$PREDS" \
    --gt "$WILDBOX_VAL_JSON" \
    --ovmono3d-repo "$OVMONO3D_REPO" \
    --out-dir "$OUT_DIR/full_metrics" \
    --eval-rel-ap3d \
    --rel-ap3d-search "0.05,3.0,32" 2>&1 | tail -10

echo "============================================================"
echo "SEED $SEED DONE at $(date)"
echo "  row dir: $OUT_DIR"
for f in bev_ap.json summary_nhd.txt full_metrics/summary.json; do
    [[ -e "$OUT_DIR/$f" ]] && echo "  ok  $f" || echo "  MISSING  $f"
done
echo "============================================================"
