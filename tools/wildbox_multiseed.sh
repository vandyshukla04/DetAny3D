#!/usr/bin/env bash
# Per-seed WildBox fine-tune + eval for DetAny3D's multi-seed variance study.
#
# Runs ONE seed end-to-end, SINGLE-GPU. Multi-GPU DDP hangs on this cluster
# (NCCL ALLGATHER SeqNum=1, bug #10 -- see FINAL_RUN_DETANY3D.md); single-GPU is
# the fix, and is why every launch is --nproc_per_node=1.
#
# Design (robust-by-construction, so concurrent array tasks never collide):
#   * Unique torchrun rendezvous port PER LAUNCH via an OS-assigned free port
#     (torchrun's default 29500 collides when SLURM packs two tasks on one node).
#   * The fine-tuned checkpoint reaches the eval via train.py's --resume CLI
#     override -- never a `sed` on the shared eval yaml (which siblings would
#     clobber). Every exp_dir / output path is namespaced by ${SEED}.
#   * Idempotent + resumable: a completed row is a no-op; a re-submit skips
#     whatever already exists (final-epoch checkpoint -> skip train; exported
#     predictions -> skip train+eval+export) and only redoes the missing tail.
#   * "Training complete" is judged ONLY by the final-epoch checkpoint
#     checkpoint_{num_epochs-1}.pth -- never a mid-epoch safety checkpoint -- so a
#     re-submit after a mid-training death retrains instead of evaluating a
#     half-trained model.
#
# Data-prep (pkls + oracle symlink) is a one-time shared step: run
# tools/wildbox_multiseed_prep.sh ONCE before the array (this script errors clearly
# if it is missing).
#
# Output (byte-for-byte the seed0/seed2 layout, so the mean+-std aggregation in
# tools/aggregate_seed_ap_detany3d.py picks up all 5 seeds unchanged):
#   $OVMONO3D_REPO/output/wildbox_detany3d_ft_ep2_seed${SEED}_int1_v3/
#     inference/iter_final/WildBox_val/instances_predictions.pth
#     bev_ap.json  summary_nhd.txt  full_metrics/{summary.json,log.2D/3D/3D-Rel.txt}
#
# Required env (defaults match FINAL_RUN_DETANY3D.md):
#   SEED  OVMONO3D_REPO  WILDBOX_VAL_JSON  DETANY3D_ENV_PREFIX  OVMONO3D_ENV_PREFIX
# Optional:
#   DETANY3D_REPO  CONFIG_FT  CONFIG_EVAL  EXP_ROOT  SKIP_SCORE=1
#
# Exit codes: 1 pre-flight/env, 2 training incomplete, 3 eval produced no preds,
#             5 scoring failed, 6 row incomplete, other = torchrun's own code.
set -eo pipefail

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
die()  { echo "ERROR [seed ${SEED:-?}]: $1" >&2; exit "${2:-1}"; }
log()  { echo "[seed ${SEED:-?}] $*"; }

# An OS-assigned free TCP port on THIS node (race-proof against every user's
# jobs, not just ours). Deterministic fallback if python is somehow unavailable.
free_port() {
    local p
    p=$(python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()' 2>/dev/null) || true
    [[ "$p" =~ ^[0-9]+$ ]] || p=$(( 20000 + (${SLURM_JOB_ID:-$$} % 20000) ))
    echo "$p"
}

# Launch a single-GPU torchrun, tee to a log, and FAIL LOUDLY on nonzero exit
# (captures torchrun's real status through the tee pipe). Args: <logfile> <train.py ...>
run_torchrun() {
    local logf="$1"; shift
    local port; port="$(free_port)"
    log "launch torchrun --nproc_per_node=1 --master_port=$port $* (log: $logf)"
    set +e
    PYTHONUNBUFFERED=1 torchrun --nproc_per_node=1 --master_port="$port" "$@" 2>&1 | tee "$logf"
    local st=${PIPESTATUS[0]}
    set -e
    (( st == 0 )) || die "torchrun failed (exit $st) -- see $logf" "$st"
}

# --------------------------------------------------------------------------
# Config / environment
# --------------------------------------------------------------------------
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
source "$CONDA_BASE/etc/profile.d/conda.sh" || die "cannot source conda.sh from $CONDA_BASE"
run_da3d() { conda activate "$DETANY3D_ENV_PREFIX" || die "conda activate detany3d failed"; }
run_ov()   { conda activate "$OVMONO3D_ENV_PREFIX" || die "conda activate ovmono3d failed"; }

cd "$DETANY3D_REPO" || die "cannot cd to DETANY3D_REPO=$DETANY3D_REPO"
mkdir -p logs "$EXP_ROOT"

OUT_DIR="$OVMONO3D_REPO/output/wildbox_detany3d_ft_ep2_seed${SEED}_int1_v3"
PREDS="$OUT_DIR/inference/iter_final/WildBox_val/instances_predictions.pth"

# --------------------------------------------------------------------------
# Idempotency: if this seed's row is already complete, do nothing.
# --------------------------------------------------------------------------
if [[ -f "$OUT_DIR/full_metrics/summary.json" && -f "$OUT_DIR/bev_ap.json" && -f "$OUT_DIR/summary_nhd.txt" ]]; then
    log "row already complete -> $OUT_DIR ; nothing to do"
    exit 0
fi

# --------------------------------------------------------------------------
# Pre-flight (fail fast, fail clear)
# --------------------------------------------------------------------------
[[ -f data/pkls/wildbox/WildBox_train.pkl && -f data/pkls/wildbox/WildBox_val.pkl ]] \
    || die "WildBox pkls missing -- run tools/wildbox_multiseed_prep.sh ONCE first" 1
[[ -e datasets/Omni3D/gdino_WildBox_val_oracle_2d.json ]] \
    || die "gdino oracle symlink missing -- run tools/wildbox_multiseed_prep.sh" 1

run_da3d
# Guard the eval stride: int1 (full 13,779-frame val) is what seed0/seed2 used.
# A leftover interval:4/20 from a smoke session would silently subsample and make
# this seed non-comparable (FINAL_RUN_DETANY3D.md bug #11).
python - "$CONFIG_EVAL" <<'PY' || die "eval-config pre-flight failed (val interval must be 1 for int1)" 1
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
iv = cfg['dataset']['val']['wildbox']['range'].get('interval', 1)
assert iv == 1, f"eval val interval={iv}, expected 1 (int1 full-val)."
print(f"[preflight] eval val interval={iv} (int1 full-val) OK")
PY

log "start $(date) | node $(hostname) | FT=$CONFIG_FT | eval=$CONFIG_EVAL"

# --------------------------------------------------------------------------
# Train -> eval -> export  (skipped wholesale if predictions already exist)
# --------------------------------------------------------------------------
if [[ -f "$PREDS" ]]; then
    log "[1-3/4] predictions already exported ($PREDS) -- skipping train + eval + export"
else
    FT_EXP_DIR="$EXP_ROOT/ft_seed${SEED}"
    NUM_EPOCHS=$(grep -E '^num_epochs:' "$CONFIG_FT" 2>/dev/null | grep -oE '[0-9]+' | head -1 || true)
    NUM_EPOCHS=${NUM_EPOCHS:-2}
    LAST_EPOCH=$(( NUM_EPOCHS - 1 ))
    final_ckpt() { ls -t "$FT_EXP_DIR"/*/checkpoint_${LAST_EPOCH}.pth 2>/dev/null | head -1 || true; }

    # -- [1/4] Fine-tune (single-GPU, seeded) -------------------------------
    if [[ -n "$(final_ckpt)" ]]; then
        log "[1/4] final-epoch checkpoint present -> skip training ($(final_ckpt))"
    else
        log "[1/4] fine-tune num_epochs=$NUM_EPOCHS seed=$SEED -> $FT_EXP_DIR"
        run_torchrun "logs/multiseed_ft_seed${SEED}.log" \
            train.py --config_path "$CONFIG_FT" --seed "$SEED" --exp_dir "$FT_EXP_DIR"
    fi

    FT_CKPT="$(final_ckpt)"
    [[ -f "$FT_CKPT" ]] || die "no final-epoch checkpoint (checkpoint_${LAST_EPOCH}.pth) under $FT_EXP_DIR -- training incomplete/skipped; re-submit to retrain" 2
    # Advisory: tqdm progress snapshots in the tee'd log (checkpoint is the real gate).
    N_PROG=$(grep -oaE "[0-9]+/[0-9]+ \[" "logs/multiseed_ft_seed${SEED}.log" 2>/dev/null | wc -l | tr -d ' ')
    log "[1/4] done. checkpoint=$FT_CKPT (tqdm snapshots: ${N_PROG:-0})"

    # -- [2/4] Fine-tuned oracle-2D eval (--resume override; parallel-safe) --
    FT_EVAL_EXP_DIR="$EXP_ROOT/ft_eval_seed${SEED}"
    rm -rf "$FT_EVAL_EXP_DIR"
    log "[2/4] fine-tuned eval -> $FT_EVAL_EXP_DIR"
    run_torchrun "logs/multiseed_ft_eval_seed${SEED}.log" \
        train.py --config_path "$CONFIG_EVAL" --resume "$FT_CKPT" --exp_dir "$FT_EVAL_EXP_DIR"

    FT_EVAL_RUN_DIR=$(ls -dt "$FT_EVAL_EXP_DIR"/*/ 2>/dev/null | head -1 || true)
    [[ -n "$FT_EVAL_RUN_DIR" ]] || die "eval produced no run dir under $FT_EVAL_EXP_DIR" 3
    FT_PRED_JSON=$(ls "$FT_EVAL_RUN_DIR"/wildbox_*.json 2>/dev/null | head -1 || true)
    [[ -f "$FT_PRED_JSON" ]] || die "eval produced no prediction json under $FT_EVAL_RUN_DIR" 3
    log "[2/4] done. predictions=$FT_PRED_JSON"

    # -- [3/4] Export to ovmono3d instances_predictions.pth -----------------
    mkdir -p "$(dirname "$PREDS")"
    log "[3/4] export -> $PREDS"
    python tools/wildbox_export_predictions.py \
        --da3d-json "$FT_PRED_JSON" --gt-json "$WILDBOX_VAL_JSON" --out-pth "$PREDS" \
        || die "wildbox_export_predictions.py failed" 3
    [[ -f "$PREDS" ]] || die "export did not write $PREDS" 3
fi

if [[ "${SKIP_SCORE:-0}" == "1" ]]; then
    log "SKIP_SCORE=1 -- stopping after export. preds=$PREDS ($(date))"
    exit 0
fi

# --------------------------------------------------------------------------
# [4/4] Score: BEV AP + class-agnostic/NHD + standard AP_3D/2D + Rel-AP_3D.
# Each step is guarded so a transient failure in one preserves the others'
# outputs; the run still exits nonzero so the failure is visible.
# --------------------------------------------------------------------------
log "[4/4] score (ovmono3d env)"
run_ov
sfail=0
python "$OVMONO3D_REPO/tools/bev_ap_eval.py" \
    --gt "$WILDBOX_VAL_JSON" --preds "$PREDS" --out "$OUT_DIR/bev_ap.json" \
    || { echo "WARN [seed $SEED]: bev_ap_eval failed" >&2; sfail=1; }
python "$OVMONO3D_REPO/tools/class_agnostic_eval.py" \
    --gt "$WILDBOX_VAL_JSON" --preds "$PREDS" --nhd > "$OUT_DIR/summary_nhd.txt" 2>&1 \
    || { echo "WARN [seed $SEED]: class_agnostic_eval failed (see summary_nhd.txt)" >&2; sfail=1; }
python "$DETANY3D_REPO/tools/wildbox_full_metrics.py" \
    --predictions "$PREDS" --gt "$WILDBOX_VAL_JSON" --ovmono3d-repo "$OVMONO3D_REPO" \
    --out-dir "$OUT_DIR/full_metrics" --eval-rel-ap3d --rel-ap3d-search "0.05,3.0,32" \
    || { echo "WARN [seed $SEED]: wildbox_full_metrics failed" >&2; sfail=1; }
(( sfail == 0 )) || die "one or more scoring steps failed; predictions preserved ($PREDS) -- re-submit to retry scoring" 5

# --------------------------------------------------------------------------
# Final completeness verdict
# --------------------------------------------------------------------------
log "DONE $(date) -> $OUT_DIR"
missing=0
for f in bev_ap.json summary_nhd.txt full_metrics/summary.json; do
    if [[ -e "$OUT_DIR/$f" ]]; then log "  ok  $f"; else log "  MISSING  $f"; missing=1; fi
done
(( missing == 0 )) || die "row incomplete -- see WARN lines / logs above" 6
log "seed $SEED complete."
