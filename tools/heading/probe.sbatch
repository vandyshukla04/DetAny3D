#!/bin/bash
# THE PROBE: does DINOv3 carry a HEAD part?   [cluster, one GPU]
#
#   sbatch tools/heading/probe.sbatch
#
# Runs all three stages in one job -- 12.7k crops through a frozen ViT-L/16 is minutes, not
# hours, so splitting them across jobs would only add queue latency.
#
#   A  extract_features  frozen DINOv3 -> pooled features + PATCH GRIDS      (GPU)
#   B  parts             the probe: is there a head part?  4-way, chance 25%
#   C  train_head        the supervised baseline, same metric, same crops
#
# Reads ONE input: $CROPS (shipped from the local disk by extract_crops.py).
# Everything is idempotent -- a re-submit skips any stage whose output already exists.
#
# Partition: gpu-A40 is usually busy. gpu-V100 (node8) is a known-good fallback:
#     sbatch --partition=gpu-V100 tools/heading/probe.sbatch
# Do NOT use gpu-1080 (node7 has a dead GPU -> "invalid device ordinal") or gpu-K80
# (Kepler sm_37; PyTorch 2.x dropped support).
#SBATCH --job-name=heading_probe
#SBATCH --partition=gpu-A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/heading_probe_%j.log
#SBATCH --error=logs/heading_probe_%j.err

set -euo pipefail

log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "[FATAL] $*" >&2; exit 1; }

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

OUT=${OUT:-data/heading}
CROPS=${CROPS:-$OUT/crops.npz}
FEATS=$OUT/features.npz
PATCHES=$OUT/patches.npz

[[ -f "$CROPS" ]] || die "no crops at $CROPS -- scp it from the local machine first"
log "crops: $CROPS ($(du -h "$CROPS" | cut -f1))"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || die "no GPU visible"

# ---- A: frozen DINOv3 -> pooled features + patch grids -------------------------------
if [[ -f "$FEATS" && -f "$PATCHES" ]]; then
    log "A: features already exist, skipping"
else
    log "A: extracting DINOv3 features + patch grids ..."
    python -m tools.heading.extract_features \
        --crops "$CROPS" --out "$FEATS" --patches "$PATCHES" \
        --patch-max "${PATCH_MAX:-6000}" --device cuda --batch "${BATCH:-64}" \
        || die "A (extract_features) failed"
fi
[[ -f "$FEATS" ]]   || die "A produced no $FEATS"
[[ -f "$PATCHES" ]] || die "A produced no $PATCHES"

# ---- B: THE PROBE --------------------------------------------------------------------
log "B: part probe -- is there a HEAD cluster?"
python -m tools.heading.parts \
    --patches "$PATCHES" --crops "$CROPS" --out "$OUT/parts" \
    --k "${K:-8}" --device cuda --viz \
    || die "B (parts) failed"

# ---- C: the supervised baseline, for comparison --------------------------------------
log "C: supervised baseline on the same crops, same 4-way metric"
python -m tools.heading.train_head \
    --features "$FEATS" --save "$OUT/head.pt" --device cuda \
    || die "C (train_head) failed"

log "DONE."
echo
echo "  Compare the two 4-way FRONT-FACE numbers (chance 25%):"
echo "    B  parts.py    -- training-free, no locomotion at all"
echo "    C  train_head  -- supervised on the motion labels"
echo "  And LOOK at $OUT/parts/parts_viz.jpg -- the part segmentation. A number can lie;"
echo "  a picture of which patches the 'head' cluster claims cannot."
