#!/bin/bash
# Everything, into ONE folder you can scp back in a single command.
#
#   bash tools/heading/run_all.sh
#   scp -r <cluster>:/storage3/3DOM/vshukla/DetAny3D/data/heading/REPORT  D:\detany3d\
#
# Every console output is TEE'd to a log, so the report is reproducible from the artefacts rather
# than from anyone's memory. The final REPORT.md reads its numbers out of results.json -- it cannot
# quote a figure that was not actually measured.
set -uo pipefail

# Locate the repo from THIS SCRIPT's own path. Never from $SLURM_SUBMIT_DIR: inside an interactive
# srun that variable points at wherever the allocation was launched from, so the script silently
# cd'd out of the repo and then reported "no data/heading/crops.npz" -- a true statement about the
# wrong directory, which is the worst kind of error message.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1
echo "repo: $PWD"

LAYER=${LAYER:-24}
FACET=${FACET:-token}
SIZE=${SIZE:-224}
BATCH=${BATCH:-96}

D=data/heading
# Where everything is written. Override to keep old runs side by side:
#     OUT=data/heading/REPORT_v2 bash tools/heading/run_all.sh
R=${OUT:-$D/REPORT}
mkdir -p "$R"/{logs,fig,camfig,exp,desc}
echo "output: $R"

CROPS=$D/crops.npz
STAND=$D/crops_stand.npz
HUMAN=$D/crops_human.npz

log() { echo -e "\n\033[1m=== $* ===\033[0m"; }
die() { echo "[FATAL] $*" >&2; exit 1; }

[[ -f "$CROPS" ]] || die "no $CROPS  (cwd is $PWD -- is the file really there?)"

HUMAN_ARG=()
if [[ -f "$HUMAN" ]]; then
    HUMAN_ARG=(--human-crops "$HUMAN")
else
    echo "  WARNING: no $HUMAN -- the HUMAN check (grazing zebras) will be SKIPPED."
    echo "           It is the only number that does not depend on our own labels being right."
fi

STAND_ARG=()
if [[ -f "$STAND" ]]; then
    STAND_ARG=(--stand-crops "$STAND")
else
    # ${STAND:+...} would have passed the flag anyway -- the VARIABLE is set even when the FILE is
    # missing -- and experiments.py would have died on a path that does not exist.
    echo "  WARNING: no $STAND -- the TRANSFER TEST (standing animals) will be SKIPPED."
    echo "           That is the experiment the approach rests on; build it with bridges.py."
fi

# ---- 1. the mask join, per species. If this is wrong, nothing below can be trusted. ----
log "1/5  SAM mask join"
python -m tools.heading.check_masks --crops "$CROPS" --workers 32 \
    2>&1 | tee "$R/logs/1_masks.txt"

# ---- 2. THE experiments: main table, ablation, transfer ----
log "2/5  experiments (main table + ablation + transfer test)"
python -m tools.heading.experiments \
    --crops "$CROPS" "${STAND_ARG[@]}" "${HUMAN_ARG[@]}" \
    --out "$R/exp" --layer "$LAYER" --facet "$FACET" --size "$SIZE" --batch "$BATCH" \
    2>&1 | tee "$R/logs/2_experiments.txt"

# ---- 3. the layer/facet/resolution sweep (appendix) ----
log "3/5  layer x facet x resolution sweep (appendix)"
python -m tools.heading.sweep --crops "$CROPS" --out "$R/desc" --batch "$BATCH" \
    2>&1 | tee "$R/logs/3_sweep.txt"

# ---- 4. figures ----
log "4/5  figures"
python -m tools.heading.paper_fig --crops "$CROPS" --out "$R/fig" \
    --layer "$LAYER" --facet "$FACET" --size "$SIZE" --n-tracks 16 --frames 10 --batch "$BATCH" \
    2>&1 | tee "$R/logs/4a_paper_fig.txt"

python -m tools.heading.cam_fig --crops "$CROPS" --out "$R/camfig" \
    --layer "$LAYER" --facet "$FACET" --size "$SIZE" --n-tracks 16 --frames 10 --batch "$BATCH" \
    2>&1 | tee "$R/logs/4b_cam_fig.txt"

# ---- 5. the final report, generated FROM the artefacts ----
log "5/5  final report"
python -m tools.heading.final_report --report "$R" 2>&1 | tee "$R/logs/5_report.txt"

cp -f METHOD.md "$R/METHOD.md" 2>/dev/null || true

echo
echo "  Everything is in $R/"
echo "    REPORT.md   the final report -- every number read from exp/results.json"
echo "    METHOD.md   the method and the progression of decisions"
echo "    exp/        results.json (the raw measurements)"
echo "    fig/        16 two-row figures + figures.csv"
echo "    camfig/     16 camera-and-heading figures"
echo "    desc/       the layer/facet sweep"
echo "    logs/       every console output, verbatim"
echo
echo "  scp -r <cluster>:$PWD/$R  D:\\detany3d\\"
