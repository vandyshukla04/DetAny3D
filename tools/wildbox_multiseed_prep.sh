#!/usr/bin/env bash
# One-time data-prep for the DetAny3D x WildBox multi-seed study.
#
# Run this ONCE (on the login node or in a throwaway srun) BEFORE submitting
# tools/wildbox_multiseed.sbatch. It writes SHARED files -- the train/val
# pickles and the gdino oracle symlink -- that every per-seed array task then
# reads read-only. Doing it here (instead of inside the per-seed script)
# avoids a write race between the concurrent array tasks.
#
# Idempotent: skips pickle conversion if both pkls already exist. Since the
# earlier seed0/seed2 runs already produced these on the cluster, this is
# normally a no-op that just re-verifies + re-links.
#
# Required env (defaults match FINAL_RUN_DETANY3D.md):
#   OVMONO3D_REPO       - path to the ovmono3d clone
#   WILDBOX_TRAIN_JSON  - ovmono3d's WildBox_train.json
#   WILDBOX_VAL_JSON    - ovmono3d's WildBox_val.json
#   GDINO_ORACLE_JSON   - ovmono3d's gdino_WildBox_val_oracle_2d.json
#   DETANY3D_ENV_PREFIX - conda env prefix used for the converter
# Optional:
#   DETANY3D_REPO       - default $(pwd)
#   PATH_REMAP          - "/old=/new" image-path prefix rewrite (see wildbox converter)
set -eo pipefail

OVMONO3D_REPO="${OVMONO3D_REPO:-/storage2/3DOM/vshukla/repos/ovmono3d}"
WILDBOX_TRAIN_JSON="${WILDBOX_TRAIN_JSON:-$OVMONO3D_REPO/datasets/Omni3D/WildBox_train.json}"
WILDBOX_VAL_JSON="${WILDBOX_VAL_JSON:-$OVMONO3D_REPO/datasets/Omni3D/WildBox_val.json}"
GDINO_ORACLE_JSON="${GDINO_ORACLE_JSON:-$OVMONO3D_REPO/datasets/Omni3D/gdino_WildBox_val_oracle_2d.json}"
DETANY3D_REPO="${DETANY3D_REPO:-$(pwd)}"
: "${DETANY3D_ENV_PREFIX:?required -- set to your DetAny3D conda env prefix}"
PATH_REMAP="${PATH_REMAP:-}"

for f in "$WILDBOX_TRAIN_JSON" "$WILDBOX_VAL_JSON" "$GDINO_ORACLE_JSON"; do
    [[ -e "$f" ]] || { echo "ERROR: missing required input: $f" >&2; exit 1; }
done

CONDA_BASE=$(conda info --base 2>/dev/null || echo /opt/miniforge3)
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$DETANY3D_ENV_PREFIX"

cd "$DETANY3D_REPO"
mkdir -p data/pkls/wildbox datasets/Omni3D logs

TRAIN_PKL=data/pkls/wildbox/WildBox_train.pkl
VAL_PKL=data/pkls/wildbox/WildBox_val.pkl
REMAP_ARG=""; [[ -n "$PATH_REMAP" ]] && REMAP_ARG="--path-remap $PATH_REMAP"

if [[ -f "$TRAIN_PKL" && -f "$VAL_PKL" ]]; then
    echo "[prep] pickles already exist -- skipping conversion:"
    echo "       $TRAIN_PKL"
    echo "       $VAL_PKL"
else
    echo "[prep] converting WildBox JSONs -> DetAny3D pickles"
    python -m detect_anything.datasets.data_creator.wildbox \
        --wildbox-json "$WILDBOX_TRAIN_JSON" \
        --category-meta data/category_meta_wildbox.json \
        --out-pkl "$TRAIN_PKL" \
        $REMAP_ARG --verify-projection
    python -m detect_anything.datasets.data_creator.wildbox \
        --wildbox-json "$WILDBOX_VAL_JSON" \
        --category-meta data/category_meta_wildbox.json \
        --out-pkl "$VAL_PKL" \
        $REMAP_ARG --verify-projection
fi

echo "[prep] linking gdino oracle json"
ln -sf "$(realpath "$GDINO_ORACLE_JSON")" datasets/Omni3D/gdino_WildBox_val_oracle_2d.json

python -c "
import pickle
for split in ['train', 'val']:
    p = pickle.load(open(f'data/pkls/wildbox/WildBox_{split}.pkl', 'rb'))
    n_obj = sum(len(r['obj_list']) for r in p)
    print(f'  {split}: {len(p)} images, {n_obj} objects')
    assert len(p) > 0
"

echo "[prep] OK -- pickles + oracle link ready. Now submit tools/wildbox_multiseed.sbatch"
