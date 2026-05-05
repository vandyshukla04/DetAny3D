# WildBox — DetAny3D fine-tune / eval code

Supplementary code for the WildBox monocular 3D wildlife detection benchmark
(anonymous review submission).

This repository extends **DetAny3D** (*Detect Anything 3D in the Wild*,
arXiv:2504.07958) with the WildBox dataset preparation, fine-tuning recipe,
and evaluation pipeline used to produce the DetAny3D rows in the paper. The
architecture itself is unchanged from upstream; for the upstream README,
checkpoints, and credits, see [`ORIGINAL_DETANY3D_README.md`](ORIGINAL_DETANY3D_README.md).

---

## What's in this repo

| Path | Purpose |
|------|---------|
| `detect_anything/` | DetAny3D model code (upstream — unchanged) |
| `GroundingDINO/` | Pinned Grounding-DINO submodule for 2D oracle prompting (upstream) |
| `train.py` | DetAny3D training / eval entry point |
| `tools/wildbox_final.sh` | one-shot WildBox eval driver: prepares oracle 2D, runs ZS + FT, exports predictions, compiles paper tables |
| `tools/wildbox_export_predictions.py` | converts DetAny3D inference outputs to the cross-arch comparison format |
| `tools/wildbox_full_metrics.py` | runs 2D AP, 3D AP, BEV AP, and NHD on a checkpoint's predictions |
| `tools/wildbox_compile_results.py` | aggregates all eval outputs into the paper tables |
| `tools/wildbox_compose_panel.py` | builds the cross-architecture qualitative comparison figure |
| `tools/visualize_class_consistent.py` | per-architecture qualitative viz with consistent class colors |
| `data/` | Dataset configs and category metadata |

---

## Setup

The DetAny3D training stack pins `torch 1.13.1+cu116`, `mmcv 2.0.1` (with CUDA
ops), `opencv-python-headless`, and a specific Grounding-DINO commit; mixing
versions silently breaks the multi-prompt 3D head. Follow upstream's setup:

```bash
# upstream env
conda create -n detany3d python=3.8.20
conda activate detany3d
pip install -r requirements.txt
# build mmcv-full and the GroundingDINO submodule per ORIGINAL_DETANY3D_README.md
```

Then download the upstream pre-trained weights as described in
`ORIGINAL_DETANY3D_README.md` ("Checkpoints" section).

---

## Reproducing the paper's DetAny3D row

The paper reports DetAny3D **fine-tuned for 2 epochs (oracle 2D), seed 0**
as the headline DetAny3D number. ep3 overfits — see the ablation table.

The eval is wrapped end-to-end by `tools/wildbox_final.sh`:

```bash
export WILDBOX_TRAIN_JSON=/path/to/WildBox_train_paper.json
export WILDBOX_VAL_JSON=/path/to/WildBox_val_paper.json
export GDINO_ORACLE_JSON=/path/to/gdino_WildBox_val_oracle_2d.json
export OVMONO3D_REPO=/path/to/the/ovmono3d/checkout       # for cross-arch metrics
export OVMONO3D_ENV_PREFIX=/path/to/your/ovmono3d/conda/env
export DETANY3D_ENV_PREFIX=/path/to/your/detany3d/conda/env
export NUM_GPUS=4

bash tools/wildbox_final.sh
```

This produces, under `exps/wildbox_final_*/`:

* `inference/iter_final/WildBox_val/instances_predictions.pth` — raw model output
* `bev_ap.json`, `summary_nhd.txt`, `full_metrics/` — per-run metrics
* `reports/` — compiled paper tables (after `wildbox_compile_results.py`)

### Single-architecture comparison

To regenerate just the DetAny3D rows of the headline table from existing eval
outputs:

```bash
python tools/wildbox_compile_results.py \
    --row exps/wildbox_final_ft_eval_zs_oracle:"DetAny3D zero-shot (oracle 2D)" \
    --row exps/wildbox_final_ft_eval_zs_gt2d:"DetAny3D zero-shot (GT 2D)" \
    --row exps/wildbox_final_ft_eval_ep1_seed0_int1:"DetAny3D fine-tuned (1 epoch)" \
    --row exps/wildbox_final_ft_eval_ep2_seed0_int1:"DetAny3D fine-tuned (2 epochs, headline)" \
    --row exps/wildbox_final_ft_eval_ep3p_seed0_int1:"DetAny3D fine-tuned (3 epochs, overfits)" \
    --out-dir reports
```

### Cross-architecture qualitative figure

For the cross-arch composite (DetAny3D vs OVMono3D-LIFT, ZS vs FT, per
species), see `tools/wildbox_compose_panel.py`. It expects predictions
from both architectures to have already been exported via
`tools/wildbox_export_predictions.py`.

---

## Repo origin and licensing

This repo is a fork of [DetAny3D by OpenDriveLab](https://github.com/OpenDriveLab/DetAny3D)
The architecture, the SAM-based 2D head, the multi-prompt 3D head, and the
foundation-model integration are all from upstream — see
`ORIGINAL_DETANY3D_README.md` for full attribution. The contributions in
this fork are: WildBox dataset registration, the 2-epoch fine-tuning recipe,
the cross-architecture eval scripts, and the qualitative-comparison
panel-builder. License is unchanged from upstream (see `LICENSE`).
