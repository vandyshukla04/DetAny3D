#!/usr/bin/env bash
#SBATCH --job-name=wb_multiseed
#SBATCH --partition=gpu-A40
# Whole GPU (not a shard): SAM ViT-H + DINOv2 + UniDepth training needs most of
# an A40's 48 GB, so --gres=gpu:1 is "really needed" per DICLUB's GPU policy.
# This mirrors the proven FINAL_RUN_DETANY3D.md srun (same gres/mem/partition).
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/multiseed_seed%a_%A.log
#SBATCH --error=logs/multiseed_seed%a_%A.err
#SBATCH --array=1,3,4
#
# DetAny3D x WildBox multi-seed study -- ADD seeds 1,3,4 to the existing 0,2
# (=> 5 seeds total for a tighter mean+-std on the rare classes).
#
# Each array task = one seed on one A40, SINGLE-GPU (multi-GPU DDP hangs on
# this cluster; that hang is why seed1 was pending). SLURM_ARRAY_TASK_ID IS the
# seed, so --array=1,3,4 maps task->seed directly. If >=3 A40s are free the
# three seeds run concurrently (~15h wall); otherwise SLURM queues them.
#
#   Prep once:  bash tools/wildbox_multiseed_prep.sh
#   Submit:     sbatch tools/wildbox_multiseed.sbatch
#   Status:     squeue -u $USER
#   One seed:   sbatch --array=1 tools/wildbox_multiseed.sbatch
#   Cancel:     scancel <jobid>        (or scancel <jobid>_<seed> for one task)
#
# 200G mem / 16 cpu mirror FINAL_RUN_DETANY3D.md (image-decode workers are
# RAM-hungry; <128G has OOM-killed `import torch` at job start). Adjust the
# partition/qos lines if your cluster names them differently.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

# task id == seed; allow SEED override for a plain `bash` (non-array) run
export SEED="${SEED:-${SLURM_ARRAY_TASK_ID:?set SEED or run under sbatch --array}}"

# Paths + envs (defaults from FINAL_RUN_DETANY3D.md; override via
#   sbatch --export=ALL,OVMONO3D_REPO=...,... tools/wildbox_multiseed.sbatch )
export OVMONO3D_REPO="${OVMONO3D_REPO:-/storage2/3DOM/vshukla/repos/ovmono3d}"
export WILDBOX_VAL_JSON="${WILDBOX_VAL_JSON:-$OVMONO3D_REPO/datasets/Omni3D/WildBox_val.json}"
export DETANY3D_ENV_PREFIX="${DETANY3D_ENV_PREFIX:-/storage3/3DOM/vshukla/envs/detany3d}"
export OVMONO3D_ENV_PREFIX="${OVMONO3D_ENV_PREFIX:-/storage3/3DOM/vshukla/envs/ovmono3d}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-NA} array=${SLURM_ARRAY_JOB_ID:-NA} task/seed=$SEED"
# DICLUB policy: use ONLY the GPU(s) SLURM allocated via CUDA_VISIBLE_DEVICES,
# and never overwrite it. train.py honours this (torch.cuda.set_device(0) maps
# to the allocated card). Log it for the record.
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

bash tools/wildbox_multiseed.sh
