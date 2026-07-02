# DetAny3D × WildBox — multi-seed run (ep2, 5 seeds)

Completes the multi-seed variance study for the DetAny3D fine-tuned row. Seeds
**0** and **2** at ep2 (2-epoch fine-tune, oracle-2D, int1 full-val) are already
done and in the paper tables. This runbook adds seeds **1, 3, 4** → **5 seeds
total** for a tighter mean ± std on the rare classes (giraffe / gazelle /
Grévy's zebra).

**Why seed1 was pending:** it hung on the multi-GPU NCCL ALLGATHER `SeqNum=1`
bug (#10). The fix is to run **single-GPU** — every command here is
`--nproc_per_node=1`, and the sbatch requests `--gres=gpu:1`.

**What changed in the code (this branch):**
- `train.py` — new `--seed` arg that pins Python/NumPy/torch RNGs **and** the
  `DistributedSampler` shuffle. Omitting it reproduces the legacy
  nondeterministic path exactly, so the zero-shot eval config and the already-run
  seed0/seed2 are unaffected.
- `tools/wildbox_multiseed_prep.sh` — one-time shared data-prep (pkls + oracle link).
- `tools/wildbox_multiseed.sh` — per-seed train→eval→export→score, single-GPU,
  parallel-safe (FT checkpoint goes to eval via `--resume`, **not** a `sed` on the
  shared eval yaml).
- `tools/wildbox_multiseed.sbatch` — array wrapper; `--array=1,3,4`, task id = seed.
- `tools/aggregate_seed_ap_detany3d.py` — turns the N seed dirs into a mean ± std
  table (validated to reproduce the existing 2-seed paper numbers).

**Budget:** ep2 single-GPU ≈ 12–14 h train + ~3 h int1 eval ≈ **~15 h/seed**. As
a 3-task array on ≥3 free A40s, wall-clock is ~15 h. Each row dir is ~1–2 GB.

---

## 0. Push from your laptop, pull on the cluster

```bash
# laptop (this repo, branch wildbox_detany3d)
# NOTE: .gitignore has a blanket `*.sh` rule, so the two shell scripts must be
# force-added (-f) -- exactly how tools/wildbox_final.sh got tracked.
git add train.py tools/wildbox_multiseed.sbatch \
        tools/aggregate_seed_ap_detany3d.py MULTISEED_RUN_DETANY3D.md
git add -f tools/wildbox_multiseed_prep.sh tools/wildbox_multiseed.sh
git commit -m "Add 5-seed multi-seed pipeline (seed plumbing + single-GPU array + aggregator)"
git push fork wildbox_detany3d
```

```bash
# cluster
ssh frontendnew
cd /storage3/3DOM/vshukla/DetAny3D
git pull   # or: git fetch fork && git checkout wildbox_detany3d && git pull fork wildbox_detany3d
conda activate /storage3/3DOM/vshukla/envs/detany3d

# sanity: --seed is wired in
python train.py --help 2>/dev/null | grep -- --seed
```

## 1. Locate ovmono3d artifacts (same contract as FINAL_RUN §1)

```bash
export OVMONO3D_REPO=/storage2/3DOM/vshukla/repos/ovmono3d
ls -la $OVMONO3D_REPO/datasets/Omni3D/WildBox_train.json \
       $OVMONO3D_REPO/datasets/Omni3D/WildBox_val.json \
       $OVMONO3D_REPO/datasets/Omni3D/gdino_WildBox_val_oracle_2d.json
```

## 2. One-time data-prep (run ONCE, before the array)

Writes the shared pkls + oracle symlink so the concurrent array tasks only ever
read them. Normally a **no-op skip**, since seed0/seed2 already produced the pkls.

First just *check* (plain `ls` is fine on the frontend — it's data management):

```bash
cd /storage3/3DOM/vshukla/DetAny3D
ls -la data/pkls/wildbox/WildBox_train.pkl data/pkls/wildbox/WildBox_val.pkl \
       datasets/Omni3D/gdino_WildBox_val_oracle_2d.json
```

- **All three present** → skip this section entirely; go to §3.
- **Only the symlink missing** → recreate it (a link is data-management, OK on the
  frontend): `ln -sf /storage2/3DOM/vshukla/repos/ovmono3d/datasets/Omni3D/gdino_WildBox_val_oracle_2d.json datasets/Omni3D/gdino_WildBox_val_oracle_2d.json`
- **pkls missing** → run prep, but **on a CPU node, not the frontend** (DICLUB
  Policy 0: the frontend is for submitting jobs + moving data only; the converter
  loads multi-GB pickles = compute). Find your CPU partition with `sinfo -s`, then:

```bash
export OVMONO3D_REPO=/storage2/3DOM/vshukla/repos/ovmono3d
export WILDBOX_TRAIN_JSON=$OVMONO3D_REPO/datasets/Omni3D/WildBox_train.json
export WILDBOX_VAL_JSON=$OVMONO3D_REPO/datasets/Omni3D/WildBox_val.json
export GDINO_ORACLE_JSON=$OVMONO3D_REPO/datasets/Omni3D/gdino_WildBox_val_oracle_2d.json
export DETANY3D_ENV_PREFIX=/storage3/3DOM/vshukla/envs/detany3d

srun -p <cpu-queue> --mem=200000 --cpus-per-task=8 --time=00:30:00 \
     bash tools/wildbox_multiseed_prep.sh      # --mem in MB per DICLUB docs
```

Expect `train: 45979 images ...` / `val: 13779 images ...` and `OK`.

## 3. Submit seeds 1, 3, 4 (3 parallel single-GPU jobs)

The sbatch already carries the FINAL_RUN defaults (paths, envs, `gpu-A40`, 1 GPU,
200 G, 16 cpu, 24 h). Task id = seed, so `--array=1,3,4` needs no other change.

```bash
sinfo -p gpu-A40 -o "%n %T %G %C %m"          # check free A40s first
sbatch tools/wildbox_multiseed.sbatch          # submits tasks 1, 3, 4
```

Run just one seed instead:

```bash
sbatch --array=1 tools/wildbox_multiseed.sbatch
```

## 4. Monitor

```bash
squeue -u $USER
tail -f logs/multiseed_seed1_*.log             # per-seed training/eval log
# training is healthy once you see 'iter:' lines (~2 it/s on A40).
```

## 5. Verify each seed's output

```bash
export OVMONO3D_REPO=/storage2/3DOM/vshukla/repos/ovmono3d
for S in 1 3 4; do
  D=$OVMONO3D_REPO/output/wildbox_detany3d_ft_ep2_seed${S}_int1_v3
  echo "=== seed$S ==="
  for f in inference/iter_final/WildBox_val/instances_predictions.pth \
           bev_ap.json summary_nhd.txt full_metrics/summary.json; do
    [[ -e "$D/$f" ]] && echo "  ok  $f" || echo "  MISSING  $f"
  done
done
```

**Recovery** — if a task died after training but before scoring, its checkpoint is
kept and re-submitting skips straight to eval+score (training is skipped when a
`checkpoint_*.pth` already exists for that seed):

```bash
sbatch --array=3 tools/wildbox_multiseed.sbatch    # re-runs only seed3, reuses its ckpt
```

## 6. Download + aggregate all 5 seeds

Pull the 3 new row dirs next to the existing seed0/seed2 (here: `/mnt/d/detany3d`),
then aggregate. The aggregator globs `seed*`, so it picks up all five:

```bash
# laptop
python tools/aggregate_seed_ap_detany3d.py \
  --run-dirs /mnt/d/detany3d/wildbox_detany3d_ft_ep2_seed*_int1_v3 \
  --out /mnt/d/detany3d/ep2_multiseed_5seed
cat /mnt/d/detany3d/ep2_multiseed_5seed/table_multiseed.md
```

This writes `table_multiseed.md` + `aggregated.json` with `mean ± std` over 5
seeds — the row that replaces the "ep2 mean (2 of 3 seeds; seed1 pending)" line in
`PAPER_RESULTS.md §3`.

---

## Troubleshooting (multi-seed-specific; see FINAL_RUN §Troubleshooting for the rest)

1. **A task hangs at 100% GPU with no `iter:` line** — the NCCL multi-GPU bug. It
   should not occur here (single-GPU), but confirm `--gres=gpu:1` wasn't edited up.
2. **`eval val interval=N, expected 1`** at pre-flight — a smoke session left a
   non-1 stride in `wildbox_eval_oracle2d.yaml`; reset its `val ... range.interval`
   to 1 (int1 is what seed0/seed2 used).
3. **`WildBox pkls missing`** — you skipped §2; run `wildbox_multiseed_prep.sh` once.
4. **`Killed` at job start** — CPU-RAM cap; the sbatch already asks 200 G, raise to 256 G if needed.
5. **Only 1–2 tasks running** — fewer than 3 free A40s; the rest queue and start as GPUs free. Nothing to fix.
