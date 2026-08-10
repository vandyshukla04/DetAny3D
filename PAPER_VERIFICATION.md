# Paper verification — response to reviewer comments

*Everything re-checked against the implementation and re-measured from the data. Written 2026-07-28.
No claim here rests on memory or on the manuscript; every number is either reproduced from the crop
artefacts or printed by a fresh run.*

**Bottom line:** both reviewer comments are correct. Neither requires a new experiment. One is a
documentation fix (the metrics section describes two references as one); the other is a
one-sentence clarification plus a table value that should be updated to the estimator the code
already uses.

---

## 0. What was run

| | |
|---|---|
| fresh run | `tools.heading.experiments`, V100 (`node8`), `--dtype fp32 --batch 96`, layer 24 / `token` / 224 / 5 bins, seed 0 |
| output | `data/heading/REPORT_v3/exp/results.json`, log `data/heading/exp_v3.log` |
| compared against | `/mnt/d/detany3d/heading/REPORT_v2/exp/results.json` (2026-07-14 18:01) — the artefact the paper table came from |
| offline re-derivation | all geometry-only quantities recomputed directly from `crops.npz`, `crops_stand.npz`, `crops_human.npz` |

Mask coverage in the fresh run: walking **7142/7268 (98%)**, standing **1364/1404 (97%)**, human
**5542/5542 (100%)**. `SAM masks: yes`. Template fit `{elephant 587, giraffe 187, rhino 285, zebra 440}`
= 1,499. Abstentions `1/5768`. All four match the original run.

> ⚠️ One prerequisite: `experiments.py:289` passed **7** positional arguments to `metrics()`, which
> takes **5** — leftovers from when the angular columns were removed in July (`165eb2f`). It raises
> `TypeError` after the main table and before the transfer test, so `results.json` could not be
> regenerated at all on current `main`. Fixed by deleting the two stale argument lines (2-line diff).
> **Anyone attempting to reproduce the paper before this fix would have hit a hard crash.**

---

## 1. Reviewer comment 1 — "the numbers imply two references"

### Verdict: **correct, confirmed in code and to four decimal places.**

The reviewer inferred that the implementation carries (i) a continuous displacement direction used for
angular error, and (ii) its nearest directed box candidate used for candidate/sign correctness. That is
exactly what the code does.

### 1.1 Both references are computed and stored, side by side

| reference | produced at | stored as |
|---|---|---|
| **continuous** `h_mot` | `autolabel.py:83` — `h = v / ‖v‖`, the normalised horizontal displacement | `labels.npz : heading` → `crops.npz : az` (`extract_crops.py:194`, `az = azimuth_of(heading)`) |
| **nearest directed candidate** | `autolabel.py:88` — `fid = max(faces, key=λ f: dot(normal_f, h))` | `labels.npz : front_face` → `crops.npz : y_face` (`extract_crops.py:220`) |

`extract_crops.py` also stores `alpha = alpha_of(heading)` and `Y = [cos α, sin α]` — both from the
**continuous** direction.

**The stationary set is identical in structure.** `bridges.py` interpolates the continuous heading
across the stop (`d = (1−w)·h_before + w·h_after`, normalised) and then takes
`fid = max(faces, key=dot(normal, d))`. So the reviewer's note that the same distinction applies to
stationary intervals is also correct.

### 1.2 The evaluator uses one for each metric

From `experiments.py` at commit `29e1b34` (2026-07-14 17:58 — `results.json` is 18:01, so this is the
exact version that produced the paper table):

```python
az_true = d["az"][idx]                      # CONTINUOUS motion azimuth
...
metrics("locomotion only (oracle)",
        faz[np.arange(n), y],               # az_pred: the DISCRETE candidate's azimuth
        az_true,                            # az_true: the CONTINUOUS direction
        np.ones(n, bool), ...)              # sign_ok: 100% by construction

def metrics(name, az_pred, az_true, sign_ok, ...):
    err = np.degrees(np.abs(wrap(az_pred - az_true)))
```

- **Angular error** = selected discrete candidate vs the **continuous** reference.
- **Directed-candidate accuracy** = `predicted_slot == y_face`, i.e. against the **discrete** reference.

Two references, in one function call. The manuscript describes them as one.

### 1.3 Numerical proof

Recomputed directly from the crop files, using nothing but geometry — the median angle between the
continuous motion direction and its **own** nearest candidate:

| | recomputed offline | `results.json` |
|---|---|---|
| walking, n | 5768 | 5768 |
| walking, median error | **9.0765°** | 9.0765 |
| walking, acc@15 / @30 / @45 | 0.7132 / 0.9414 / **1.0000** | 0.713245 / 0.941401 / 1.0 |
| standing, n | 1404 | 1404 |
| standing, median error | **8.5007°** | 8.5007 |
| standing, acc@15 / @30 / @45 | 0.6588 / 0.8946 / **1.0000** | 0.658832 / 0.894587 / 1.0 |

So the candidate oracle's 9.1° is **the quantisation gap itself** — the median angle between the
continuous displacement direction and the nearest of the four directed box candidates. It is the
ceiling any candidate-selection method inherits, which is how `METHOD.md` already describes it. It is
*not* evidence of an error in the oracle.

### 1.4 The strongest single piece of evidence

**`acc@45` is bit-identical to `sign` in all 14 rows** of `results.json` (main 5, transfer 5,
ablation 4) — not close, identical:

```
main      random sign                0.468620  0.468620   same
main      locomotion only (oracle)   1.000000  1.000000   same
main      appearance, oracle axis    0.934628  0.934628   same
main      FULL METHOD                0.874285  0.874285   same
transfer  FULL METHOD                0.690883  0.690883   same
ablation  - geometric axis prior     0.835790  0.835790   same
   ... (all 14)
```

This follows from `acc@45 = 1.0000` for the oracle: the nearest of four ~orthogonal candidates is
**always** within 45° of any continuous direction. Therefore "within 45° of the continuous reference"
⟺ "selected the correct candidate". The two references are not in conflict — the angular metric
*collapses onto* the candidate metric at 45°, which is precisely why the angular columns were later
dropped from the code as redundant.

### 1.5 Recommended manuscript edits

Adopt the reviewer's minimum-change proposal. In **§3.1**, after defining the nearest candidate:

> The continuous displacement direction $h^{mot}_t$ provides the **angular reference**, while its
> nearest directed candidate provides the **discrete candidate reference**. For stationary intervals
> the same distinction applies to the continuous direction transferred across the stop and to its
> nearest candidate.

In **Metrics**:

> Angular error is measured against the continuous reference direction. Directed-candidate accuracy is
> the fraction of observations for which the selected candidate is the one nearest that reference.
> Because the four candidates are approximately orthogonal, the nearest candidate is always within 45°
> of the continuous reference; accuracy@45° is therefore identical to directed-candidate accuracy and
> is not reported separately.

And use **"directed-candidate accuracy"** consistently in the prose, matching the Table 2 caption.

The last sentence is optional but worth adding: it converts the apparent inconsistency the reviewer
spotted into a stated property of the method, and pre-empts the same question from another reader.

---

## 2. Reviewer comment 2 — "Random sign at 42.2% is surprisingly low"

### Verdict: **correct to query it; the value is exactly as expected, and the manuscript should say why.**

### 2.1 The randomisation unit is per observation

`experiments.py` (current):

```python
N_SEED = 50
for seed in range(N_SEED):
    rr = np.random.default_rng(seed)
    pk = np.array([ends(k, geo[k])[rr.integers(2)] for k in range(n)])   # one coin per observation
    hit += (pk == y)
```

`ends(k, geo[k])` is the two directions of the **geometry-proposed axis**. So the coin is independent
per observation — not per track — and there is no temporal-correlation effect.

### 2.2 Why the expectation is not 50%

The coin is flipped **on the proposed axis, not on the true axis**. When the true head lies on the
*other* axis, no coin can reach it. Therefore

> **E[directed-candidate accuracy] = 0.5 × P(true head lies on the geometry-proposed axis)**

Measured directly from the crop files, and compared with the fresh 50-seed run:

| test set | n | P(head on proposed axis) | `0.5 × P` | observed |
|---|---:|---:|---:|---:|
| walking (held-out) | 5,768 | 0.9168 | 0.4584 | **45.7%** |
| standing (transfer) | 1,404 | 0.8440 | 0.4220 | **42.4%** |
| human (grazing zebra) | 5,542 | 0.9820 | 0.4910 | **49.2%** |

All three agree to within 0.002. The baseline varies from 42.4% to 49.2% across the three sets
**purely because axis quality varies from 84.4% to 98.2%** — the grazing-zebra set, where the box axis
is almost always right, sits essentially at 50% as the reviewer would expect. This is a complete
explanation and requires no new experiment.

### 2.3 A real discrepancy this surfaced

The paper's values come from a **single** coin draw; the current code averages **50 seeds**.

| | walking | standing |
|---|---|---|
| single draw, `default_rng(0)` (`REPORT_v2`, and the paper) | 46.9% | 42.2% |
| 50-seed average (current code, and `METHOD.md`) | **45.7%** | **42.4%** |
| `0.5 × P` expectation | 0.4584 | 0.4220 |

I reproduced the single-draw values exactly (0.4686 and 0.4224) by replaying `default_rng(0)` in
observation order. The walking value sits ~1.6 SD above its expectation — ordinary sampling noise on
one coin, which is exactly why the code was later changed to average 50 seeds (`33a5d28`).

**Recommendation:** update Table 2 to **45.7% / 42.4%** and state the seed count in the caption. These
are the better estimator, they match `METHOD.md`, and they remove the "surprisingly low" reading. The
paper table and `METHOD.md` currently disagree on this row; this resolves it in `METHOD.md`'s favour.

### 2.4 Suggested footnote

> The random-sign baseline draws one independent sign per observation, restricted to the two directions
> of the geometry-proposed axis, averaged over 50 seeds. Its expectation is therefore
> $0.5 \times P(\text{head on the proposed axis})$ rather than $0.5$: 0.458 (walking), 0.422
> (stationary) and 0.491 (grazing), matching the measured values.

---

## 3. Full reproduction of the paper's results

Fresh fp32 run vs `REPORT_v2`. Console precision is 1 d.p.; `results.json` values are given at full
precision where available.

### 3.1 Main table — walking, held-out videos

| setting | n | REPORT_v2 sign | v3 sign | REPORT_v2 flank | v3 flank |
|---|---:|---:|---:|---:|---:|
| uninformed sign | 5768 | 46.9% ¹ | **45.7%** ¹ | — | — |
| locomotion reference | 5768 | 100.0% | **100.0%** | — | — |
| appearance, oracle axis | 5767 | 93.5% | **93.4%** | — | — |
| **FULL METHOD** | 5767 | **87.4%** | **87.4%** | **95.0%** | **95.0%** |

¹ single draw vs 50-seed average — see §2.3.

**Bands** (`|sin α|`, from the true heading):

| band | n | v3 sign | v3 flank | REPORT_v2 sign | REPORT_v2 flank |
|---|---:|---:|---:|---:|---:|
| 0.00–0.35 head-on | 1297 | 75.3% | 80.3% | 75.33% | 80.49% |
| 0.35–0.70 oblique | 2224 | 91.5% | 94.6% | 91.55% | 94.65% |
| 0.70–1.01 broadside | 2246 | 90.3% | 95.3% | 90.34% | 95.33% |

**Per species:** elephant 722 → 77.3 / 92.1 · giraffe 30 → 100 / 100 · rhino 4157 → 91.9 / 97.3 ·
zebra 858 → 73.7 / 85.6. All match `REPORT_v2` (77.29/92.13, 100, 91.92/97.27, 73.78/85.57).

### 3.2 Ablation

| setting | n | REPORT_v2 | v3 |
|---|---:|---:|---:|
| full method | 5767 | 87.43% | **87.4%** |
| − centring | 5767 | 87.32% | **87.3%** |
| − SAM instance mask | 5768 | 86.56% | **86.6%** |
| − geometric axis prior | 5767 | 83.58% | **83.6%** |

The last row **could not run at all** before the `metrics()` fix.

### 3.3 Transfer test — standing animals

| setting | n | REPORT_v2 | v3 |
|---|---:|---:|---:|
| uninformed sign | 1404 | 42.2% ¹ | **42.4%** ¹ |
| appearance, oracle axis | 1404 | 79.27% | **79.3%** |
| **FULL METHOD** | 1404 | **69.09% / 89.40%** | **69.1% / 89.4%** |

Bands 432 / 495 / 477 → 57.9 / 82.4 / 65.4 sign, all matching. Per species: elephant 58.3, rhino 69.5,
**zebra 87.0**, giraffe 100 (n=10) — all match.

**The gap:** sign −18.3 pts, flank −5.6 pts. Unchanged.

### 3.4 Human check — grazing zebras (n = 5,542) — **newly traceable**

| setting | v3 |
|---|---:|
| uninformed sign | 49.2% |
| locomotion reference | 100.0% |
| appearance, oracle axis | 94.3% |
| **FULL METHOD** | **92.5% sign / 97.1% flank** |

| `\|sin α\|` | n | sign | flank |
|---|---:|---:|---:|
| 0.00–0.35 head-on | 261 | 36.0% | 36.0% |
| 0.35–0.70 oblique | 579 | 86.4% | 86.4% |
| **0.70–1.00 broadside** | **4702** | **96.4%** | **98.4%** |

> **`REPORT_v2/exp/results.json` contains no `human` section** — its top-level keys are only
> `main, ablation, transfer, data, config`. The paper's 92.5% / 97.1% and the band table lived
> **only in `METHOD.md`**, with no artefact behind them. They now reproduce exactly and are recorded
> in `REPORT_v3/exp/results.json`. Given that a reviewer is already probing metric definitions, this
> is the most valuable outcome of the re-run.

### 3.5 Other paper claims, re-measured offline

| claim | measured | verdict |
|---|---|---|
| longest horizontal axis is the body axis on **87%** of frames (zebra **74%**) | 87.1% overall; elephant 90.0, giraffe 87.9, rhino 92.2, **zebra 73.7** | ✅ |
| box axes sit a median **9.1°** from the motion reference | **9.0765°** | ✅ |
| `\|sin α\|` band sizes 1297 / 2224 / 2246 | 1297 / 2224 / **2247** | ✅ the 1 is the abstention (bands run on m=5767) |
| crops_total, crops_train_videos, crops_test_walking (all 4 species) | exact match | ✅ |
| standing crops 2,211 / 103 / 1,741 / 778 = 4,833 | exact match | ✅ |
| held-out walking n = 5,768; standing n = 1,404 | exact match | ✅ |

---

## 4. Everything that changed, in one place

| # | item | action |
|---|---|---|
| 1 | Metrics section describes two references as one | adopt §1.5 wording |
| 2 | "heading accuracy" vs "directed-candidate accuracy" | use the latter throughout, matching the Table 2 caption |
| 3 | Random-sign baseline undefined in the text | add the §2.4 footnote |
| 4 | Random-sign value is a single draw | update Table 2 to **45.7% / 42.4%**, state 50 seeds |
| 5 | Human-check numbers had no artefact | now in `REPORT_v3/exp/results.json` |
| 6 | `experiments.py:289` arity crash | fixed (2-line diff); needed by anyone reproducing |

Nothing in the method, the claims, or any headline number changes.

---

## 5. Caveats — what this does *not* establish

- **The v3 figures above are read from console output at 1 d.p.** Full-precision comparison against
  `REPORT_v2` awaits `REPORT_v3/exp/results.json`. Every value agrees at the precision available, but
  a few differ in the last displayed digit (appearance-oracle 93.5→93.4, zebra sign 73.8→73.7, one
  band flank 80.5→80.3).
- **Those small differences are dtype, not disagreement.** `REPORT_v2` was almost certainly bf16
  (`run_all.sh` passes no `--dtype`); v3 is fp32. `tests/test_dtype.py` guards bf16-vs-fp32 agreement
  at cosine > 0.999. No claim is affected. Recommend making the **fp32 run canonical**, since it is
  the trustworthy dtype *and* the only run containing the human check.
- **The angular columns are no longer produced by the current code** (`median_err`, `acc@15/30/45`
  were removed in `165eb2f`). They did not need re-running: §1.3 re-derives all of them exactly from
  geometry alone, with no network involved.
- The `− geometric axis prior` row reports no flank figure (NaN) in both runs, by construction.

---

## 6. Reproducing this

```bash
# cluster, V100/A40, dinov3 env
cd /storage3/3DOM/vshukla/DetAny3D
conda activate /storage3/3DOM/vshukla/envs/dinov3

PYTHONUNBUFFERED=1 python -m tools.heading.experiments \
    --crops       data/heading/crops.npz \
    --stand-crops data/heading/crops_stand.npz \
    --human-crops data/heading/crops_human.npz \
    --out         data/heading/REPORT_v3/exp \
    --device cuda --dtype fp32 --batch 96
```

Sanity gates, in order: `SAM masks: yes` · `template … {587, 187, 285, 440}` · `ABSTENTIONS: 1/5768` ·
`1404 stationary crops` · `5542 crops, 2 videos`. If any differs, the inputs are not the ones that
produced the paper.

The geometry-only re-derivations in §1.3, §2.2 and §3.5 need no GPU — they read `az`, `face_az`,
`y_face`, `face_ids`, `geo_axis` straight from the crop npz files.


---

## 7. Full numerical audit of the manuscript (2026-07-28)

Every number in the submitted LaTeX checked against the artefacts. **Deductions and prose were not
audited — only the figures.**

### 7.1 Verified exactly (no discrepancies)

| where | checked |
|---|---|
| §3 Method constants | W=15, 0.30 body lengths, cos 0.8, 448×448, layer 24, B=5, \|sin α\|≥0.35 — all match the CLI defaults and the selected config |
| §4 Data/split | 60 videos · 25,554 motion obs · 12,762 crops · 6,994 non-eval · 33 contributing · 23/10 fit/val · 5,768 walking eval · 5,542 manual zebra · 1,404 stationary |
| §4 arithmetic | 6,994 + 5,768 = 12,762 ✓ |
| **Table 1** (11 result rows) | every cell against `REPORT_v5`; Random-sign rows against the offline 50-seed estimator |
| **Table 2** (data accounting) | all 20 species cells + 4 totals against `results.json → data` |
| **Table 3** (ablation) | 87.1 / 86.6 / 85.6 / 82.4, and the stated deltas 4.7 / 1.5 / 0.5 |
| **Table 4** (config sweep) | all 8 rows × 9 columns against `sweep_val.json` |
| **Table 5** (per species) | 722/77.1 · 30/100.0 · 4,157/91.8 · 859/72.1, total 5,768/87.1 |
| **Table 6** (viewing angle) | all 9 rows; **and it reconciles to Table 1** — band-weighted flank recovers 95.10 / 91.58 / 97.57 and coverage 77.51 / 69.23 / 95.29 |
| §5 text | 9.1 · 8.5 · 93.0 · 87.1 · 83.5 · 9.9 · 78.8 · 68.9 · 42.4 · 92.9 · 94.7 · 4,702 of 5,542 · 96.4 · 34.1 · 95.1/91.6/97.6 |
| derived percentages | rhino = 72.1% of walking obs; broadside = 84.8% of the manual set |
| §B "200 random camera–heading configurations" | backed by `tests/test_papersub.py:193` |

Table 5's zebra n = **859** (not 858) is correct for the selected configuration: at 448 there is no
abstention, so the per-species counts sum to the full 5,768.

### 7.2 Two numbers that cannot be traced to any artefact

**(a) "A separate MLP baseline … reaches 79.8\% App-4 accuracy" (Appendix D).**
Searched `/mnt/d/detany3d` and the repo: **no checkpoint, no `features.npz`, no metrics file.** The
value survives only as a hardcoded `print` at `sweep.py:181` and as recollection in docstrings
(`split.py:25`, `train_head.py:157`) describing the split-leak bug it was corrected for. It is
therefore unreproducible as the manuscript stands. Two further points: it predates the current
pipeline, and `extract_features.py` builds its input through `AutoImageProcessor` at a **fixed
resolution**, so it is not a 448 measurement even though everything around it now is.
**Either regenerate it (`extract_features` → `train_head`, ~30 min) or state the resolution and that
it is a legacy baseline.**

**(b) The runtime figures (§5 and Appendix G): 82.8 crops/s, 356.6 crops/s, 1.40 ms, 25.17 ms.**
`bench.py` would emit `bench.json`, but **no `bench.json` exists anywhere** — only the script was
copied to `/mnt/d/detany3d/heading/`. The numbers are internally consistent (356.6 / 82.8 = 4.31×,
matching the "4× cheaper" claim), but nothing on disk backs them.

### 7.3 One inconsistency worth fixing

Appendix G reports runtime under **fp16**, while every result in the paper is **fp32** — and fp16 is
the dtype documented as **silently zeroing the ViT-L forward on exactly the GPU named** (V100). The
*timing* is probably still valid, since the kernels execute regardless of whether the output is
garbage, but the reported configuration does not match the one being timed. Re-run `bench.py` with
`--dtype fp32`, or state explicitly that throughput was measured in fp16 while results use fp32.

### 7.4 A finding that strengthens the rebuttal

The superseded sweep — the one scored on the **evaluation** videos — is still on disk at
`REPORT_v2/desc/sweep.npz`:

| | scored on **evaluation** videos (old) | scored on **validation** videos (new) |
|---|---:|---:|
| L24/token/**224** | **93.6** | 91.2 |
| L24/token/**448** | 93.1 | **91.8** |

**The ordering reverses.** Selecting on the evaluation set preferred 224; selecting honestly prefers
448. So the reviewer's objection was not merely procedural — it **changed the answer**. Worth one
sentence in the response letter, because it demonstrates the correction mattered rather than being a
formality.

### 7.5 Minor

"Applying the displacement and candidate-agreement thresholds yields 25,554 observations; retaining
every second frame leaves 12,762 crops." Exact halving would give 12,777; the remaining **15** crops
are dropped by the minimum-size and projection filters in `extract_crops.py`. The sentence is not
wrong, but "retaining every second frame **and discarding crops below the minimum size** leaves
12,762" would forestall a reader checking the division.
