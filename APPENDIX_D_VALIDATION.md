# Appendix D, corrected — configuration selected on validation, not on the evaluation videos

*Response to the one review point that needs a real experiment. Sections **A** and **E** are complete.
**B**, **C** and **D** are filled by the run in §6.*

---

## 1. The criticism, confirmed in code

`tools/heading/sweep.py:102-108`:

```python
te, test_v = video_split(crops.species, crops.video, seed=args.seed)   # the 20 EVALUATION videos
idx_fit = ...np.where(~te)[0][:subset]      # fit on the template videos
idx_te  = ...np.where( te)[0][:subset]      # ...and SCORE ON THE EVALUATION VIDEOS
```

The eight DINOv3 configurations were compared on the same 20 videos the headline numbers are reported
on. The reviewer is right: the configuration choice is a selection effect on the evaluation set. No new
data and no change to the 20 evaluation videos is required — only that the choice be made elsewhere.

---

## 2. The corrected protocol

```
33 template videos                              20 evaluation videos
        │                                                │
   ┌────┴────┐                                           │
  23 fit   10 validation                                 │
   │          │                                          │
   └── 8 configurations ──► validation 2-way ──► SELECT  │   (evaluation videos take no part)
                                                  │      │
                          all 33 template videos ─┴──────┤
                                   │                     │
                            final templates ─────────────┴──► final numbers
```

Implemented in `tools/heading/sweep_val.py` (new) and `split.template_val_split` (new).
`sweep.py` is left in place, superseded.

**Selection rule, fixed before any result was seen:** highest **overall (crop-weighted) validation
2-way accuracy**. Ties break to the **lower resolution**, then the **lower layer**. `App-4`, `Geo`,
`Prior`, the per-species columns and a macro (per-species mean) 2-way are reported as diagnostics and
play no part in the choice.

2-way is the criterion because layer/facet/resolution is being chosen for the **front-vs-back
appearance decision**, and 2-way measures exactly that with the unsigned axis supplied — so it does not
confound the descriptor choice with the separate geometric-axis module.

**Split rule, also fixed in advance:** largest-remainder apportionment of 10 validation videos across
species, proportional to each species' template-video count, subject to (i) a species with ≥2 videos
gets ≥1 validation video, and (ii) every species keeps ≥1 fit video. Seed 0. Verified programmatically:
fit and validation are disjoint, and **neither contains any of the 20 evaluation videos**.

---

## A. The split summary

> ⚠️ **The template side has 33 videos, not 40.** The crop total (6,994) matches the manuscript
> exactly, but **7 of the 40 non-evaluation videos yield zero walking crops**, so they never enter
> `crops.npz` and cannot contribute to a template. The manuscript's "40 non-evaluation videos" should
> read "40 non-evaluation videos, 33 of which contribute walking anchors". The reviewer's suggested
> 30/10 split becomes **23/10** on the videos that actually exist.

```
Template-fit: 23 videos, 4,966 walking crops   (templates built from a 1,500-crop sample, as in the final method)
Validation:   10 videos, 2,028 walking crops
Evaluation:   20 videos, 5,768 walking observations   — untouched
```

| species | fit videos | fit crops | validation videos | validation crops |
|---|---:|---:|---:|---:|
| elephant | 6 | 2,113 | 3 | 779 |
| giraffe | 1 | 237 | 1 | 561 |
| rhino | 8 | 813 | 3 | 487 |
| zebra | 8 | 1,803 | 3 | 201 |
| **total** | **23** | **4,966** | **10** | **2,028** |

Validation videos: `DJI_20230607092449_0002_V`, `DJI_20240118140338_0007_V`,
`DJI_20250218180030_0034_D`, `DJI_20250224155903_0001_D`, `DJI_20250312171212_0026_D`,
`DJI_20250802120902_0003_V`, `DJI_20250802130951_0002_V`, `DJI_20260227082000_0001_V`,
`DJI_20260227082707_0003_V`, `DJI_20260227083814_0001_V`.

### A caveat that must be stated with the table

**Giraffe has only 2 template videos.** Representing it in validation therefore leaves exactly **one**
video to fit its template, and puts **561 crops (28% of the validation set)** on the species whose
template is weakest. Meanwhile **zebra — the species the layer/facet finding actually hinges on —
contributes only 201 crops (10%)**.

Because overall 2-way is crop-weighted, the criterion is dominated by elephant (38%) and giraffe (28%).
This is a consequence of the video counts, not a choice, and the seed was **not** re-rolled to improve
it — re-rolling after seeing the shape of a split is precisely the error being corrected. The macro
(per-species mean) 2-way is reported alongside so this dependence is visible; if micro and macro
disagree about the winner, that must be reported rather than resolved silently.

---

## B. The validation sweep

*Pending the run in §6. Produced verbatim in this format by `sweep_val.py`, and written to
`sweep_val.json`.*

| configuration | 2-way | App-4 | Geo | Prior | macro | Ele | Gir | Rhi | Zeb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L12/key/448 | | | | | | | | | |
| L16/key/448 | | | | | | | | | |
| L18/key/448 | | | | | | | | | |
| L24/key/448 | | | | | | | | | |
| L12/token/448 | | | | | | | | | |
| L24/token/448 | | | | | | | | | |
| L12/token/224 | | | | | | | | | |
| L24/token/224 | | | | | | | | | |

Templates for every row are built from the **23 fit videos only**; the 10 validation videos contribute
nothing to any template.

---

## C. Selected configuration

*Pending.*

```
Selected:         <config>
Selection metric: overall (crop-weighted) validation 2-way accuracy
Ties:             lower resolution, then lower layer
```

---

## D. Final evaluation of the selected configuration

Templates rebuilt from **all 33 template videos** (1,500-crop sample, exactly as the method operates),
then evaluated **once** on the 20 evaluation videos.

*Pending. Format as requested:*

```
Walking:
  appearance axis given =        full heading =        @30 =        median =        flank =    / coverage
Stationary:
  appearance axis given =        full heading =        @30 =        median =        flank =    / coverage
Manual zebra:
  appearance axis given =        full heading =                                     flank =    / coverage
```

> **Note on `@30` and `median`.** These columns were removed from `metrics()` in July (`165eb2f`) as
> redundant — `acc@45` is bit-identical to directed-candidate accuracy on all 14 rows, because the
> nearest of four ~orthogonal candidates is always within 45°. They have been restored for this
> appendix. **The two columns use different references:** angular error is against the *continuous*
> motion direction; `sign` is against its *nearest candidate*. That is why the locomotion row scores
> 100% sign yet ~9° median error — the 9° is the quantisation gap itself. See `PAPER_VERIFICATION.md`
> §1.

If the winner is **L24/token/224**, the downstream tables (ablation, per-species, viewing-angle,
angular-threshold) are unchanged and only need to be confirmed to reproduce. If another configuration
wins, all four are regenerated from that configuration.

---

## E. How the MLP was trained

> **The MLP was trained on frozen DINOv3 features — the concatenation of the CLS token and the
> mean-pooled patch tokens (2,048-d for ViT-L/16) — with the motion-derived allocentric angle α as the
> only supervision, regressed as (cos α, sin α) under a cosine loss, on the training videos of the
> same video-level split (the 20 evaluation videos are never seen).**

Details, for the manuscript sentence:

| | |
|---|---|
| input | `extract_features.py:115` — `torch.cat([out[:, 0], patch.mean(1)])`, i.e. `[CLS ‖ mean-pooled patches]` from the frozen backbone's `last_hidden_state`. The backbone is never fine-tuned. |
| target | `train_head.py:72` — `Y = (cos α, sin α)`, where α is the **allocentric** angle of the motion-derived heading. Not an image angle: a crop determines orientation relative to the viewing ray, so regressing an image angle would ask the network for something the pixels do not contain. |
| loss | `train_head.py:121` — `1 − (p · Y)`, a cosine loss, i.e. angular error |
| supervision source | the free motion labels — **no human orientation annotation** |
| split | `train_head.py:89` — the same `video_split(seed=0)`; the fingerprint is stored in the checkpoint (`train_head.py:159`) |
| scoring | predicted α snapped to the nearest of the box's 4 candidate faces, giving the same 4-way question as the training-free probe, plus raw allocentric angular error |

So "uses supervision" should be replaced with the sentence above. The important clarification is that
the supervision is the **motion-derived α**, not a human label — the MLP is a supervised *baseline
against the same free labels*, which is what makes it comparable to the training-free method.

---

## 6. How to run it

Two GPUs, two jobs, in parallel — they are independent.

### Get the code across

Locally (three files changed: `split.py` +`template_val_split`, `sweep_val.py` new,
`experiments.py` angular columns restored):

```bash
git push fork wildbox_detany3d
```

On the cluster:

```bash
cd /storage3/3DOM/vshukla/DetAny3D && git pull origin wildbox_detany3d
```

### Job 1 — the validation sweep  **[A40, ~10 min]**

The A40 is the better fit: two of the three passes are at 448 px, where it is substantially faster.

```bash
conda activate /storage3/3DOM/vshukla/envs/dinov3
PYTHONUNBUFFERED=1 python -m tools.heading.sweep_val \
    --crops data/heading/crops.npz \
    --out   data/heading/REPORT_v3/sweep_val \
    --device cuda --dtype fp32 --batch 64 \
    2>&1 | tee data/heading/sweep_val.log
```

Eight configurations, but only **three** model passes over the fit set and three over validation —
forward hooks capture every layer in a `(facet, size)` group from one forward.

Check before it starts scoring: `FIT 23 videos 4966 crops` / `VALIDATION 10 videos 2028 crops`, and
the per-species table matching §A exactly. If it does not, stop — the split has moved.

### Job 2 — the final evaluation, incumbent configuration **[V100, ~40 min]**

Start this at the same time. It regenerates the full table **with the restored `median` / `@30` /
`coverage` columns**, which `REPORT_v3` does not have.

```bash
conda activate /storage3/3DOM/vshukla/envs/dinov3
PYTHONUNBUFFERED=1 python -m tools.heading.experiments \
    --crops       data/heading/crops.npz \
    --stand-crops data/heading/crops_stand.npz \
    --human-crops data/heading/crops_human.npz \
    --out         data/heading/REPORT_v4/exp \
    --layer 24 --facet token --size 224 \
    --device cuda --dtype fp32 --batch 96 \
    2>&1 | tee data/heading/exp_v4.log
```

### Then

- **If the sweep selects L24/token/224** — Job 2 *is* section D. Nothing further to run.
- **If it selects anything else** — rerun Job 2 once with `--layer/--facet/--size` set to the winner,
  into `REPORT_v5`, and regenerate the four downstream tables from it.

### Send back

```bash
cat data/heading/REPORT_v3/sweep_val/sweep_val.json
cat data/heading/REPORT_v4/exp/results.json
```

That is everything needed to fill B, C and D.
