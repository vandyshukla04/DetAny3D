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
Template-fit: 23 videos, 4,966 walking crops   (1,500 crops drawn; 1,499 contribute a profile -- see D.7)
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

Templates for every row built from the **23 fit videos only**; scored on the **10 validation videos**
(n = 2,027 — one crop abstains, being exactly end-on). The 20 evaluation videos take no part.

| configuration | 2-way | App-4 | Geo | Prior | macro | Ele | Gir | Rhi | Zeb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L12/key/448 | 75.5 | 60.2 | 67.7 | 68.3 | 71.4 | 69.4 | 92.9 | 77.2 | 46.3 |
| L16/key/448 | 77.1 | 62.3 | 69.3 | 69.1 | 72.6 | 72.9 | 93.2 | 78.0 | 46.3 |
| L18/key/448 | 75.5 | 60.5 | 67.7 | 64.9 | 71.4 | 70.1 | 93.4 | 75.8 | 46.3 |
| L24/key/448 | 78.4 | 61.6 | 70.6 | 67.4 | 73.5 | 76.3 | 93.2 | 77.8 | 46.8 |
| L12/token/448 | 74.9 | 60.6 | 67.4 | 63.2 | 71.3 | 67.7 | 91.2 | 79.1 | 47.3 |
| **L24/token/448** | **91.8** | 69.7 | 82.7 | 82.9 | **87.8** | 95.3 | 92.0 | 95.9 | 68.2 |
| L12/token/224 | 75.0 | 60.6 | 67.4 | 63.8 | 71.1 | 68.5 | 91.2 | 78.6 | 45.8 |
| L24/token/224 | 91.2 | 71.5 | 82.4 | 82.9 | 87.2 | 95.1 | 89.8 | 95.9 | 68.2 |

**The two headline findings survive selection on held-out validation data, and are not marginal:**

- **The last layer wins, decisively.** L24/token beats L12/token by **~17 points** (91.8 vs 74.9 at
  448; 91.2 vs 75.0 at 224). The mid-layer prior was wrong, and this is now demonstrated without the
  evaluation videos.
- **The `token` facet beats `key`, decisively.** At layer 24: 91.8 (token/448) vs 78.4 (key/448), a
  **13-point** gap. On zebra alone the gap is **68.2 vs 46.8** — below chance for `key`.

Both effects are far larger than the resolution effect, so the paper's central claim about layer and
facet is unaffected by which resolution is chosen.

---

## C. Selected configuration

```
Selected:         L24/token/448
Selection metric: overall (crop-weighted) validation 2-way accuracy = 91.8%  (n = 2,027)
Ties:             lower resolution, then lower layer  — not invoked
```

**This is not the configuration the manuscript uses.** The incumbent, L24/token/224, comes second at
91.2%.

### Reporting this honestly

The margin is **0.6 points on 2,027 paired observations** — roughly a dozen crops net, which is well
inside sampling noise. It would be easy to argue for keeping 224. Three reasons the report selects 448
anyway:

1. **The rule was fixed before the table was seen**, and says "highest overall 2-way", with a tie-break
   only for an *exact* tie. Adding a tolerance now — after seeing that 224 loses by less than it — is
   the same class of error as the original criticism. The rule stands.
2. **448 also wins the macro average** (87.8 vs 87.2), so the choice is not an artefact of the
   crop-weighting imbalance flagged in §A. That was the one thing that could have made the criterion
   untrustworthy, and it does not apply.
3. The manuscript's own claim — that 224 and 448 are "within a point" — is **confirmed** by this table
   (0.6 pt). What changes is only which of the two is selected, not the finding that they are close.

Note that **zebra (68.2) and rhino (95.9) are identical** at both resolutions; the entire 0.6-point
difference comes from giraffe (92.0 vs 89.8) and elephant (95.3 vs 95.1) — and giraffe is the species
whose template is built from a single video (§A). That is worth one sentence in the appendix, because
it means the selected resolution is being decided by the weakest species in the split.

**Consequence:** all appearance-dependent results must be regenerated at 448 (§D). The efficiency
argument for 224 can still be made in the text — as a deployment note, not as the selection criterion.

---

## D. Final evaluation of the selected configuration

Templates rebuilt from **all 33 template videos** (1,500 crops drawn, **1,499 contribute** -- D.7),
then evaluated **once** on the 20 evaluation videos.

### D.1 Reference run — the incumbent L24/token/224 (`REPORT_v4`, fp32)

Not the selected configuration, but run in parallel with the sweep and reported so the effect of the
change is visible. All values reproduce the manuscript.

```
Walking:      appearance axis given = 93.4%   full heading = 87.4%   @30 = 83.8%   median =  9.84°   flank = 95.0% / coverage 77.5%
Stationary:   appearance axis given = 79.3%   full heading = 69.1%   @30 = 66.0%   median = 15.16°   flank = 89.4% / coverage 69.2%
Manual zebra: appearance axis given = 94.3%   full heading = 92.5%   @30 = 92.5%   median =  0.00°   flank = 97.1% / coverage 95.3%
```

Quantisation ceiling (locomotion reference): walking median **9.08°**, @30 **94.1%**; stationary
**8.50°**, @30 **89.5%**; manual zebra **0.00°**, @30 **100%**.

### D.2 ⚠️ The manual-zebra angular columns are DEGENERATE — do not quote them

`median = 0.00°` and `@30 = 92.5% = sign` exactly, on every row. This is not a bug and not a
suspiciously good result. On the human set the reference is a **human face-lock**, i.e. the annotator
labelled *which box face is the front*. The reference direction is therefore **itself one of the four
candidates the method chooses from**, so the angular error can only ever read 0° (right face) or ~180°
(wrong face). Median is 0 whenever accuracy exceeds 50%, and `@30 ≡ @45 ≡ directed-candidate accuracy`.

This is the cleanest possible illustration of the point in reviewer comment 1:

| test set | angular reference | candidate reference | are they distinct? |
|---|---|---|---|
| walking | continuous motion direction | its nearest candidate | **yes** — 9.08° apart |
| stationary | continuous direction carried across the stop | its nearest candidate | **yes** — 8.50° apart |
| **manual zebra** | the human-locked **face** | the same face | **no** — 0.00° apart |

So the manuscript should report angular error for walking and stationary only, and state explicitly
that it is undefined-by-construction on the human set. Reporting a 0.00° median there would look like
an error to any careful reader — and reporting `@30` there is just directed-candidate accuracy under
another name.

### D.3 Selected configuration — L24/token/448 (`REPORT_v5`, fp32)

Templates rebuilt from all 33 template videos (1,500 drawn, **1,499 contribute** -- D.7), evaluated once on the 20 evaluation
videos. **This is the result of record.**

```
Walking:      appearance axis given = 93.0%   full heading = 87.1%   @30 = 83.5%   median =  9.90°   flank = 95.1% / coverage 77.5%
Stationary:   appearance axis given = 78.8%   full heading = 68.9%   @30 = 65.7%   median = 15.30°   flank = 91.6% / coverage 69.2%
Manual zebra: appearance axis given = 94.7%   full heading = 92.9%                                   flank = 97.6% / coverage 95.3%
```

(Manual-zebra angular columns omitted deliberately — degenerate by construction, §D.2.)

### D.4 What changed relative to the manuscript's L24/token/224

| | 224 | **448 (selected)** | Δ |
|---|---:|---:|---:|
| **walking** — appearance, oracle axis | 93.4 | 93.0 | −0.4 |
| **walking** — full heading (sign) | 87.4 | **87.1** | −0.3 |
| **walking** — flank | 95.0 | **95.1** | +0.1 |
| **walking** — median / @30 | 9.84° / 83.8 | 9.90° / 83.5 | +0.06° / −0.3 |
| **stationary** — full heading | 69.1 | **68.9** | −0.2 |
| **stationary** — flank | 89.4 | **91.6** | **+2.2** |
| **manual zebra** — full heading | 92.5 | **92.9** | **+0.4** |
| **manual zebra** — flank | 97.1 | **97.6** | **+0.5** |
| transfer gap (sign / flank) | −18.3 / −5.6 | −18.2 / **−3.5** | flank gap narrows |

**Every qualitative claim in the paper survives, and the deliverable improves.** Sign accuracy is
0.2–0.4 pt lower; **flank accuracy — the tag the downstream task actually consumes — is equal or better
on all three test sets**, and the walking→stationary flank gap narrows from 5.6 to 3.5 points.

Ablation (walking sign), ordering and interpretation unchanged, gaps slightly larger:

| | 224 | **448** |
|---|---:|---:|
| full method | 87.4 | **87.1** |
| − centring | 87.3 | **86.6** |
| − SAM instance mask | 86.6 | **85.6** |
| − geometric axis prior | 83.6 | **82.4** |

> One interpretation does need softening. At 224, removing centring cost 0.1 pt, supporting the claim
> that centring and the geometric axis prior are *redundant*. At 448 it costs **0.5 pt**, and removing
> the SAM mask costs **1.5 pt** rather than 0.8. The components are less redundant at higher
> resolution — sensible, since finer patches make both the along-body gradient and a neighbour's
> intruding patches more resolvable. State the ablation as measured; drop the strong "redundant"
> phrasing.

Per species (walking, sign): elephant 77.1 · giraffe 100 (n=30) · rhino 91.8 · **zebra 72.1** (was
73.7). Zebra is the one species that is *worse* at 448 — the only place the resolution change costs
anything material, and worth a sentence given zebra carries the layer/facet argument.

Viewing-angle bands, walking: 74.4 / 91.5 / 90.1 (sign), 79.6 / 94.8 / 95.4 (flank).
Manual zebra: **the broadside band is identical at both resolutions — 96.4 sign / 98.4 flank on
n=4,702**. The headline "98.4% on the frames that matter" is unchanged by the selection.

### D.5 The main results table, complete — every cell at the selected configuration

Assembled from `REPORT_v5` plus the geometry-only Random-sign row (computed with the same 50-seed
estimator `experiments.py` uses; that row needs no network, so it is resolution-independent).

| setting | n | Med. err. ↓ | ≤30° ↑ | Heading ↑ | Flank (cov.) ↑ |
|---|---:|---:|---:|---:|---:|
| *Walking — displacement-derived reference* | | | | | |
| Random sign *(geometry axis)* | 5,768 | 91.5 | 43.7 | **45.7** | — |
| Candidate oracle *(best of four)* | 5,768 | 9.1 | 94.1 | 100.0 | — |
| Appearance *(axis given)* | 5,768 | **9.5** | **88.4** | 93.0 | — |
| **Full method** | 5,768 | 9.9 | 83.5 | **87.1** | **95.1** (77.5) |
| *Stationary — reference transferred across a stop* | | | | | |
| Random sign *(geometry axis)* | 1,404 | 89.6 | 39.9 | **42.4** | — |
| Candidate oracle *(best of four)* | 1,404 | 8.5 | 89.5 | 100.0 | — |
| Appearance *(axis given)* | 1,404 | **10.8** | **73.8** | 78.8 | — |
| **Full method** | 1,404 | 15.3 | 65.7 | **68.9** | **91.6** (69.2) |
| *Manually labelled (zebra)* | | | | | |
| Random sign *(geometry axis)* | 5,542 | — | — | **49.2** | — |
| Appearance *(axis given)* | 5,542 | — | — | 94.7 | — |
| **Full method** | 5,542 | — | — | **92.9** | **97.6** (95.3) |

**Seven cells in the manuscript's version of this table were stale** — carried over from `REPORT_v2`
(224 px, and a *single* coin draw rather than the 50-seed average):

| cell | manuscript | corrected |
|---|---:|---:|
| walking Random sign — med / ≤30 / Heading | 80.4 / 44.8 / 46.9 | **91.5 / 43.7 / 45.7** |
| stationary Random sign — med / ≤30 / Heading | 90.4 / 39.8 / 42.2 | **89.6 / 39.9 / 42.4** |
| manual-zebra Random sign — Heading | 49.5 | **49.2** |

The walking median moves the most (80.4 → 91.5), and that is correct rather than a bug. The
random-sign error distribution is **trimodal**: ~46% near 9° (coin right), ~46% near 171° (coin wrong),
and ~8% near 90° — the crops where the head lies on the *other* axis, which no coin can reach. The
median therefore falls inside the sparse 90° cluster and is highly sensitive to where the 50th
percentile lands. A single draw put it at 80.4°; the 50-seed pooled estimate is 91.5°, which is also
what one expects a priori.

Two consistency checks on the corrected row: **`≤45` equals Heading exactly** on all three random-sign
rows (45.7 / 42.4 / 49.2), the same identity that holds in every other row; and on the manual-zebra set
**`≤30` also equals Heading** (49.2), the degeneracy of §D.2 — which is why `—` is the right entry
there.

### D.6 The viewing-angle table, complete at the selected configuration

All from `REPORT_v5`. **This replaces the mixed 224/448 version wholesale.** Band membership is by
`|sin α|` computed from the *true* heading, so the bin boundaries and counts depend only on geometry —
but the walking counts change from the manuscript's because **the 448 configuration has no abstention**
and evaluates all 5,768 walking observations (§D.7), where 224 evaluated 5,767.

| set | band | n | Heading ↑ | Flank ↑ |
|---|---|---:|---:|---:|
| **Walking** | [0, 0.35) head-on | 1,297 | 74.4 | — |
| | [0.35, 0.70) oblique | 2,224 | 91.5 | 94.8 |
| | [0.70, 1] broadside | 2,247 | 90.1 | 95.4 |
| **Stationary** | [0, 0.35) head-on | 432 | 57.2 | — |
| | [0.35, 0.70) oblique | 495 | 82.4 | 90.7 |
| | [0.70, 1] broadside | 477 | 65.4 | 92.5 |
| **Manual zebra** | [0, 0.35) head-on | 261 | 34.1 | — |
| | [0.35, 0.70) oblique | 579 | 90.8 | 90.8 |
| | [0.70, 1] broadside | 4,702 | 96.4 | 98.4 |

**The head-on flank cell is `—` by definition, not by omission.** `experiments.py` does print a number
there (79.6 walking, 57.4 stationary, 34.1 manual zebra), because it computes band flank over *every*
crop in the band rather than over visible ones. But the method defines flank accuracy only where
`|sin α| ≥ 0.35`, and in the head-on band **no crop satisfies that** — so the printed value is the
agreement rate on frames where no flank exists, which is a different quantity. It must not be placed in
a column headed "flank accuracy".

**Reconciliation — every headline number is recovered from these bands**, which is the check that the
table is internally consistent rather than transcribed:

| set | Σ band n | reported n | coverage from bands | reported | flank from bands | reported |
|---|---:|---:|---:|---:|---:|---:|
| walking | 5,768 | 5,768 | 77.51% | 77.5 | **95.10** | 95.1 |
| stationary | 1,404 | 1,404 | 69.23% | 69.2 | **91.58** | 91.6 |
| manual zebra | 5,542 | 5,542 | 95.29% | 95.3 | **97.57** | 97.6 |

(coverage = the [0.35,1] bands as a fraction of n; flank = their n-weighted mean.)

Two things worth a sentence in the text: the manual-zebra head-on band is **34.1%, below chance** — the
honest signature of a band where the body axis barely projects and there is nothing to read; and the
**broadside band is unchanged at 96.4 / 98.4 on n = 4,702** between 224 and 448, so the headline claim
about the frames that matter does not depend on the configuration change.

### D.7 Template accounting — the 1,499 vs 1,500 discrepancy, resolved

**The manuscript is right; this report's earlier wording was wrong.** The two records were describing
different quantities.

| species | template crops |
|---|---:|
| elephant | 587 |
| giraffe | 187 |
| rhino | 285 |
| zebra | 440 |
| **total** | **1,499** |

**Identical in `REPORT_v4` (224 px) and `REPORT_v5` (448 px)** — the fit indices are the same (seed 0,
same split) and the same crop fails at both resolutions, so the configuration change does not touch
this table.

The mechanism, from `template.py`:

```python
prof, cnt = axis_profile(g, fg, it.face_uv[t], it.face_uv[it.y_face], self.cfg.bins)
if not cnt.sum():
    return                      # <- returns WITHOUT incrementing self.n
...
self.n[sp] += 1
```

`--fit 1500` draws exactly **1,500** crop indices. `n_fitted` counts only those that produced an
occupied profile, and one crop in the draw is exactly end-on — its body axis projects to a point, so
`axis_profile` yields no occupied bins and it contributes nothing. Hence **1,500 drawn, 1,499
contributing**.

**Action: change this report's wording, not the manuscript's table.** The manuscript's 1,499 is the
number that belongs in a data-accounting table, because it is the number of crops that actually
entered the template. Where the sampling procedure is described, say "1,500 crops sampled, of which
1,499 yield a profile" so the two numbers are never again in apparent conflict.

### D.8 Two caption additions

The `—` entries in the manual-zebra block are currently unexplained, and a reader who notices that
`≤30` would equal Heading there will (correctly) wonder why:

> Angular metrics are omitted for the manually labelled zebra set: the annotator labels *which box face*
> is the front, so the reference is itself one of the four candidates and the angular error is
> degenerate (0° or ~180°).

And since Random sign is now an averaged estimator whose expectation is not 50%:

> Random sign flips one independent sign per observation, restricted to the two directions of the
> geometry-proposed axis, averaged over 50 seeds; its expectation is
> 0.5 × P(head on the proposed axis), not 0.5.

### D.9 One incidental change worth a footnote

**At 448 there are no abstentions: n = 5,768 on every row.** At 224 exactly one crop was exactly end-on,
its body axis projected to a point, and the appearance rows reported n = 5,767. At 448 that crop has
enough patches to form a profile. If the manuscript explains the 5,768 → 5,767 discrepancy, that
sentence is no longer needed for the selected configuration.

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

## Summary — what the manuscript must change

| # | change | why |
|---|---|---|
| 1 | Appendix D: report the **validation** sweep (§B), not the evaluation-set sweep | the criticism |
| 2 | Selected configuration becomes **L24/token/448** | the predefined rule |
| 3 | All appearance-dependent numbers → `REPORT_v5` (§D.3, §D.4) | configuration changed |
| 4 | "40 non-evaluation videos" → "40, of which **33** contribute walking anchors" | 7 yield no crops |
| 4b | Main table: refresh the **Random sign** rows (7 stale cells, §D.5) and add the two caption sentences (§D.8) | carried over from REPORT_v2 at 224 px with a single coin draw |
| 4c | Replace the **viewing-angle table wholesale** with §D.6 | current version mixes 224 and 448; walking counts change 5,767 → 5,768 |
| 4d | Template accounting stays at **1,499** — no change to the manuscript | §D.7: 1,500 drawn, 1,499 contribute a profile. Only the sampling *description* needs the extra clause |
| 5 | Drop the strong "centring and the axis prior are redundant" phrasing | gap is 0.5 pt at 448, not 0.1 |
| 6 | State that manual-zebra angular error is **undefined by construction**; report sign/flank only | §D.2 |
| 7 | Remove the 5,768→5,767 abstention sentence for the selected configuration | no abstentions at 448 |
| 8 | Replace "the MLP uses supervision" with the sentence in §E | ambiguous as written |
| 9 | Keep the efficiency argument for 224 as a **deployment note**, not a selection criterion | it is 4× cheaper and statistically indistinguishable — but that cannot decide the choice after the fact |

Unchanged: the method, the three-signal argument, the last-layer and `token`-facet findings (17 and 13
points on validation), the transfer result, and the 98.4% broadside flank figure on the human set.

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

### Job 3 — the final evaluation at the SELECTED configuration **[A40, ~1.5–2.5 h]**

The sweep selected **L24/token/448**, so this run is required: it is section D.

```bash
conda activate /storage3/3DOM/vshukla/envs/dinov3
PYTHONUNBUFFERED=1 python -m tools.heading.experiments \
    --crops       data/heading/crops.npz \
    --stand-crops data/heading/crops_stand.npz \
    --human-crops data/heading/crops_human.npz \
    --out         data/heading/REPORT_v5/exp \
    --layer 24 --facet token --size 448 \
    --device cuda --dtype fp32 --batch 48 \
    2>&1 | tee data/heading/exp_v5.log
```

448 px is 784 patch tokens against 196, so attention cost rises ~16× per image and the run is several
times longer than the 224 one. Use the **A40** (more memory and faster at this size) and `--batch 48`;
if it OOMs, drop to 32. Everything else — the split, the fit-set size, the crops — is identical to
Job 2, so the two runs differ **only** in resolution.

Templates are rebuilt from all 33 template videos, exactly as in the final method; the fit/validation
distinction used for selection is discarded at this point, as intended.

### Send back

```bash
cat data/heading/REPORT_v3/sweep_val/sweep_val.json
cat data/heading/REPORT_v4/exp/results.json
```

That is everything needed to fill B, C and D.
