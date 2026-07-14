# 3D Animal Heading from Locomotion, 3D Cues and DINOv3
### Method, experiments, and the progression of decisions
*A report. Findings are stated as measured; no conclusions are drawn beyond them.*

---

## 1. The goal

Give every detected animal a **3D heading in the world**, and from it the **viewpoint tag** a
re-identification system needs:

> *"we are looking at this animal's **left** flank, 0.85 broadside, slightly from behind."*

Wildlife re-ID is **side-dependent** — a zebra's left flank carries a different stripe pattern from
its right — so a left-flank query matched against a right-flank gallery entry is a guaranteed miss.
The heading is the means; the viewpoint tag is the end.

**Constraint held throughout: no human orientation labels.**

---

## 2. The three signals and how they relate

| signal | what it provides | what it cannot provide |
|---|---|---|
| **Locomotion** (from WildBox's 3D tracks) | the **name**: which end is the head. Signed, exact, free. | only while the animal walks — **16%** of frames |
| **DINOv3** (frozen, dense features) | **transport**: the same body part maps to the same place in feature space across frames | the name itself; and left/right — it is **mirror-blind** |
| **3D cues** (WildBox boxes + cameras) | the body **axis**, the **ground plane**, **left/right**, and the lift into **world** coordinates | the **sign** — box axis signs are arbitrary |

Each blind spot is covered by another signal. Appearance is asked for **front vs back only** (not a
mirror pair). Left/right is geometry: `left = up × forward`.

### 2.1 WildBox as a proxy for the detector's output

The pipeline needs three things from a monocular 3D detector: camera intrinsics, a ground plane, and
the box's **body axis**. Measured against DetAny3D's code:

| requirement | DetAny3D | WildBox |
|---|---|---|
| camera intrinsics | ✅ `pred_K`, genuinely predicted (`unidepth.py:611`); `deploy.py:281` already runs on it | ✅ `cameras.json` |
| ground plane | ✅ derivable from its dense metric depth map (`wrap_model.py:101`), computed every forward and then discarded | ✅ consensus ground normal |
| 3D position | ✅ `center_cam` (`train.py:336`) | ✅ per-frame world centres |
| **body axis** | ❌ **rotation head is untrained** | ✅ from the box |

So **WildBox substitutes for exactly the one component DetAny3D cannot supply.**

### 2.2 Why the box's orientation cannot be used directly (verified in code)

| layer | state |
|---|---|
| GT rotation | raw PCA/SVD fit; **axis signs arbitrary**. Measured: **12.4%** of consecutive frames of one animal flip an axis. |
| GT yaw | **hardcoded `0.0`** — `data_creator/wildbox.py:270` |
| `alpha` head (12 bins over 2π; heading-aware) | trained against that placeholder |
| 6D rotation head | **zero gradient** — `chamfer_loss` absent from `wildbox_final.yaml:54`'s `loss_list`; also 180°-blind |
| metrics (3D IoU, BEV AP, NHD) | **all flip-invariant** ⇒ published APs are **not** corrupted, but no box rotation is usable for front/back |

---

## 3. Dataset: WildBox (`papersubdata`)

**60 videos · 305 segments · 52,443 frames · 177,973 animal instances · 4 species**
(gazelles excluded).

| species | videos | segments | frames | tracks | animal instances |
|---|---:|---:|---:|---:|---:|
| elephant | 17 | 88 | 14,184 | 304 | 42,930 |
| rhino | 19 | 99 | 18,060 | 258 | 45,853 |
| zebra | 20 | 108 | 18,669 | 525 | 85,502 |
| giraffe | 4 | 10 | 1,530 | 28 | 3,688 |
| **total** | **60** | **305** | **52,443** | **1,115** | **177,973** |

Mean 172 frames per segment; **3.4 animals per frame** — the herd density that makes instance masks
necessary rather than optional.

### 3.0 What we actually used — **we do NOT use all 177,973 animal images**

| | elephant | giraffe | rhino | zebra | total | |
|---|---:|---:|---:|---:|---:|---|
| animal images in WildBox | 42,930 | 3,688 | 45,853 | 85,502 | **177,973** |  |
| **WALKING** → a free heading label | 7,229 | 1,671 | 10,914 | 5,740 | **25,554** | only a walking animal shows you which way it faces |
| crops kept (every 2nd frame) | 3,614 | 828 | 5,457 | 2,863 | **12,762** | adjacent frames of a walking animal are the same picture |
| → from the 40 training videos | 2,892 | 798 | 1,300 | 2,004 | **6,994** |  |
| → **used to build the template** | 587 | 187 | 285 | 440 | **1,499** | the template is a mean — nothing is trained |
| → **held out, walking: TESTED ON** | 722 | 30 | 4,157 | 859 | **5,768** | 20 videos the template never saw |
| | | | | | | |
| standing crops produced (bridges) | 2,211 | 103 | 1,741 | 778 | **4,833** | a POOL, not a test set |
| → from training videos: **NOT USED** | 1,650 | 93 | 1,216 | 470 | **3,429** | the template is built from *walking* crops only; these are discarded |
| → **held out, standing: TESTED ON** | 561 | 10 | 525 | 308 | **1,404** | **the transfer test** |

**In words.** WildBox holds **177,973** animal images (one animal in one frame). Only
**25,554** of them (**14%**) come with a free heading label — the ones where the animal is
**walking**, because a walking animal shows you which way it faces. The other **86%** are
standing still and tell us nothing for free.

After sub-sampling (two adjacent frames of a walking animal are the *same* measurement) and
holding out 20 of the 60 videos, the template is built from **1,499** crops, and every number
reported is measured on **5,768 walking** and **1,404 standing** crops from videos the
template never saw.

### 3.1 Coordinate facts (verified, not assumed)

| | |
|---|---|
| `centers`, `rotation_matrices` | **world** |
| `extrinsic` | 3×4, world → camera |
| `cameras.json` intrinsics | **full-res** (fx ≈ 1266, cx = 960, cy = 540) |
| `tracking_summary.bbox_2d` | **518-space** — × (width/518) = 3.707 to reach full-res |
| world scale | VGGT-arbitrary ⇒ every threshold is in **body-lengths**, never metres |
| verification | projecting `centers` through `cameras.json` → **(870.8, 577.7)**; the kitti 2D box centre → **(870.7, 580.6)** |

### 3.2 Frame population

| | frames |
|---|---|
| **stationary** | **133,809** (84%) |
| walking | 25,554 (16%) |

### 3.3 Viewing geometry (measured over all 12,762 crops)

| | |
|---|---|
| drone **elevation** above the animal's horizontal plane | median **18°** (p10 9°, p90 30°) |
| drone **azimuth** relative to the animal | full circle (−179° … 179°) |

The footage is **aerial-oblique, not top-down**. This is a precondition of the task rather than an
incidental detail: from directly overhead an animal presents its back and no flank at all, and the
viewpoint question would not arise. The 18° median grazing angle is what makes a flank visible, and
what `|sin α|` quantifies.

---

## 4. Formulae

**Ground basis (per segment).** `up` = consensus over box axes (axes only; signs are discarded).
`e1 ⟂ up` deterministic, `e2 = up × e1`. World azimuth of a horizontal direction `d`:

```
azimuth(d) = atan2(d·e2, d·e1)
```

**Allocentric angle α** (per instance). `r` = horizontalised camera→animal ray, `s = up × r`,
so `(r, s, up)` is right-handed:

```
α = atan2(h·s, h·r)          h = heading (world, horizontal)
h = cos α · r + sin α · s
```

α is the animal's orientation **relative to the viewing ray**, which is what a crop determines. It is
exactly DetAny3D's `alpha` (the allocentric/observation angle).

**Viewpoint tag — how visibility is computed (the deliverable).**
The animal's own left side is `left = up × h = cos α · s − sin α · r`, and the camera lies in the
`−r` direction from the animal. Substituting:

```
left · (−r) =  sin α      →   sin α > 0 : we are seeing its LEFT flank
                              sin α < 0 : we are seeing its RIGHT flank
   h · (−r) = −cos α      →   cos α < 0 : we are seeing its FACE  (walking toward us)
                              cos α > 0 : we are seeing its REAR  (walking away)
```

| quantity | meaning |
|---|---|
| **\|sin α\|** | how much **flank** we see. 1 = fully broadside, 0 = none. |
| **\|cos α\|** | how much **face or rear** we see. |
| sign of `sin α` | LEFT or RIGHT |
| sign of `cos α` | FACE or REAR |

`|sin α|` and `|cos α|` are the two components of one unit vector, so they trade off exactly: an
animal 0.95 broadside is necessarily 0.31 rear. A head-on animal reads `LEFT 0.08 / FACE 0.99` —
which is the honest statement that **no flank is visible**, not a failed prediction. Hence the
figures print the *weights*, not a binary side label: re-ID needs to know **how much** of a flank it
is looking at, not merely which one. **Flank accuracy is therefore reported only where
`|sin α| ≥ 0.35`** — where a flank exists to be named at all.

*Verified against the independent construction (`left = up × forward`, `forward · to_camera`) over
200 random cameras and headings: exact, 200/200. A sign error here would invert every re-ID match
while looking entirely plausible, because LEFT and RIGHT are equally common.*

**Motion label.** Smooth the world centres (moving average, k=9); take the displacement over a
window of W = 15 frames; project out `up`; normalise:

```
h_motion = normalize( (1 − up upᵀ)(C̃[t+W] − C̃[t]) )
speed    = ‖…‖ / body_length              body_length = median(max(dimensions))
```

**Anatomical profile** (the appearance cue). For a hypothesis *"the head is at face j"*, project every
foreground patch onto the image-space axis (opposite face → face j) to get `s ∈ [0,1]`, bin into B = 5
slices, and average within each bin:

```
P[b] = mean{ f_p : patch p in bin b },   b = 0 (rump) … B−1 (head)
```

**Score** (centred — see §6.2). With the species template `T`:

```
μ      = Σ_b w_b P[b] / Σ_b w_b                (w_b = patch count in bin b)
score  = Σ_b w_b ⟨ P[b] − μ , T̂c[b] ⟩ / Σ_b w_b        T̂c = centre(T) / ‖centre(T)‖
```

The profile of the **opposite** hypothesis is the exact reverse of this one, so 4 hypotheses cost 2
profile computations.

---

## 5. Method

```
1.  LOCOMOTION names the head            (walking frames only; free; 25,554 labels)
2.  the TEMPLATE is the mean profile of those crops, per species   (nothing is trained)
3.  DINOv3 TRANSPORTS it: score the 4 horizontal faces of a new box against the template
4.  GEOMETRY proposes the axis, appearance disposes the sign
5.  α → the VIEWPOINT TAG (which flank, how broadside, face or rear)
```

Nothing in step 2 is trained: the template is an **average**.

---

## 6. Progression of the work

### 6.0 Labels: locomotion (§ measured)

A walking animal's world velocity is its heading, and — unlike the box — it is **signed**.

| | |
|---|---|
| **labels produced** | **25,554** across 4 species / 60 videos, **zero human annotation** |
| per species | elephant 7,229 · rhino 10,914 · zebra 5,740 · giraffe 1,671 |
| **box-vs-motion agreement** (two independent signals) | median cos **0.984** after gating; **0.953** raw (rhino .979, elephant .969, giraffe .934, zebra .818). Random baseline = 0.707. |
| gates | displacement > 0.30 body-lengths per 15-frame window **and** cos(velocity, face normal) > 0.8 |
| "longest horizontal axis" shortcut picks the wrong axis | **13%** of the time (zebra **26%**) — so a sign-only cue is capped at 87% |
| visual check | the arrow lands on elephant trunks, rhino horns, giraffe necks, and grazing zebras' lowered heads (`preview_labels.py`) |

### 6.1 Hypotheses tested and rejected

| cue | result |
|---|---|
| GroundingDINO `"head"` prompt | **failed** — domain shift; grounds onto the whole animal. Failed at 149 px as well as 69 px. |
| Mask taper / "hindquarters are bulkier" | **failed** — view-dependent |
| Motion at inference | **rejected** — `velocities` never populated; grazing animals move 10–25% of a body length per segment |
| **Image-space** temporal smoothing | **made it worse**: 96.8% → 79.0%. The drone moves, so a stationary animal's *projected* angle jumps; smoothing it smooths the camera. |
| **Track-level splits** | **leaky**: reported 96.7% while the model visibly broke on unseen footage. All numbers below use a **video-level** split. |
| k-means over DINOv3 patch tokens | **16.6%** (below the 25% chance line). **Retracted as a valid negative** — it used global k-means across species, a 14×14 grid, the last layer only, and cluster centroids instead of correspondence. |

### 6.2 The DC-domination bug and its correction

**Symptom.** Head/tail on the true axis was **83.7%**, but the 4-way front-face choice was **39.4%**.
The foreground PCA showed a clean rump→head colour gradient while the *predicted* head landed in rump
colours.

**Cause.** Every patch on an animal is, first of all, "animal". So `cos(profile_bin, template_bin)`
≈ 0.9 **for any axis profiled along** — including the wrong one. A profile taken across the animal's
**width** is nearly flat but still scored high on that shared component. The discriminative signal is
the **variation along the axis**, not the absolute descriptors.

**Evidence in the figures:** the printed decision margins were `+0.007`, `+0.003`, `+0.000` — all four
hypotheses within ~0.005 of each other. The choice was noise in the third decimal.

**Correction.** Centre the profile and the template across bins ⇒ score the **gradient**.

| | uncentred | centred |
|---|---|---|
| true axis | +1.000 | **+0.260** |
| wrong axis (flat) | **+0.814** | **0.000** |
| reversed (180°) | +0.362 | **−0.231** |

**Falsifiable prediction made before the run:** *the margins go from ~0.005 to ~0.2, or the diagnosis
is wrong.* **Measured after: decision margin median = 0.1545 (≈30×).**

Two sub-corrections found while testing, both silent:
- the profile must **not** be unit-normalised — its magnitude *is* its contrast along the axis, which
  is evidence; normalising rescales a flat profile into amplified noise;
- centring must use **only the bins that carry evidence** — an empty bin is a zero row, and including
  it drags the centre toward the origin and lets the DC creep back in.

### 6.3 Geometry proposes, appearance disposes

Letting appearance choose the axis was letting it override geometry at the one thing it is bad at.
Restricting the hypotheses to the **two ends of the geometric body axis** (chance 50%) and asking
appearance only for the **sign** is reported as `GEO`; a soft bonus on that axis is reported as
`PRIOR`.

### 6.4 Instance masks (the herd problem)

Zebra head/tail sat at **52%** — exactly chance — while giraffe (never occluded) was at 100%. Zebras
are the herd species: crops contain several **overlapping** zebras, and an appearance-only foreground
mask pools a **neighbour's rump into the target's head bin**.

SAM3 masks resolve this. The join (`obj_<N>` ↔ track `N`; `frame_%06d.png` ↔ `frame_%06d.jpg`) was
**asserted, not assumed** — each mask's centroid is checked against the target's own 2D box:

| | centroid offset (1.0 = box edge) |
|---|---|
| accepted masks | **0.13** (p90 0.24) — on their own animal |
| rejected masks | **4.87** — five box-widths away, a *different* animal |

| species | mask hit-rate |
|---|---|
| rhino | 100% |
| giraffe | 100% |
| zebra | 94% |
| elephant | 91% |
| **all** | **96%** (2% rejected by the centroid check; 1% no `obj_` dir; 1% frame-stride) |

### 6.5 The DINOv3 configuration sweep

Two prior assumptions were tested and **both were contradicted by the data**:

| assumed | measured |
|---|---|
| a **mid-layer** would win (part semantics peak mid-network) | the **last layer (24)** wins |
| the **`key` facet** would win (Amir et al., *Deep ViT Features as Dense Visual Descriptors*) | the **`token`** facet wins |

The difference is not marginal — it is the entire zebra result:

```
zebra 2-way:   L12/key 52%    L24/key 52%    L12/token 52%    L24/token 79–81%
```

Every other configuration leaves zebra at chance; only the **last-layer token output** separates them.
(DINOv3's stated contribution, Gram anchoring, targets the quality of the *final* dense features; Amir
et al. studied the original DINO.)

**Full sweep, held out by video, SAM masks on** (chance: 2WAY 50%, others 25%):

| config | 2WAY | APP4 | GEO | PRIOR | eleph | giraffe | rhino | zebra (2way) |
|---|---|---|---|---|---|---|---|---|
| L12/key/448 | 85.1 | 73.7 | 79.8 | 80.6 | 74 | 100 | 94 | 52 |
| L16/key/448 | 84.7 | 75.1 | 79.5 | 80.3 | 74 | 100 | 93 | 53 |
| L18/key/448 | 83.7 | 70.7 | 78.4 | 79.2 | 70 | 100 | 93 | 52 |
| L24/key/448 | 87.9 | 77.7 | 81.8 | 83.0 | 83 | 100 | 96 | 52 |
| L12/token/448 | 87.4 | 76.4 | 81.4 | 81.0 | 77 | 100 | 96 | 52 |
| L24/token/448 | 93.1 | 81.7 | 86.7 | 87.2 | 82 | 100 | 98 | 79 |
| L12/token/224 | 87.2 | 75.5 | 81.2 | 80.2 | 76 | 100 | 96 | 53 |
| **L24/token/224** | **93.6** | **82.5** | **87.3** | **87.5** | 82 | 100 | 98 | **81** |

- **224 ≈ 448** (93.6 vs 93.1) — resolution buys nothing; 224 costs 4× less.
- **APP4 = 82.5%** vs the **supervised MLP baseline of 79.8%** on the identical 4-way metric — with
  **nothing trained**.

### 6.6 Track-level decoding (Viterbi)

**Hypothesis tested:** per-frame errors are independent, so integrating over a track in **world
azimuth** (a shortest path that permits genuine turns but penalises instantaneous 180° flips) should
cancel them.

**Result: it did not.**

| config | per-frame RAW | per-frame VITERBI |
|---|---|---|
| L12/key/448 | 79.8% | 79.9% (+0.1) |
| L24/token/224 | **87.4%** | 87.7% (+0.3) |

**Why (measured).** Per-track accuracy is **bimodal**, not noisy. On zebra under L12/key:

| | tracks <20% correct | 20–80% | >80% |
|---|---|---|---|
| zebra (n=72) | **46** | 12 | 14 |
| rhino (n=95) | 9 | 23 | 63 |
| elephant (n=27) | 5 | 10 | 12 |

The errors are **whole-track inversions**, not sporadic flips. A smoothness prior is designed to
*preserve* a coherent trajectory, so it cannot un-invert a track that is self-consistently backwards.

The inversions were removed **at the source** by the configuration change (§6.5), not by the decoder:

| | L12/key/448 | L24/token/224 |
|---|---|---|
| per-frame (2 candidates, chance 50%) | 79.8% | **87.4%** |
| zebra | 42.0% *(below chance)* | **73.7%** |
| elephant | 75.1% | 77.3% |
| rhino | 88.2% | 91.9% |
| giraffe | 100% | 100% |

### 6.7 Engineering faults found and corrected (all silent)

| fault | consequence | correction |
|---|---|---|
| `bbox_2d` (518-space) read as full-res | every crop cut from the **background**; a model that had never seen an animal reported **91.8% flank accuracy** | `Segment.scale` derives the ratio and `check_scale()` asserts it against the projected 3D centres |
| checkpoint carried no record of its split | a head trained under one split, scored under another, reported **94.3%** where the truth was **79.8%** | `split_fingerprint()` in every checkpoint; every evaluator calls `assert_matches()` or dies |
| `_final_norm` outside `torch.no_grad()` | the `token` facet and every layer past the first two configs **never ran** | whole forward under `no_grad` |
| `output_hidden_states=True` | materialised **all 25** hidden states (~7.7 GB/batch) to use one | forward hooks capture only the requested layers |
| `score_faces` called 3× per crop | 3× the descriptor cost | scoring separated from choosing (`choose()`) |
| **`np.load` returns a lazy `NpzFile`** | **every `z[key][i]` re-inflated the whole 259 MB member — 4,113 ms per crop, ~6.6 h to score the held-out set.** This, not the GPU, was the entire slowness. | `load_npz()` materialises once (4.1 s, then 0.001 ms); every consumer routed through it |

---

## 7. Test cases (all in `tools/heading/tests/`, 26 total)

| test | what it prevents |
|---|---|
| allocentric round-trip under a random camera | a sign slip that is invisible in the loss and fatal in the output |
| `α = 0` ⇔ walking directly away from the camera | α off by π, which nothing else would catch |
| **viewpoint tag vs independent geometry** (200 random cameras) | an inverted LEFT/RIGHT tag — which would look plausible, since both are equally common |
| reversing a hypothesis exactly reverses the profile | 2 of the 4 scores silently wrong (the code derives them by reversal) |
| centring makes a flat profile score exactly 0 | the DC bug returning |
| empty bins contribute **exactly** nothing | an occluded rump changing the score |
| profile magnitude is preserved | contrast (which is evidence) being normalised away |
| `load_npz` returns a dict, `CropSet` never holds an `NpzFile` | the 4,113 ms-per-crop bug returning |
| shared-vs-per-frame PCA basis (4.6× more stable) | the paper figure's central claim becoming decorative |
| a real 180° turn survives Viterbi; an injected single-frame flip is removed | a decoder that does only one of the two |
| synthetic `WALK → turn → WALK` is **rejected** by the bridge filter | unsafe stationary labels |
| real-data 518-scale check over 30 segments | the background-crop bug returning |

---

## 8. Results, as measured

**Setting:** held out by **video** (never by track or frame); the two human-locked zebra videos are
excluded from training permanently; SAM instance masks; `L24/token/224`; template = the mean profile
of walking crops (**nothing trained**).

| quantity | value | chance |
|---|---|---|
| head-vs-tail on the true axis (2-way) | **93.6%** | 50% |
| front face, appearance chooses the axis too (APP4) | **82.5%** | 25% |
| front face, geometry proposes the axis (GEO) | **87.3%** | 25% |
| front face, soft axis prior (PRIOR) | **87.5%** | 25% |
| supervised MLP baseline (for comparison; trained) | 79.8% | 25% |
| per-frame over held-out tracks (2 candidates) | **87.4%** | 50% |
| — zebra | 73.7% | 50% |
| — elephant | 77.3% | 50% |
| — rhino | 91.9% | 50% |
| — giraffe | 100% | 50% |

---

## 9. Not yet tested

- **Standing animals.** Every number above is measured on **walking** frames, because that is where
  motion labels exist. **133,809 frames (84%) are stationary and none of them have been tested.**
  A stationary test set has been built and not yet run: `bridges.py` yields **4,837** labelled
  stationary frames from `WALK → STAND → WALK` runs whose before/after headings agree within 30°
  (median agreement **0.98**); only 48% of stops pass that filter (zebra 29%).
- **Human face-locks.** 11,084 human-annotated instances on 2 videos of *grazing* zebras, held out of
  training. ⚠️ The join must go through the **world direction**, not the face index: papersubdata
  carries **raw** rotations while the locks index the **canonical** ones, and **16.9% (1,874/11,084)**
  differ by 180°.
- **DetAny3D integration.** `data_creator/wildbox.py:270` currently feeds the `alpha` head a hardcoded
  `0.0`. 3D/BEV AP must not regress against the 5-seed baseline (**3.97 ± 0.65**).
- **The re-ID payoff.** Gallery accuracy with vs without side-consistent matching.

---

## 10. Appendix — commands to test on further data (not part of the work above)

**Stationary animals (the covariate-shift test).** CPU for the labels, GPU for the scoring:
```bash
# local: build the stationary test set from WALK -> STAND -> WALK bridges
python -m tools.heading.bridges --root /mnt/d/3DBOX/papersubdata --out data/heading/bridges.npz
python -m tools.heading.extract_crops --labels data/heading/bridges.npz \
       --out data/heading/crops_stand.npz --stride 1
# ship crops_stand.npz to the cluster, then score with the SAME template as the walking crops
```

**Gazelles (the excluded 5th species).** Removing the exclusion tests whether the template transfers
to a species never seen:
```bash
python -m tools.heading.autolabel --root /mnt/d/3DBOX/papersubdata --out data/heading/labels_gaze.npz
# (EXCLUDED_GROUPS in papersub.py must be emptied first)
```

**Species transfer.** Train the template on three species, test on the fourth:
```bash
python -m tools.heading.sweep --crops data/heading/crops.npz --out data/heading/desc \
       --device cuda --batch 96          # per-species columns already report this per config
```

**The remaining WildBox archive** (`/storage3/3DOM/vshukla/sam3/wd_data/wildbox/archive`) contains
datasets not in `papersubdata` — e.g. `data202401KZebras`, `data202307KZebras`, `data202401KRhinos`.
They carry the same `vggt_results/` + `sam3_masks/` structure and could extend the evaluation, at the
cost of re-deriving the full-res intrinsics (the archive's `cameras.json` is 518-space, whereas
papersubdata's is full-res).
