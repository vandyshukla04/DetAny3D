# 3D Animal Heading from Locomotion, 3D Cues and DINOv3
### Method, metrics, and results
*What we did, what each metric means, and the measured numbers.*

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
`e1 ⟂ up` deterministic, `e2 = up × e1`. Azimuth of a horizontal direction `d` (its compass bearing
*within* the ground plane — this **is** the heading, expressed as one scalar):

```
azimuth(d) = atan2(d·e2, d·e1)
```

**Heading candidates.** Project the two horizontal box axes onto the ground plane to get unit
directions `a`, `b`. Their four signed directions are the candidate headings:

```
H = { +a, −a, +b, −b }              ĥ ∈ H
```

The box gives the two *axes*; it does not say which of the four directions is the front. The
predicted heading is therefore **one of these four discrete candidates** — we select a direction, we
do not regress a free angle.

**Allocentric angle α** (per instance). `r` = horizontalised camera→animal ray, `s = up × r`,
so `(r, s, up)` is right-handed:

```
α = atan2(ĥ·s, ĥ·r)         h = cos α · r + sin α · s
```

α is the animal's orientation **relative to the viewing ray** — the quantity a single crop
determines, and exactly DetAny3D's `alpha` (the allocentric/observation angle). Because ĥ is one of
four discrete candidates, α is **discrete per frame**, not continuous.

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

**Score of a candidate.** The species template `T` is a (B, D) matrix — the mean anatomical profile,
rump→head, over the locomotion-labelled training crops. A candidate's profile `P` is scored against
it after removing, from **both**, the component shared across bins (the "animal-ness" every patch
carries), so that only the along-body variation is compared:

```
P̄   = Σ_b w_b P[b] / Σ_b w_b            weighted mean of P over its OCCUPIED bins   (w_b = patch count)
T̄   = (1/B) Σ_b T[b]                    UNweighted mean of T over its B bins
Tᶜ  = T − T̄ ,   T̂ᶜ = Tᶜ / ‖Tᶜ‖_F        Frobenius norm over ALL bins and channels (one scalar)

S(candidate) = (1/Σ_b w_b) Σ_b w_b ⟨ P[b] − P̄ , T̂ᶜ[b] ⟩
```

Three details that fix the formula exactly (verified against `template.py`):
- **`P̄` is patch-count-weighted** over occupied bins; **`T̄` is unweighted** over all B bins.
- **The hat is a single Frobenius normalisation of the whole centred template** (over all bins and
  channels together), **not per bin**.
- **The crop profile `P` is left un-normalised.** Its magnitude is the along-axis *contrast*, which
  is itself evidence — a true body axis has a strong head→rump gradient, a cross-body axis is nearly
  flat — so keeping it lets the true axis win on alignment *and* contrast. Only the template carries
  the hat.

The profile of the **opposite** candidate is the exact reverse of this one (the bins read back to
front), so the four candidates cost two profile computations.

**Selecting the heading.** Geometry proposes the axis; appearance resolves the sign:

```
ĥ = argmax_{candidate ∈ H} [ S(candidate) + G(candidate) ]

G(candidate) = 0     if candidate is one of the two directions of the PROPOSED axis
             = −∞    otherwise                       (a hard mask, not a finite bonus)
```

i.e. the maximisation is **restricted to the two directions of the geometrically-proposed axis**, and
appearance chooses only between them. The **proposed axis** is the horizontal box axis with the
larger ground-plane-projected extent:

```
proposed axis = argmax_c  d_c · ‖ (I − up upᵀ) R[:,c] ‖
```

— the box's dimension along local axis `c`, scaled by how much of that axis survives projection onto
the ground plane. (Geometry is measurably better at the axis than appearance is; §Design choices.)

---

## 5. Method summary

```
1.  LOCOMOTION names the head    a walking animal's world velocity is its heading — free, signed
2.  TEMPLATE = the mean rump→head profile of those crops, per species   (nothing is trained)
3.  DINOv3 TRANSPORTS it         score the box's 4 candidate directions against the template
4.  GEOMETRY proposes the axis   the hard mask G above; appearance resolves only the sign
5.  α → the VIEWPOINT TAG        which flank, how broadside, face or rear
```

Nothing is trained: the template is an average, and scoring, axis selection and the viewpoint tag are
all closed-form.

---

## 6. Design choices (each measured)

- **DINOv3 layer 24, token facet, 224 px.** We swept transformer layer, facet (block-output *token*
  vs the attention *key* projection) and input resolution. The last-layer token output is best; on
  zebras it is the difference between chance and a working cue (head-vs-tail 52% at other settings,
  79–81% here). 224 px and 448 px are within a point, so 224 is used (4× cheaper). DINOv3's dense
  features are the frozen backbone throughout — nothing is fine-tuned.

- **Score the gradient, not the descriptors.** Every patch on an animal is first of all "animal", so
  an un-centred cosine between a candidate's profile and the template sits near 0.9 for *any* axis,
  including a wrong one — a flat cross-body profile scores ~0.81 of the true axis'. Centring both
  profiles across bins removes that shared component and compares only the head→rump gradient; a
  flat profile then scores ~0.

- **Geometry proposes the axis; appearance resolves the sign.** The box's longer horizontal axis is
  the body axis on **87%** of frames (zebra 74%), whereas appearance is unreliable at *choosing an
  axis*. Restricting the search to the two ends of the proposed axis and asking appearance only for
  the sign is the hard mask `G` in §4.

- **SAM3 instance masks.** The frames are herds (3.4 animals/frame). An appearance-only foreground
  mask pools a neighbour's patches into the target's profile; on zebras — the herd species — this
  put head-vs-tail at chance (52%). A per-instance SAM mask restricts the profile to the target and
  removes it. The mask→track join is verified by mask-centroid-inside-own-box (accepted masks sit at
  0.13 of a box half-width; a neighbour's would sit ~5×) rather than assumed.

---

## 7. Metrics

The predicted heading is one of four discrete box-axis directions (§4). We evaluate what the method
*decides*, not the box it inherits:

- **Sign accuracy** — of the two directions of the true body axis, is the correct **head end**
  chosen? Chance 50%. This is the head-vs-tail decision DINOv3 makes.
- **Flank accuracy** — is the **camera-facing side** (LEFT/RIGHT) named correctly? Computed only
  where a flank is actually visible, `|sin α| ≥ 0.35`; elsewhere the animal is head-on and there is
  no flank to name. This is the tag re-ID consumes.
- **|sin α| bands** — sign and flank against how broadside the animal is, so the *distribution* of
  errors is visible (head-on / oblique / broadside).

**Why not an angular (azimuth) error.** Azimuth *is* the heading (a compass bearing in the ground
plane), but the azimuth *error* does not isolate the method. Because the heading is quantised to the
box's face normals, a correct prediction still inherits the box-axis error and a wrong one is a ~180°
flip — nothing between — so acc@45° ≡ sign accuracy, adding no information. On the human check it is
worse than uninformative: the annotator labelled *which box face is the front*, so the reference
heading **is** a box-face direction from the very set the prediction chooses from, and azimuth error
can only read 0° (right face) or 180° (wrong face). Sign and flank are the metrics that mean the same
thing across all three test sets. *(For context, the box axes themselves sit a median 9.1° from the
motion reference — the ceiling any face-selection method inherits.)*

The **reference** differs by test set: on walking animals it is the animal's own motion direction; on
the stationary transfer set it is the motion direction from either side of a `WALK→STAND→WALK` stop
(§8); on the human set it is a person's face annotation.

---

## 8. Results (`L24/token/224`, SAM masks, held out by video, nothing trained)

### 8.1 Walking animals — held-out videos (n = 5,768; appearance rows 5,767, see below)

| setting | sign | flank* |
|---|---:|---:|
| random sign | 46.9% | — |
| locomotion reference *(= the label)* | 100% | — |
| appearance, oracle axis *(axis given; isolates the sign)* | 93.5% | — |
| **full method** | **87.4%** | **95.0%** |

Per species, full method — sign: rhino 91.9% · elephant 77.3% · zebra 73.8% · giraffe 100% (n=30).

*One crop of 5,768 is exactly end-on: its body axis projects to a point, there is no profile, and the
appearance rows **abstain** on it (n=5,767). The geometry-only rows keep all 5,768. An abstention,
not a dropped sample.*

### 8.2 Stationary animals — the transfer test (n = 1,404)

The template is built only from walking crops; here it is scored on **standing** animals, on
held-out videos, with references from `WALK → STAND → WALK` stops whose before/after headings agree
within 30° (the animal provably did not turn). Labels: 4,833 bridge crops exist; 3,429 fall in
training videos and are discarded; **1,404** are held out and tested.

| | sign | flank* |
|---|---:|---:|
| walking (§8.1) | 87.4% | 95.0% |
| **standing** | **69.1%** | **89.4%** |
| difference | −18.3 pts | −5.6 pts |

Per species (standing, sign): zebra 87.0% · rhino 69.5% · elephant 58.3% · giraffe 100% (n=10).

### 8.3 Human check — grazing zebras, human face-locks (n = 5,542)

The only reference not derived from our own motion labels, and the only test on *committed* grazers
(the animals the bridge test cannot reach). Two videos of grazing zebras, hand-annotated, held out of
training permanently. The face-locks are joined to the boxes **by world direction, not face index**:
papersubdata carries the raw box rotations while the annotations index the sign-aligned (canonical)
ones, and **16.9%** of instances differ by a 180° rotation, so an index copy would invert head/tail
on one instance in six. The world-direction join matches to >0.999 on all 11,084 instances.

| setting | sign | flank* |
|---|---:|---:|
| **full method** | **92.5%** | **97.1%** |

By how broadside the animal is:

| \|sin α\| | n | sign | flank |
|---|---:|---:|---:|
| 0.00–0.35 (head-on) | 261 | 36.0% | 36.0% |
| 0.35–0.70 (oblique) | 579 | 86.4% | 86.4% |
| 0.70–1.00 (broadside) | 4,702 | 96.4% | 98.4% |

On the broadside frames (85% of the set — what re-ID needs) the camera-facing side is named on 98.4%.
The errors concentrate in the head-on band, where `|sin α|` is small, the body axis barely projects,
and no flank exists to name.

### 8.4 Ablation (walking set; each row removes one component)

| | sign |
|---|---:|
| full method | 87.4% |
| − centring | 87.3% |
| − SAM instance mask | 86.6% |
| − geometric axis prior | 83.6% |

Centring and the geometric axis prior are **redundant with one another**: each solves the *axis*
problem alone. Under the axis prior the two compared candidates are the two ends of one axis, whose
profiles are exact reverses; the shared (DC) component is symmetric under that reversal and cancels,
so centring has nothing left to remove. Centring matters when appearance must choose *between* axes,
which the axis prior removes.

---

## 9. Not yet done

- **DetAny3D integration.** The detector's `alpha` head (12 bins over 2π, a live loss) is currently
  fed a hardcoded `0.0` (`data_creator/wildbox.py:270`). Feeding it the recovered heading would let a
  detector emit heading natively at inference. 3D/BEV AP must not regress against the 5-seed baseline
  (**3.97 ± 0.65**).
- **The re-ID payoff.** Gallery accuracy with vs without side-consistent matching — the downstream
  metric the viewpoint tag exists to serve.

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
