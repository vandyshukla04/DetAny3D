<!-- ============================================================================================ -->
# NORTH STAR — THE FIXED GOAL

**Read this block before acting on ANY prompt about this project. It does not change.
If a request seems to narrow the project to one task, that is DRIFT — re-read this and restore scope.**

## The goal

Build **ONE elegant monocular model** that, from a **single aerial drone image**, outputs for every animal:

1. **a 3D box** — aerial monocular 3D wildlife detection
2. **its heading / orientation** — which way it faces, which side of it we can see

**BOTH ARE GOALS. Neither is a control, a constraint, or a fixed backdrop.**

- 3D detection is **NOT** a reference to preserve. **13.17 3D AP / 8.68 BEV@0.50 is a BASELINE TO BEAT.**
- Orientation is **NOT** a bolt-on auxiliary head. It is a **co-equal output**.
- "Detection must not move" is the **WRONG** framing and is banned. Every arm reports a **two-task
  scorecard** — (detection, orientation) — paired across 3 seeds. A detection GAIN is a headline result,
  not a side-effect.
- The two tasks are **coupled**: alpha's gradients reach the shared trainable FPN, and better depth and
  resolution sharpen the features alpha reads. **Whether a joint objective beats either task alone is the
  central research question.**

## The bet

**Monocular 3D detection is advancing rapidly because of general-purpose vision foundation models.**
DINOv3, monocular depth foundation models, SAM3-class segmenters. The thesis of this project is that
those models make aerial wildlife 3D detection tractable in a way it was not two years ago —
**so use them, and use them elegantly.** Prefer one well-chosen frozen foundation model plus a small
trained head over a pile of bespoke fixes. Elegance and ingenuity are requirements, not garnish.

## The purpose

**Wildlife monitoring from drones.** Count animals, support identification, and know whether a given
observation was good enough to use. Orientation/visibility is what answers that last question — it is why
viewpoint is in the model at all, and it is a monitoring deliverable, not an academic curiosity.

## The viewing geometry — MEASURED, and it rules things out

**This is NOT a nadir / top-down dataset.** Measured over all 23,137 labelled crops (`cam_elev` in
`crops*.npz`), camera elevation above the animal's horizontal plane:

| statistic | value |
|---|---|
| median elevation | **15.8 deg** |
| within 20 deg of nadir (>70 deg elev) | **0.00%** |
| within 10 deg of nadir (>80 deg elev) | **0.00%** |
| low / grazing (<20 deg elev) | **67.2%** |
| oblique (20-70 deg) | 32.8% |

The drone films animals **from a distance at a shallow angle** — it does not look down on them. That is
precisely why FLANK visibility is the natural signal: you see an animal's SIDE, not its back
(71.8% of instances have usable `|sin alpha| >= 0.35`).

**WHY there is no nadir — and why that is NOT a boundary on the solution.** The drone is being flown to
*monitor and recognise individual animals from different sides*. Nadir would show only backs and would
defeat the entire purpose. So the absence of nadir is a consequence of the DATA-COLLECTION GOAL, not a
limitation to design around. **The only fixed fact is: nadir is not the view.** Nothing else here is a
constraint on what methods may be considered.

**The regime is OBLIQUE AERIAL — high altitude, shallow depression angle, and ZOOM.**
⚠ [CORRECTED 2026-08-20] An earlier version of this section claimed ground-level monocular 3D detection is
"MORE relevant than aerial work". **That was an over-correction and is withdrawn.** This is neither nadir
NOR ground-level: the camera is genuinely airborne and high, viewing obliquely, usually zoomed in.

**The suspected crux, to be RESEARCHED not assumed: depth estimation degrades specifically when zooming in
from high altitude.** Under strong zoom the focal length is large, variable and effectively unknown
(measured: VGGT `fx` spans 10.9x pooled and 1.5x median *within one video*), which is precisely the regime
where the apparent-size depth cue becomes unidentifiable — a large far animal and a small near one project
identically. Whether that is the dominant mechanism, and what resolves it, is an open research question for
this project, not a settled claim.

**Do NOT over-constrain the search.** Ideas may come from anywhere — aerial/oblique perspective detection,
ground-level mono3D, photogrammetry, self-calibration, foundation models, equivariance, or elsewhere.
The one thing ruled out is assuming a top-down view.

## Settled — do NOT reopen

- **Metric scale is out of scope.** WildBox GT is per-segment scale-normalised (median depth = 1.0);
  metric depth/size is unidentifiable here. Everything we measure is scale-invariant by construction.
- **Orientation needs its own output + flip-sensitive grading.** NHD / 3D IoU / BEV are EXACTLY invariant
  to a 180 deg flip (proven: min BEV IoU 1.000 over 3000 boxes), so the existing pose loss can never teach heading.
- **Tracking is a label-generation dependency, not a runtime one.** The model is per-frame, permanently.
- **VGGT is not a runtime dependency** — it authored the dataset offline. A 3D box is not a reconstruction.
- **Do not retry:** Viterbi/temporal decoding of alpha (measured +0.3%, null); AM3D/CARLA external validation
  (E8, cut).

## How progress is judged

| task | baseline to beat | how it is graded |
|---|---|---|
| 3D detection | 3D AP 12.4 / BEV@0.50 8.68 / NHD 7.063 (depth = 84.5% of the error) | 3D AP, BEV@0.25/0.50, disentangled NHD, paired 3 seeds |
| orientation | chance — the train-transferred constant on that exact set | sign + flank, PER SPECIES, track-clustered, vs the train-transferred constant floor |

<!-- ============================================================================================ -->

# Aerial-wildlife monocular 3D detection, for orientation and visibility

> **STEP 0 ON APPROVAL (context-loss-proofing):** copy this file verbatim to
> `/home/shuklva/DetAny3D/AEROVIEW_PLAN.md`; add a pointer line to
> `~/.claude/projects/-home-shuklva-ovmono3d/memory/MEMORY.md`; write memory file
> `aeroview-monocular-viewpoint.md` pointing at that doc, `HEADING_PROJECT.md`, `METHOD.md`.
> (Supersedes the re-ID heading study previously here — closed with an honest null, documented in
> `wildlift-framework/REID_GEOMETRY_PROJECT.md` + `REID_RUNBOOK.md`.)

## Context

Wildlife monitoring needs **viewpoint and visibility** per animal: which side we see, how squarely, how
occluded. The system decomposes as

```
image → [monocular 3D detector] → 3D box → geometry (4 candidate axes) → DINOv3 template → α → viewpoint/visibility
```

The **right-hand side is done**: `DetAny3D/tools/heading/` derives everything from one allocentric angle α
(`viewpoint.py:59-71`), scoring **92.5% sign / 97.1% flank** on 5,542 human-locked instances, nothing
trained. It is validated using WildBox's 3D boxes as a **proxy for a monocular detector's output**.

> ### TERMINOLOGY - fixed, do not conflate (this error was made twice)
> **3D reconstruction != 3D box.** VGGT is a *reconstruction* and it built WildBox's boxes **offline, once**.
> It is **NOT** a runtime dependency of anything we run or build. The heading pipeline's actual per-detection
> inference inputs are exactly: **a 3D box (`centers`, `dims`, `rotations`), the camera, gravity (`up`), and the
> image crop** (`predict_crops.py:89,113`). A monocular 3D detector emits precisely that box - which is why
> WildBox's boxes are a valid stand-in for one. Never write "the pipeline needs a reconstruction".
>
> Stronger still: `tools/heading/visibility.py` shows the **3D box is not fundamentally required either** -
> `visible_flank(image_angle, up_cam, p_cam)` gets the flank from a **2D box + camera + gravity**, because
> `left = up x forward` and the box was "only ever a scaffold" for turning a predicted image angle into a 3D
> direction. That module is written and reasoned but **imported by nothing** (verified by grep) - an available
> fallback, not the operational path.

**The bottleneck is the detector**, and WildBox exists partly to show that: zero-shot 3D AP = **0.00** even
given GT 2D boxes; fine-tuned OVMono3D-LIFT reaches **13.17 macro 3D AP / 8.68 BEV@0.50**, while 2D is
essentially solved (**88.99 class-agnostic 2D AP@0.25**). So there are **two problems, one training run**:
adapt the detector to this domain, and make it emit what monitoring actually needs.

---

## Part 1 — Adapt the detector to the aerial wildlife domain

Generalist 3D detectors are built on four assumptions, each of which this domain violates. Each fix targets
one, so this is domain adaptation, not a grab-bag.

| assumption | reality here | fix |
|---|---|---|
| objects span many patches | median animal **142 px** natively [CORRECTED — and note **giraffe is SMALLER than elephant**, not larger] — but `SQUARE_PAD: 560` shrinks 1920→560, leaving **55% under 3×3 DINOv2 patches** | **native-resolution crop stream** (336 px at context ×2.5 ⇒ ~9.6 patches/animal, capped 32 crops/iter). Cheaper *and* sharper than upscaling the whole frame, since animals occupy 6.8% of width |
| depth ranges ~100× | within a segment depth spans **±10–20%**; absolute is unidentifiable (GT normalised to median 1.0, `prepare_wildbox_dataset.py:393-406`) | **predict depth relative to the scene**, `z = z̃·exp(δ)` anchored at the frame median, δ bounded. Model *starts at* the strongest trivial baseline and learns residuals |
| calibrated intrinsics | VGGT `fx` spans **10.9×** (1011–11055), varies **1.5× median and up to 3.4×** *within one video*, r=0.49 with true zoom, `cx,cy` pinned to image centre | **give K to the head as an input** — it is currently blind to box size and focal length (`roi_heads.py:366/430`), so it structurally cannot express `z = f·L/s`. Keep K in the *xy* decode (`roi_heads.py:802-803`), which works (NHD-xy = 2.1) |
| rigid objects, CAD-like size priors | deformable animals, 6 species; `DIMS_PRIORS` is **off**, so dims initialise at 1.0 vs GT ≈ 0.03 — a **33× error** | enable per-species dimension priors; normalise the disentangled losses by the GT cuboid diagonal (training currently uses absolute units while eval normalises) |

**Backbone.** Swap DINOv2 → **DINOv3** (stronger dense features, Gram anchoring) — as an ablation arm, not
an assumption. Backbone stays frozen; heads and neck train (~30–40 M params, the existing recipe).

**Free wins to fold in:** AMP is a silent no-op (`train_net.py:265` is a bare `losses.backward()`, no
autocast/GradScaler, so `SOLVER.AMP.ENABLED: True` does nothing); the loader ran at 54% GPU util; and
`roi_heads.py:317` computes `heights` from x-coordinates — a real bug that silently squares every RoI when
`SCALE_ROI_BOXES` is enabled.

---

## Part 2 — Make the outputs the ones monitoring needs

**Orientation is currently unsupervised.** DetAny3D's yaw target for WildBox is the literal constant `0.0`
(`wildbox.py:270`, "yaw placeholder"), so its existing α head (12 bins + residual,
`mask_decoder.py:116/165`; loss live at `train.py:170-182`) learns pure ray-angle. And WildBox's own box
rotations come from PCA with **arbitrary signs — the horizontal body axis flips on 10.9% of consecutive
annotated same-track frame pairs** (236,027 pairs) — so they are an
*unsigned axis* and must **never** be used as heading supervision.

1. **Supervise α for real**, from labels the heading project already produces (below). One-line change in
   the data creator, then a training run.
2. **Add a per-face visibility head** — 5-vector `[front, back, left, right, top]` of `n·v`, plus
   `effective = |n·v|·(1 − occ)`. The records already carry unused `visibility`/`truncation` slots
   (`prepare_wildbox_dataset.py:518-519`). The 3D box is retained precisely because **inter-animal occlusion
   needs 3D positions of every animal in the frame**.

Everything downstream — flank/end/strengths, `usable`, `view_code`, duty, coverage, exemplars, re-ID gating
— is a deterministic readout of α + visibility and is **lifted unchanged**.

**Why this is well-posed even though depth is not:** viewpoint consumes only the **body axis** from the box.
Depth carries 85–99% of all 3D error and is unidentifiable here — and is not on the critical path for
orientation. We fix depth as far as it is fixable (Part 1) and grade the system on what it is for (Part 2).

---

## Supervision (all already on disk)

| source | size | role |
|---|---|---|
| **Template teacher** (frozen DINOv3 heading pipeline: α + `margin`) | **177,973 animal instances across 52,443 frames**, 60 videos, 4 species (NO gazelle) | primary α target; `margin` = per-sample confidence weight |
| **Motion-derived heading** (`autolabel.label_track`, gated) | 25,554 images (14%) + 4,833 standing (`bridges.py`) | free and appearance-independent — the reason the student can *exceed* the teacher |
| **Geometric visibility teacher** (`n·v`, ray-OBB occlusion, SAM-mask overlap; deterministic) | any frame with boxes + camera + masks | visibility-head target |
| **Human face locks** (66 tracks, 2 zebra videos) | **5,542** instances [CORRECTED from 11,084] | **SECONDARY gold — "labels held out, IMAGES SEEN"** ⚠ both lock videos are 100% in `WildBox_train`, and 142 exact keys on 12 of the 66 tracks (1,105 instances = 19.9%) overlap the training label pool unless explicitly subtracted — see B2. The old claim "never trained on" is FALSE. |
| **Human visibility labels** | 512 frame-sets, 3 species | **gold** for the visibility head |

Splits: reuse `split.py` — **by video**, locked videos held out permanently, fingerprint-asserted
(`split_fingerprint:135`; a past leak reported 94.3% where truth was 79.8%).

---

## Is WildBox a valid benchmark for this? (yes, with stated limits)

WildBox's 3D GT is **per-segment scale-normalised** (median depth ≡ 1.0). That is fine here, because almost
everything this project measures is **scale-invariant**:

| quantity | invariant? | why |
|---|---|---|
| α, sign, flank | ✅ | angles unchanged by uniform scaling |
| per-face visibility `n·v` | ✅ | dot product of unit vectors |
| inter-animal occlusion (ray-OBB, mask overlap) | ✅ | ordering unchanged when boxes *and* camera scale together |
| depth ordering / relative depth | ✅ | ratios preserved |
| 3D IoU, BEV AP | ✅ | prediction and GT share the same normalised space |
| **NHD** | ✅ | normalised by the GT cuboid diagonal — scale-free *by design*, so it is the natural metric here |
| metric size / metric depth | ❌ | unidentifiable by construction — **out of scope, stated explicitly** |

So WildBox supports valid **within-benchmark** comparison. Its numbers must **not** be quoted against
KITTI/nuScenes, and no metric claim may be made.

**The real caveat is GT provenance, not scale.** Boxes are pseudo-labels (VGGT+SAM3+PCA) with known biases
(aspect ratios collapse to ~2.3:1; giraffe l/h measured **1.59** [CORRECTED from 2.5–2.9]; per-frame dims are
unstable — median within-track relative-std **0.21**, NOT "62% frozen" [CORRECTED]). So
*dimension* accuracy against them partly measures agreement with VGGT. The orientation/visibility targets are
**LESS** VGGT-derived than the boxes — human face locks (**5,542** [CORRECTED from 11,084]), human visibility
labels (512 frame-sets), and motion-derived heading (25,554, from trajectory direction — far more robust than
box shape) — **but they are NOT free of VGGT.** ⚠ [CORRECTED 2026-08-19] The earlier claim "the orientation and
visibility targets are not VGGT-derived" is WITHDRAWN: the allocentric BASIS in which every alpha is expressed
(`Segment.allocentric_basis`, `papersub.py:214`) rests on a per-segment `up` consensus fitted over VGGT box
rotations (`papersub.py:131-140`). See the VGGT circularity ledger in the AUDIT VERDICT section.
Part 2 still rests on firmer ground than Part 1 — the head/tail decision and the motion direction are genuinely
independent of VGGT — but the basis is a shared, fourth circularity channel.

**External validation for the Part-1 claims.** Because WildBox cannot adjudicate metric geometry, validate
the domain-adaptation fixes on an aerial benchmark with **real metric GT**: **AM3D-Real** (DJI M300, 40–80 m)
and/or **AM3D-Sim / CARLA-Drone**. Vehicles rather than wildlife, but the same aerial small-object,
narrow-depth regime — so they test resolution, relative-depth parameterisation and K-robustness independently
of VGGT's pseudo-labels. This replaces the weaker synthetic-only control.

## Evaluation

**Detection (Part 1):** 3D AP, BEV@0.25/0.50, and **disentangled NHD** {z, xy, dims, pose} — the last is what
shows *which* component a fix moved. Plus a **pairwise depth-ordering AUC** (scale/shift-free) and a
**K-perturbation Δ** (rescale `fx,fy` by U[0.7,1.4]: the baseline should collapse, the adapted model should
not — the quantitative zoom-robustness claim).

**Orientation + visibility (Part 2, the objective):** sign and flank accuracy vs the 5,542 human-locked
instances — **the teacher/CEILING is 92.5 / 97.1 — see the success-criteria ladder at the end; it is NOT the first-experiment bar**. Per-side visibility P/R/F1/κ/ROC-AUC on the
512 human-labelled frame-sets via the existing `visibility.ConfusionMatrixEvaluator:583`. Angular error and
acc@15/30 vs motion on held-out videos (pipeline: 87.4/95.0 walking, 69.1/89.4 standing), band-resolved by
`|sin α|`.

**Label-free, all species:** the **impossible-flank-switch** count (`viewpoint.py:181-192`) — a flank can
only flip through end-on, so a switch with `|sin α| > 0.35` on both sides is provably wrong. No ground truth
needed; runs on all 177,973 animal instances including species with no labels.

**Downstream:** rerun `reid_openset_loso.py` with the model's `view_code`; target the pipeline's **+0.117
open-set AUC / +21.3 TAR@FAR=0.1**.

**Do NOT report:** angular error on the human-locked set (degenerate — the human reference *is* one of the
four candidates, median 0.00° by construction); `acc45` (≡ sign bit-for-bit); track-level splits (leaky).

---

## Ladder (one A40; existing recipe ≈ 3.5 h / 15k iters)

| arm | content |
|---|---|
| **D0** | **Gate.** Trivial predictor — GT 2D boxes, `z≡1`, per-class median dims/pose — through the real evaluator. If it matches 13.17/8.68, the current 3D head contributes ~nothing and *that* is the headline finding. 30 min, CPU |
| **D1** | Sensitivity curve: inject relative depth error ε ∈ {0…20%} with GT elsewhere; plot NHD/AP/BEV. Turns "improve depth" into a target. **Publish this figure** |
| **E0** | Reproduce 13.17/8.68; freeze the split fingerprint; reproduce the teacher's 92.5/97.1 (guards the harness) |
| **E1** | Part-1 bundle (relative-depth parameterisation, K to the head, dims priors, diagonal-normalised losses, AMP, loader) + leave-one-out ablations |
| **E2** | Resolution curve: 560 / 896 / native-crop stream, plotted against patches-per-animal |
| **E3** | Wire α as the orientation target (the `wildbox.py:270` fix); train on template labels, `margin`-weighted |
| **E4** | Add motion labels; ablate template-only / motion-only / both — the test of whether the student beats the teacher |
| **E5** | Visibility head from the geometric teacher; grade vs the 512 human labels |
| **E6** | DINOv3 backbone swap |
| **E7** | Downstream re-ID LOSO + label-free switch rate across all species |
| **E8** | **External validation**: run the Part-1 fixes on AM3D-Real / CARLA-Drone (real metric GT, aerial small-object regime) to show the domain adaptation holds independently of WildBox's pseudo-labels |

---

## Risks, honestly

- **D0 may invalidate the framing** — if a constant predictor matches the fine-tuned model, the paper becomes
  "the 3D head learns nothing here", and Part 1 must be re-scoped. That is why it runs first.
- **We may be fitting VGGT's biases**, since GT is pseudo-label (aspect ratios collapse toward ~2.3:1;
  giraffe l/h measured 1.59 [CORRECTED]; within-track dims relative-std 0.21 [CORRECTED]). Non-circular checks: biological
  plausibility of predicted aspect ratios against published body sizes (**if we reproduce the broken giraffe
  ratio, we are fitting VGGT**), and agreement with a depth foundation model that never saw VGGT.
- **A distilled student is capped by its teacher** — except motion labels are independent of the template, so
  E4 can exceed it. Report student-vs-teacher on held-out locks either way.
- **The teacher is zebra-validated only** (locks cover 2 zebra videos). For elephant/rhino/giraffe the only
  checks are motion agreement and the label-free switch rate. Say so plainly.
- **Never train orientation on WildBox `rotation_matrices`** — arbitrary PCA signs, 10.9% frame-to-frame flips.
- **Landmines to carry verbatim:** `--dtype fp32` (fp16 silently zeroed the ViT-L forward *and still produced
  a plausible number*); `papersub.Segment.scale` (518-space boxes vs full-res intrinsics — once gave "91.8%
  flank accuracy from a model that had never seen an animal"); the raw-vs-canonical rotation join
  (`human_labels.py:113-121`, 180° apart on 16.9% of instances, asserted >0.999); `cropset.load_npz` (lazy
  `NpzFile` = 4,113 ms/crop); `DIMS_PRIORS` makes the cube head per-class, so pretrained cube layers can no
  longer warm-start — add a re-init control arm.

## Verification
1. D0/D1 run before any training. 2. E0 reproduces both baselines (detector 13.17/8.68 and teacher 92.5/97.1).
3. Each Part-1 rung moves the *intended* disentangled-NHD component, not just AP. 4. K-perturbation: baseline
collapses, adapted model does not. 5. Student ≥ teacher on held-out human locks, or the gap is reported.
6. Label-free flank-switch rate does not rise on any species. 7. Downstream re-ID gain retained.

---

# EXECUTION LOG (append-only; the context-loss survival record)

## Environment facts (re-derived more than once — do not re-discover)
- `/mnt/d` goes **stale-mounted** in WSL (`No such device` while still listed in `mount`).
  Fix, run by the user in their terminal: `sudo umount /mnt/d && sudo mount -t drvfs D: /mnt/d`.
  Verify with `ls /mnt/d/3DBOX/papersubdata/WildBox_val_paper.json`.
- Python: `/home/shuklva/miniconda3/envs/ovmono3d/bin/python` (has torch 2.4.1+cu121, shapely, scipy).
  Local GPU is an RTX 2060 (6 GB); the A40 is the training machine.
- **`configs/category_meta.json` trap.** `tools/bev_ap_eval.py:282` hardcodes the *top-level*
  `configs/category_meta.json`, which currently holds the **97-class Omni3D** list — every WildBox class then
  scores 0.00. **Do not edit the shared repo file.** Instead run from a private working dir that has its own
  `configs/category_meta.json` copied from `configs/wildbox/category_meta_wildlife6.json`:
  ```
  mkdir -p /mnt/d/aeroview/work/configs
  cp configs/wildbox/category_meta_wildlife6.json /mnt/d/aeroview/work/configs/category_meta.json
  cd /mnt/d/aeroview/work && PYTHONPATH=/home/shuklva/ovmono3d python /home/shuklva/ovmono3d/tools/bev_ap_eval.py --preds X --gt Y
  ```
- Offline evaluator that needs **no inference**: `tools/bev_ap_eval.py --preds <.pth> --gt <json>`.
  GT reads `center_cam`/`dimensions`/`R_cam`; predictions read `center_cam`/`dimensions`/`pose`/`score`/
  `category_id` — **both through the same `bev_footprint()`**, so seeding predictions from GT `R_cam` is
  a valid, consistent convention.
- Prediction `category_id` is **contiguous (0..5) for fine-tuned runs**, dataset ids (1000..1005) for
  zero-shot runs. Detect, don't assume (`ids & set(cats)`); mapping is `sorted(dataset_ids) -> 0..5`
  = giraffe, grevys_zebra, elephant, plains_zebra, rhino, gazelle.
- **HARNESS VALIDATED**: `bev_ap_eval` reproduced published seed0 exactly — BEV@0.25 macro 24.31 / micro
  19.63, BEV@0.50 macro 8.20 / micro 4.42, and every per-class value.

## D0 — the gate. RESULT: does NOT trip. Framing survives. (2026-07-26)
Script: `tools/aeroview/d0_trivial_predictor.py`. Artefacts in `/mnt/d/aeroview/`.

| variant | BEV@0.25 macro | BEV@0.50 macro | BEV@0.50 micro |
|---|---:|---:|---:|
| trivial geometry (GT 2D boxes, `z≡1`, per-class median dims + mean pose), **tied scores** | 12.62 | 2.31 | 0.92 |
| trivial geometry, **oracle ranking** (score = true BEV IoU; an upper bound, not achievable) | 23.66 | 8.94 | 9.09 |
| **fine-tuned `wl6_init5sp` seed0, own confidence** (published) | 24.31 | 8.20 | 4.42 |
| **fine-tuned geometry, oracle ranking** | **43.03** | **20.39** | 17.61 |

**Finding 1 — the trained 3D head learns real geometry.** Under identical (oracle) ranking it beats the
trivial predictor **1.8× at IoU 0.25 and 2.4× at 0.50**. Fraction of predictions reaching IoU>0.25 / >0.5:
model **30.5% / 12.0%** vs trivial **28.6% / 4.0%**. Part 1 (domain adaptation) is worth doing.

**Finding 2 (unplanned, and probably the cheapest win available) — the confidence head throws away more
than half the model's performance.** Its own scores realise only **24.31 of 43.03 (56%)** at IoU 0.25 and
**8.20 of 20.39 (40%)** at 0.50. So **recalibrating confidence is worth up to ~+19 BEV@0.25 / ~+12 BEV@0.50**,
independent of any geometry change. This confirms the D6 hypothesis (`roi_heads.py:825` ranks by
`score_2d · exp(−uncert)`, with `bbox_3D_uncertainty` bias-init 5 ⇒ 3D losses start scaled by ~0.007).
**Promote confidence calibration to a first-class arm, ahead of the backbone swap.**

**Methodological note that must survive:** a trivial predictor with *no* confidence signal scores 12.62,
with *oracle* confidence 23.66 — a 1.9x spread from ranking alone. **Never quote a floor without stating its
ranking**, and always report new arms against the oracle-ranked floor (23.66 / 8.94), not the tied one.

**Verified correct in D0** (checked, not assumed): class mapping is exact (per-class prediction counts land
on the known supports 110/3405/17119/12586/17632/16099); GT `R_cam` ↔ pred `pose` share one footprint
function; the 2D-box-centre proxy for the projected 3D centre is off by a median 9.6 px = 6.1% of box size.

**Still open / next:** D1 (depth-sensitivity curve, ε ∈ {0…20%}) — but note D0 already shows ranking is a
bigger lever than expected, so run the confidence-calibration arm early.

## PATHS, COMMANDS, GPU (everything needed to resume cold)

**Machines.** Local WSL box = CPU work + a 6 GB RTX 2060 (too small to train). **A40 (48 GB) = the training
machine**; existing recipe ≈ 3.5 h per 15k-iter run, ~12–20 GB used.

**Repos.** `/home/shuklva/ovmono3d` (OVMono3D/cubercnn: detector, evaluator, configs) ·
`/home/shuklva/DetAny3D` (heading pipeline `tools/heading/`, and `AEROVIEW_PLAN.md` = this plan) ·
`/home/shuklva/wildlift-framework` (re-ID graders, visibility/occlusion).

**Data.**
| what | where |
|---|---|
| released annotations | `/mnt/d/3DBOX/papersubdata/WildBox_{train,val}_paper.json` (241 MB / 94 MB) |
| per-segment source (frames, cameras, tracking_summary, kitti_labels) | `/mnt/d/3DBOX/papersubdata/<group>/<video>/<seg>/` — groups `elep1-3, gira1-2, rhin1-2, zebr1-3` (+ `gaze1`, **excluded from the heading project**) |
| SAM3 masks + human face locks + human visibility labels | `/mnt/d/3DBOX/Data/WildBox/data/<shoot>/WildBox_sam3-vggtv1_processed/WildBox/<video>/<seg>/` |
| DJI telemetry | `<video>/<video>.SRT` under the same tree (3 layouts; see rebuttal notes) |
| model predictions (baselines) | `/mnt/d/ovmono3d-lift/<run>/…/WildBox_val/instances_predictions.pth`, `/mnt/d/detany3d/<run>/…` |
| published metrics | `/mnt/d/PAPER_FINAL/{ovmono3d,detany3d}/per_run_metrics/` |
| **AeroView artefacts** | `/mnt/d/aeroview/` (D0 preds, re-ranked variants); eval working dir `/mnt/d/aeroview/work/` |
| heading crops/labels | `/mnt/d/detany3d/heading/{crops,crops_stand,crops_human,labels,human_labels,bridges}.npz` |
| α predicted so far | `/mnt/d/detany3d/headings_v1v2.npz` (12,014), `headings_0006.npz` (2,346) — **NOT the full set** |

**Label inventory for Part 2 (measured):** motion-labelled `crops.npz` 12,762 crops / 53 videos ·
standing `crops_stand.npz` 4,833 / 40 videos · human-gold `crops_human.npz` 5,542 / 2 zebra videos.
Extending α to all 60 videos (~178k) is the first GPU job.

**Reproduce any BEV number (CPU, no inference):**
```
mkdir -p /mnt/d/aeroview/work/configs
cp /home/shuklva/ovmono3d/configs/wildbox/category_meta_wildlife6.json /mnt/d/aeroview/work/configs/category_meta.json
cd /mnt/d/aeroview/work && PYTHONPATH=/home/shuklva/ovmono3d \
  /home/shuklva/miniconda3/envs/ovmono3d/bin/python /home/shuklva/ovmono3d/tools/bev_ap_eval.py \
  --preds <predictions.pth> --gt /mnt/d/3DBOX/papersubdata/WildBox_val_paper.json
```

**GPU job #1 (A40) — α teacher labels for all videos.** Prereq (CPU, local, per group, ~1–2 h total):
```
python -m tools.heading.predict_crops --root /mnt/d/3DBOX/papersubdata --group <grp> \
    --videos <comma-separated video dirs> --out /mnt/d/detany3d/heading/crops_all_<grp>.npz
```
then on the A40 (**`--dtype fp32` is MANDATORY** — fp16 silently zeroed the ViT-L forward and still produced
a plausible number):
```
python -m tools.heading.predict_headings --crops data/heading/crops.npz \
    --reid-crops <crops_all_*.npz> --out-table <headings_all_*.npz> \
    --out-feats <feats_*.npz> --device cuda --dtype fp32
```
Cost ≈ 356 crops/s @224 / 82.8 @448 ⇒ 178k crops ≈ 10–40 min. Needs `--mem=32G` (host RAM, not GPU, is the
constraint). Verify `margin` median ≈ 0.09, **not 0**.

**Training run (A40), when we get there:** `configs/wildbox/OVMono3D_wildbox_wildlife6.yaml`
(IMS_PER_BATCH 8, BASE_LR 0.002, MAX_ITER 15000, SQUARE_PAD 560, 6 classes). Backbone frozen at
`tools/train_net.py:438-441`. `tools/run_multi_seed.sh` is resumable.

## D0b — confidence re-ranking (CPU only, no retraining). RESULT: free gains. (2026-07-26)
Script `tools/aeroview/d0b_confidence_rerank.py`; same geometry, only `score` changed.

| ranker (all inference-available) | BEV@0.25 macro | BEV@0.50 macro | BEV@0.50 micro |
|---|---:|---:|---:|
| model's own confidence (published) | 24.31 | 8.20 | 4.42 |
| depth plausibility `−|log z|` | 19.44 | 6.41 | 6.15 |
| dims plausibility (vs per-class train median) | 17.23 | 5.91 | 10.43 |
| 2D box area | 16.77 | 6.11 | **11.00** |
| **combo = score × depth × dims** | **25.13** | **9.31** | 6.53 |
| *oracle ceiling (true IoU as score)* | *43.03* | *20.39* | *17.61* |

**A hand-weighted product beats the trained confidence head** (+0.8 macro@0.25, **+1.1 macro@0.50 = +13.5%**)
with zero learning. Instance-weighted the gap is larger: plain 2D box area gives **micro@0.50 11.00 vs 4.42
(2.5×)**, recovering 64% of the oracle ceiling vs the model's 26%. Depth/dims/size rankers help abundant
classes and hurt giraffe (110 instances) — hence they win micro and lose macro until combined with the
class-aware model score. **Next: replace the hand-weighted product with a fitted logistic calibrator
(train/held-out split, ~5 coefficients, CPU-seconds).**

## D0c — FITTED confidence calibration (CPU). Partial win; ceiling NOT reachable post-hoc. (2026-07-26)
Script `tools/aeroview/d0c_fit_calibrator.py` (`--per-class`, `--tau`). Leakage guard: **GroupKFold over
VIDEOS** — no prediction is scored by a model that saw its video, so these are honest full-val numbers.
Features (all inference-available): model score, `|log z|`, dims log-deviation vs per-class TRAIN median,
log box area, log aspect, log n-preds-in-image, class one-hot.

| ranker | macro@0.25 | macro@0.50 | micro@0.25 | micro@0.50 |
|---|---:|---:|---:|---:|
| model's own confidence (published) | 24.31 | 8.20 | 19.63 | 4.42 |
| hand-weighted `combo` (score×depth×dims) | 25.13 | **9.31** | 21.09 | 6.53 |
| fitted, **global** | 15.78 | 4.59 | 20.84 | 5.27 |
| fitted, **per-class**, τ=0.25 (corrected targets) | **26.21** | 8.64 | **23.37** | 6.85 |
| fitted, per-class, τ=0.50 | 24.88 | 7.52 | — | 6.40 |
| 2D box area alone | 16.77 | 6.11 | 21.90 | **11.00** |
| *oracle ceiling* | *43.03* | *20.39* | *40.85* | *17.61* |

**Three findings, two of them negative — record all three.**
1. **A GLOBAL fit is the WRONG objective and actively hurts** (15.78 vs 24.31 macro@0.25) *even though its
   global AUC is better* (0.711 vs 0.683). macro-AP ranks **within** a class; a per-class intercept cannot
   change within-class order, so the global fit collapses to ranking by `−4.85·|log z|` inside every class
   and discards the model's own score (+0.57), which carries good within-class ordering. **Never tune a
   detector's ranking on pooled AUC when the metric is macro-AP.**
2. **Per-class fitting fixes that** and beats the model on per-class OOF AUC for every class
   (plains_zebra 0.780 vs 0.717, elephant 0.681 vs 0.641, rhino 0.680 vs 0.612) → best macro@0.25 **26.21**
   and micro@0.25 **23.37**; with corrected targets it beats the model on EVERY class
   (giraffe 0.944 vs 0.875, grevys 0.909 vs 0.833, plains_zebra 0.780 vs 0.717).
3. **Matching the fit threshold to the eval threshold did NOT help** (τ=0.5 gives 7.52 < τ=0.25's 7.90 <
   `combo`'s 9.31): at τ=0.5 only ~12% of predictions are positive, so the fit is noise-limited. The
   hand-weighted product remains best at strict IoU.

**The load-bearing conclusion for the project.** Post-hoc re-ranking buys about **+1.2 macro@0.25 / +1.1
macro@0.50 for free**, and up to **2.5× on micro@0.50** — but every ranker lands ~26 vs the oracle's **43.03**.
So the confidence problem is **not solvable post-hoc from these features**: the remaining ~17 points require
the model to *know* which of its own boxes are good. That is a training-time change — **an uncertainty head
supervised by the realised IoU** (rather than the current free-floating `bbox_3D_uncertainty`, bias-init 5,
ranked via `exp(−uncert)` at `roi_heads.py:825`) — and it is now a first-class arm.

**Artefacts:** `/mnt/d/aeroview/rerank_{depth,dims,size2d,combo,fitted,fitted_pc,fitted_pc50}.pth`,
`model_oracleorder.pth` (score = true BEV IoU, the ceiling), `d0_trivial_preds.pth`.

## CORRECTION (2026-07-26) — a bug in my own oracle-ceiling helper, and the corrected numbers

**The bug.** The first oracle-ordering scripts reimplemented the BEV footprint by hand and got it wrong two
ways: (a) they used a **4-corner mid-height slice** instead of the **convex hull of all 8 corners**, and
(b) they put `dims[1]` on the Z-extent when the Omni3D convention (`tools/bev_ap_eval.py::cuboid_corners`)
is **`dims=[W,H,L]` with W→Z, H→Y, L→X**. This *understated* true IoU.

**The fix, and the rule going forward:** `tools/aeroview/oracle_order.py` now **imports `bev_footprint`
and `rotated_iou` from `tools/bev_ap_eval.py`** instead of reimplementing them. **Never hand-roll geometry
that the evaluator already defines** — import it, or the ceiling you measure is not the ceiling it scores.

**What was affected:** only the *oracle-ordering* files (ranking) and the D0c training targets. `D0`'s
trivial predictions were **not** affected (they carry GT dims/pose; the evaluator computes footprints itself),
so the D0 rows stand unchanged.

**Corrected ceilings** (fraction of predictions reaching IoU>0.25 / >0.5: model **30.5% / 12.0%**
(was 26.3 / 9.9), trivial **28.6% / 4.0%**):

| variant | macro@0.25 | macro@0.50 | micro@0.25 | micro@0.50 |
|---|---:|---:|---:|---:|
| trivial geometry, tied scores | 12.62 | 2.31 | — | 0.92 |
| trivial geometry, oracle ranking | **23.66** | **8.94** | 24.45 | 9.09 |
| model, own confidence (published) | 24.31 | 8.20 | 19.63 | 4.42 |
| model geometry, **oracle ranking (corrected)** | **43.03** | **20.39** | 40.85 | 17.61 |

**Conclusions after correction — both strengthen, neither reverses:**
- The trained head still clearly beats trivial geometry at equal ranking (**43.03 vs 23.66 @0.25**,
  **20.39 vs 8.94 @0.50** — 1.8× and 2.3×).
- The confidence gap is **larger** than first reported: the model realises **24.31 of 43.03 (56%)** at
  IoU 0.25 and **8.20 of 20.39 (40%)** at 0.50. Headroom from ranking alone is **~19 macro@0.25 / ~12 @0.50**.
- ⚠ Note the trivial predictor with oracle ranking (23.66 / 8.94) **essentially ties the trained model's
  published numbers (24.31 / 8.20)** — it even wins at strict IoU. The model's advantage over trivial
  geometry is real but is **entirely spent** by its poor confidence. That is the single sharpest way to state
  the finding.

## STORAGE POLICY (keep /mnt/d lean)
Re-ranking variants differ from the base predictions **only in `score`**, so storing full 74 MB copies is
waste. Keep **`*_scores.npy` (~464 KB)** and materialise a .pth on demand with
`tools/aeroview/apply_scores.py --scores <npy> --out /tmp/x.pth`. Applied: `/mnt/d/aeroview` **649 MB → 140 MB**.
Everything lives under `/mnt/d/aeroview/` (alongside `/mnt/d/detany3d/heading/`); nothing is written to the
Linux home or the repos except small scripts under `ovmono3d/tools/aeroview/`.

## DECISION — the 178k teacher pass is NOT needed yet (do not run it)
"All 60 videos" = **64 total − 4 gazelle** (`gaze1`); gazelle was excluded from the heading project. So the
60 is about gazelle, but the *reason* to consider a teacher pass was coverage, not species count.
**We already hold 12,762 motion-labelled + 4,833 standing crops (17.6k) plus 5,542 human-gold held out.**
That is ample to train a small α head. A 178k distillation pass would cost ~2 h CPU + ~5.6 GB of crops and
yields labels **capped by the teacher**. Correct order: **train the α head on the existing 17.6k first**; only
if it is demonstrably data-limited (not bias-limited) generate more. Its real justification would be pose
coverage — motion labels exist only where the animal walks (14% of frames) — and `crops_stand.npz` already
covers part of that gap.

**D0c RE-RUN on corrected targets (2026-07-26).** The first D0c fit used the buggy oracle file as its
label source, so those numbers were stale. Re-fitted against the corrected IoUs: per-class τ=0.25 now gives
**macro@0.25 26.21 / macro@0.50 8.64 / micro@0.50 6.85** (was 25.53 / 7.90 / 6.80) and beats the model's own
score on **every** class by OOF AUC. The hand-weighted `combo` still edges it at strict IoU (9.31 vs 8.64).
Conclusions unchanged; only the digits moved. Stale values have been corrected in place throughout this file.

---

# CLUSTER (fbk) — verified layout, 2026-07-26. Everything needed to run on GPU.

**Login/compute.** `frontendnew` is the login node (**never run work there**). Get a node with:
```
srun -p gpu-V100 --gres=gpu:1 --cpus-per-task=8 --mem=32G --pty bash     # lands on node8
srun -p gpu-A40  --gres=gpu:1 --cpus-per-task=8 --mem=32G --pty bash     # node81
```
| partition | node | GPUs | **memory** |
|---|---|---|---|
| `gpu-V100` | node8 | 4 × Tesla V100-PCIE | **16 GB each** ⚠ (not 32) |
| `gpu-A40`  | node81 | 8 × A40 (`shard:tesla:96`) | 48 GB |
⚠ **The V100s are 16 GB.** The wildlife6 recipe was measured at ~12–20 GB, so batch 8 at `SQUARE_PAD 560`
may not fit — drop `IMS_PER_BATCH` to 4 with 2× grad accumulation, or wait for the A40.
⚠ **`--dtype fp32` is MANDATORY on V100** — fp16 silently zeroed the ViT-L forward there and still produced
a plausible-looking number (the worst failure mode we have hit).
⚠ **Never change `CUDA_VISIBLE_DEVICES`** — cluster policy; the scheduler sets it.

**Repos (note the split across storage2 and storage3):**
| repo | path | notes |
|---|---|---|
| **ovmono3d** (cubercnn, detector + evaluator) | **`/storage2/3DOM/vshukla/repos/ovmono3d`** | ← storage2, NOT storage3 |
| DetAny3D (heading pipeline) | `/storage3/3DOM/vshukla/DetAny3D` | branch `wildbox_detany3d`, has `data/heading` |
| LabelAny3D, sam3, sam-3d-objects, dinov3, GroundingDINO | `/storage3/3DOM/vshukla/…` | |
Sync is by **git push/pull between the local CPU box and the cluster** (user's workflow).

**Conda envs** (`/storage3/3DOM/vshukla/envs/`) — verified contents:
| env | torch | detectron2 | timm | use |
|---|---|---|---|---|
| **`ovmono3d`** | 2.4.1+cu121 | ✅ | ✅ | **the detector — matches the local env exactly** |
| `ovmono3d_py310` | 2.4.1+cu121 | ✅ | ✅ | py310 variant |
| `la3d` | 2.2.2+cu121 | ✅ | ✅ | |
| **`dinov3`** | 2.6.0+cu124 | — | ✅ | **heading template / `predict_headings`** |
| `detany3d` | 1.13.1+cu116 | — | ✅ | DetAny3D |
| `sam3`, `sam3d-objects` | 2.7.0 / 2.5.1 | — | ✅ | |
Activate with `conda activate /storage3/3DOM/vshukla/envs/<name>`.

**Checkpoints present:** `/storage2/3DOM/vshukla/repos/ovmono3d/checkpoints/`**`ovmono3d_lift.pth`** (the
fine-tune init), `groundingdino_swinb_cogcoor.pth`, `sam_vit_h_4b8939.pth`. DetAny3D has
`detany3d.pth`, `dinov2_vitl14_pretrain.pth`, `unidepth model.pth`, `sam_vit_h`.

**Datasets on cluster:** `datasets/Omni3D` exists under BOTH `/storage2/…/ovmono3d/` and
`/storage3/…/DetAny3D/`. ⚠ **Not yet verified**: whether the WildBox *image root* (the ~60 k jpgs) is
present, or only the annotation jsons. Check before training:
`ls /storage2/3DOM/vshukla/repos/ovmono3d/datasets/Omni3D/ && du -sh .../datasets/*`

**Disk:** storage2 45 T free, storage3 62 T free; vshukla currently 1.2 T. Space is not a constraint.

**Probe gotcha for next time:** `ls -d a b c 2>/dev/null || echo "NOT FOUND"` prints NOT FOUND if *any*
pattern misses, even when others matched — it produced a false "ovmono3d NOT FOUND" here. Use
`ls -d … | grep . || echo` instead.

## First GPU job (revised, given the D0 findings)
Not the α teacher pass (dropped — see the decision above). The target is the **uncertainty head supervised
by realised IoU**, since the model realises only 56% of its own geometry at IoU 0.25 and a no-learning
predictor with good ranking ties it. Order:
1. **Smoke test** the training loop on the cluster (short run, confirm data + env + checkpoint load).
2. Code change in `cubercnn/modeling/roi_heads/roi_heads.py` / `cube_head.py` so `bbox_3D_uncertainty` is
   trained against the box's realised 3D/BEV IoU rather than floating free (today: bias-init 5, consumed as
   `score_2d · exp(−uncert)` at `roi_heads.py:825`).
3. Train, then evaluate with `tools/bev_ap_eval.py` against the frozen references in this document.

## ⚠ TRAINING BLOCKER — WildBox IMAGES are (probably) not on the cluster
Verified on node8: `/storage2/3DOM/vshukla/repos/ovmono3d/datasets/` contains
`Omni3D/` (**5.2 GB — the Omni3D annotation jsons**: ARKitScenes, Hypersim, KITTI, Objectron, …),
plus empty stubs `ARKitScenes/` (8 K), `objectron/` (8 K), `coco_examples/` (1.6 MB).
**There is no WildBox image root under `datasets/`.** Locally the images live at
`/mnt/d/3DBOX/papersubdata/<group>/<video>/<seg>/frame_*.jpg` (~60 k jpgs, 59,758 released).

**RESOLVED on node8 (2026-07-26):** the annotation jsons **ARE** present —
`WildBox_train.json`, `WildBox_val.json`, `WildBox_{train,val}_10zip.json`, and crucially the oracle-2D
files **`gt2d_WildBox_val_oracle_2d.json`** and **`gdino_WildBox_val_oracle_2d.json`** (the frozen 2D boxes
the controlled-comparison protocol needs — do not regenerate them, reuse these so every arm shares one 2D
source). But `find -name frame_000001.jpg` over storage2+storage3 returns **nothing**, and there is no
`papersubdata` tree: **the images are local-only.**

**Transfer required before training: 31.5 GB / 59,598 jpgs** (mean 554 KB, 1920x1080). Per group:
zebr3 7.09 · zebr2 3.86 · rhin2 3.69 · elep3 3.36 · rhin1 2.96 · gaze1 2.85 · zebr1 2.52 · elep1 2.45 ·
elep2 1.81 · gira1 0.71 · gira2 0.18 GB. (Note 59,598 on disk vs 59,758 referenced by the jsons — the 160
missing are the unshipped `rhin1/DJI_20250303174548_0001_D/seg4`, a known dataset packaging gap.)
**Gazelle (gaze1, 2.85 GB) can be skipped** if we stay on the heading project's 60-video non-gazelle scope —
but the detector's 6-class training DOES use gazelle, so transfer it for Part 1.

Consequences:
- **Evaluation-only work needs NO images** — `tools/bev_ap_eval.py` consumes a predictions `.pth` + the GT
  json. All of D0/D0b/D0c ran this way, on CPU. This is why the diagnostics were cheap.
- **Training DOES need the images.** Before any training run, confirm/transfer:
  1. are `WildBox_{train,val}.json` in `datasets/Omni3D/`? (the `head` listing was alphabetical and cut off
     before "W" — check explicitly)
  2. is there an image root anywhere on storage2/3 (search for `frame_000001.jpg` or a `papersubdata` tree)?
  3. if absent, transfer papersubdata images (jpgs only, not the npz/zips) — tens of GB; storage2 has 45 T
     free so space is fine, transfer time is the cost.
- `file_path` in the jsons is relative (`<group>/<video>/<seg>/frame_XXXXXX.jpg`), so the image root just has
  to be registered as the dataset's `image_root` — no path rewriting needed if the tree is copied intact.

---

# FULL INDEPENDENT AUDIT (18 agents) + ALL FIXES — 2026-07-26. Read this before trusting anything above.

Three parallel audits (numbers / code / records) re-derived everything from raw artefacts, then skeptics
attacked each finding, then a completeness sweep. **Every BEV number in this file reproduced exactly** —
the arithmetic was sound. The failures were in *experimental design* and in *cited facts carried over
without re-derivation*. 12 findings survived attack; 2 were refuted. Corrections below are authoritative.

## ✅ REFUTED (claims that were challenged and SURVIVED — do not "fix" these)
- **"D0 does not trip" STANDS.** An auditor claimed the trivial predictor's per-class mean rotation is SVD
  noise (singular values ~[1.0,0.2,0.2]) and that a constant 90° yaw lifts the floor to 26.36/12.08, beating
  the model. **Refuted:** degenerate singular values do NOT make the polar factor ambiguous — `R=UVᵀ` is the
  unique nearest rotation whenever σ₂+σ₃>0, and bootstrap gives BEV-yaw circular std **0.78–1.66°** (a stable
  statistic). Decisively, fitting the yaw offset on **TRAIN** — the auditor's own preferred method — selects
  **θ≈0–10°**, and **θ=90° is 11% WORSE on TRAIN**. The θ=90 gain was val-set cherry-picking.
- **"Depth carries 85–99% of 3D error" is CORRECT** — measured span over all 13 runs is **84.52%–99.65%**
  (the auditor had searched only one run directory).

## 🔴 THE ONE FINDING THAT CHANGED A CONCLUSION — seed selection, and its resolution
Every re-ranking experiment was run on **seed0, the weakest of three seeds**, while the paper headline is the
3-seed mean. The auditor compared `combo`(seed0)=25.13 against **mean**(own)=25.74 and concluded the gain was
illusory. **That comparison is invalid — it is unpaired.** The correct test is paired per seed, which I ran:

| seed | own confidence macro@0.25 → combo | own macro@0.50 → combo |
|---|---|---|
| seed0 | 24.31 → 25.13 (**+0.82**) | 8.20 → 9.31 (**+1.11**) |
| seed1 | 25.76 → **28.97** (**+3.21**) | 9.14 → **12.25** (**+3.11**) |
| seed2 | 27.14 → **28.20** (**+1.06**) | 8.72 → **9.44** (**+0.72**) |
| **mean** | **25.74 → 27.43 (+1.70)** | **8.69 → 10.33 (+1.65)** |

**The re-ranking gain REPLICATES on all three seeds** (+0.82/+3.21/+1.06 macro@0.25; every seed positive at
both thresholds). Mean gain **+1.70 macro@0.25 / +1.65 macro@0.50**, larger than the single-seed result.
So the conclusion survives and strengthens — but **all future arms MUST be reported paired across 3 seeds**,
never on seed0 alone. ⚠ The old flagship line "2D box area gives micro@0.50 11.00 vs 4.42 (2.5×)" is
**WITHDRAWN**: seed2's *own* confidence already scores 11.29, so that ratio was a seed0 artefact.

## 🔧 CORRECTIONS APPLIED (facts I had carried over without re-deriving)
| claim (was) | corrected | note |
|---|---|---|
| animals median **130 px**; giraffe 335 > elephant 214 | median **142 px**; **giraffe is SMALLER than elephant** under all 7 size definitions tested | my Part-1 table had the species ordering backwards |
| "**62%** of tracks have frozen dims" | **0%** — median within-track dims relative-std **0.21** | no artefact ever supported 62% |
| "giraffe l/h **2.5–2.9** vs true 0.90" | measured **1.59** | overstated |
| PCA sign flips on **12.4%** of consecutive frames | **10.9%** of consecutive annotated same-track pairs (236,027 pairs) | |
| fx varies **1.4–2.6×** within a video | **1.5× median, up to 3.4×** | pooled span 10.9× is correct |
| template teacher "**177,973 images**" | **177,973 animal instances across 52,443 frames**, 60 videos | instances ≠ images |
| model geometry beats trivial "**2.4×** @0.50" | **2.3×** | |
| NHD-xy = **2.1** | **2.2** | headline-run 3-seed mean is 2.193 |
| D0c fitted-global AUC "0.711 vs 0.683" | **0.679 vs 0.672** | two D0c rows were never re-fitted after the oracle fix; **now regenerated** |

## 🐛 REAL CODE BUG FOUND IN THE EVALUATOR (not my code) — must fix before AM3D/CARLA
`tools/bev_ap_eval.py` — the **micro** pass matches predictions to GT **class-agnostically**: `category_id`
is dropped when the per-image GT list is built (~:225), so a prediction can match a GT of a *different*
class. Verified **harmless on WildBox (+0.02)** because classes are spatially separated here, but it **will
inflate micro AP on datasets where classes co-occur** — exactly the planned AM3D / CARLA-Drone runs.
Fix: carry `category_id` into the per-image GT dicts and require a class match in the micro matching loop.
**Macro AP is unaffected** (it already filters per class), so every macro number in this file stands.

## 📌 STANDING RULES (each earned by a bug that produced a believable wrong number)
1. **Never hand-roll geometry the evaluator defines** — import `bev_footprint`/`rotated_iou`.
2. **Report every arm paired across all 3 seeds.** Seed spread (2.83 macro@0.25) exceeds most effects.
3. **Never quote a floor without its ranking** (12.62 tied vs 23.66 oracle is ranking alone).
4. **Re-derive cited facts from data** before putting them in this file — 6 of the 8 wrong numbers above were
   inherited from earlier context, not measured.
5. **Fit calibrators per class**, and on TRAIN/held-out — a global fit optimises the wrong objective and a
   val-fitted constant is cherry-picking.

---

# DATA TRANSFER — space analysis + the HuggingFace shortcut (2026-07-26)

## Space is NOT a constraint. Do not delete anything.
Measured on node8:
| filesystem | total | used | **free** |
|---|---|---|---|
| storage2 | 109 T | 59 T | **45 T** |
| storage3 | 109 T | 43 T | **62 T** |
| `vshukla` total footprint | | 1.2 T (on storage3) | |

The WildBox image set is **31.5 GB** = **0.07% of the 45 T free on storage2**. There is no space problem and
**no reason to delete anything on the cluster.** If space were ever needed, the safely-reconstructible items
would be: `/storage2/.../ovmono3d/datasets/Omni3D` (5.2 GB — public Omni3D annotation jsons, re-downloadable),
and the public checkpoints (`ovmono3d_lift.pth`, `groundingdino_swinb_cogcoor.pth`, `sam_vit_h_4b8939.pth`,
`dinov2_vitl14_pretrain.pth`, UniDepth `model.pth`) — all re-downloadable. **But none of this is necessary.**

## ⭐ PREFERRED ROUTE: pull from HuggingFace, do NOT upload from the laptop
The dataset is published for NeurIPS review at
**`https://huggingface.co/datasets/wildbox-anon-2026/wildbox-review`**.
Pulling on the cluster is far better than pushing 31.5 GB from the WSL laptop over a home uplink:
the cluster has fast symmetric bandwidth, `hf` transfers are resumable and parallel, and it removes the
laptop as a bottleneck/failure point.

```
conda activate /storage3/3DOM/vshukla/envs/ovmono3d      # or any env with huggingface_hub
pip install -U "huggingface_hub[cli]" hf_transfer        # if absent
export HF_HUB_ENABLE_HF_TRANSFER=1                       # parallel chunked download
huggingface-cli download wildbox-anon-2026/wildbox-review \
    --repo-type dataset \
    --local-dir /storage2/3DOM/vshukla/repos/ovmono3d/datasets/wildbox_hf
```
⚠ **VERIFY FIRST what the HF repo actually contains** — it was created to support the NeurIPS review (it
holds `REBUTTAL_VISUALIZATION.md` and visualization gifs), so it may be a *review-support* repo rather than
the full 60 k-image release. Check before relying on it:
```
huggingface-cli download wildbox-anon-2026/wildbox-review --repo-type dataset --local-dir /tmp/hfprobe \
    --include "*.md" "*.json"      # metadata only, cheap
# or list without downloading:
python -c "from huggingface_hub import list_repo_files; \
  fs=list_repo_files('wildbox-anon-2026/wildbox-review', repo_type='dataset'); \
  print(len(fs),'files'); print([f for f in fs][:40])"
```
Decision rule: if it contains the `<group>/<video>/<seg>/frame_*.jpg` tree (~59.6 k jpgs), **use HF and skip
the upload entirely**. If it only holds review material, fall back to rsync from the laptop:
```
rsync -av --partial --include='*/' --include='frame_*.jpg' --exclude='*' \
  /mnt/d/3DBOX/papersubdata/ vshukla@<host>:/storage2/3DOM/vshukla/repos/ovmono3d/datasets/papersubdata/
```

## What the images are actually needed FOR (scope the transfer)
- **NOT needed** for evaluation/diagnostics — `tools/bev_ap_eval.py` consumes a predictions `.pth` + the GT
  json only. All of D0/D0b/D0c ran on CPU with no images. Those can run on the cluster today.
- **Needed** only for TRAINING (the detector reads frames). Required groups for the 6-class detector:
  **all 11 including `gaze1`** (gazelle is a class). If only the heading/orientation work were in scope,
  gazelle (2.85 GB) could be skipped — but Part 1 needs it.
- Already on the cluster and must NOT be re-created: `WildBox_{train,val}.json`, and the frozen 2D-box files
  **`gt2d_WildBox_val_oracle_2d.json`** + **`gdino_WildBox_val_oracle_2d.json`** (every arm must share these).

## ✅ CLUSTER NETWORK: HuggingFace IS reachable; only pypi is not (resolved 2026-07-26)
**Verified on `frontendnew`:** `getent hosts huggingface.co` resolves (IPv6), `curl -I https://huggingface.co`
returns **HTTP/2 200**, and `list_repo_files(...)` returns **76 files**. The only failure is **pip/pypi**
(`Name or service not known` for pypi.org) — so **never run pip on the cluster; it is not needed.**
Consequences:
- **`pip install` is not needed anyway**: `huggingface_hub 0.36.2`, `hf-xet 1.4.3`, `requests`, `tqdm` are
  **already present** in `/storage3/3DOM/vshukla/envs/ovmono3d`. Only `hf_transfer` (an optional speed-up)
  was missing. Do not fight pip.
- **If there is genuinely no route to the internet, the HuggingFace shortcut is dead** and the only path is
  transferring from the laptop, which already holds the images at `/mnt/d/3DBOX/papersubdata`
  (31.5 GB, 59,598 jpgs). Downloading from HF onto the laptop would be pointless — the laptop already has
  the data; HF was only ever a way to avoid the slow laptop uplink.
- Before giving up on HF, check for the standard HPC escapes: an **HTTP(S) proxy** (`http_proxy`/`https_proxy`,
  often set only in `/etc/environment` or by a module), or a **dedicated transfer/DTN node** with egress.

### The HF repo DOES contain the full dataset (verified from the laptop)
`wildbox-anon-2026/wildbox-review` (repo_type=dataset), 76 files:
**65 per-video `.zip`** across all 11 groups (rhin1 12, zebr3 12, elep3 7, rhin2 7, elep1 6, zebr2 5,
elep2 4, gaze1 4, zebr1 3, gira1 2, gira2 2), `WildBox_{train,val}_paper.json`, `DATASET_README.md`,
`croissant.json(ld)`, and **two trained checkpoints**:
`checkpoints/ovmono3d_lift_init5sp_seed0/model_final.pth` and `checkpoints/detany3d_ep2_seed0/checkpoint_0.pth`.
⭐ That means **the fine-tuned seed0 checkpoint is publicly recoverable** — useful if the cluster copy is ever
lost. Zero loose images: everything is zipped per video, so a transfer is 65 files, not 59,598.

### Practical consequence for transfers
Because HF stores **per-video zips**, the laptop→cluster transfer should also send **zips, not loose jpgs** —
far fewer files, better throughput, resumable per video. If the laptop lacks the zips, they exist on HF and
also as `/mnt/d/3DBOX/papersubdata/<group>/<video>.zip` for at least some groups (verify).

### ⭐ THE WORKING DOWNLOAD COMMAND (verified prerequisites; do not deviate)
```
conda activate /storage3/3DOM/vshukla/envs/ovmono3d
unset HF_HUB_ENABLE_HF_TRANSFER      # hf_transfer is NOT installed and pypi is unreachable; setting this
                                     # makes huggingface_hub raise. hf-xet 1.4.3 IS installed and is faster.
cd /storage2/3DOM/vshukla/repos/ovmono3d/datasets
huggingface-cli download wildbox-anon-2026/wildbox-review --repo-type dataset --local-dir wildbox_hf
cd wildbox_hf && for z in */*.zip; do unzip -q -n "$z" -d "$(dirname "$z")/"; done
```
**Zip layout verified**: each archive contains `<video>/<seg>/...` with **no group prefix**, so unzipping in
place under `<group>/` yields `<group>/<video>/<seg>/frame_XXXXXX.jpg` — exactly what `file_path` in
`WildBox_{train,val}.json` expects. **No path rewriting.** Each zip ≈ 580 MB / 1,466 files and includes
`kitti_labels/`, `cameras.json`, `tracking_summary.json` as well as the jpgs, so the full pull is ~35-40 GB
(vs 31.5 GB of jpgs alone). Nothing needs deleting — storage2 has 45 TB free.

**IMAGE ROOT after unzip:** `/storage2/3DOM/vshukla/repos/ovmono3d/datasets/wildbox_hf`

**Post-transfer check (expected values):**
```
find .../wildbox_hf -name "frame_*.jpg" | wc -l     # expect 59,598
```
⚠ **59,598, not 59,758** — the 160-image shortfall is the unshipped `rhin1/DJI_20250303174548_0001_D/seg4`
(known dataset packaging gap, documented above). Do not treat it as a failed transfer.

**Bonus:** the HF repo also carries `checkpoints/ovmono3d_lift_init5sp_seed0/model_final.pth` — the exact
fine-tuned seed0 model whose predictions every D0/D0b/D0c diagnostic analyses. It is therefore publicly
recoverable and can never be permanently lost.

---

# SANITY CHECK before training — label join, gazelle scope, and the exact model contract (2026-07-26)

## GAZELLE: in the detector, absent from orientation. Both are correct.
| | classes | gazelle? |
|---|---|---|
| **Part 1 — detector** (WildBox benchmark) | 6: giraffe, grevys_zebra, elephant, plains_zebra, rhino, **gazelle** | **YES — must keep.** Dropping it changes the benchmark and breaks comparability with the published 13.17 3D AP / 8.68 BEV@0.50 |
| **Part 2 — orientation/visibility** (heading project) | 4: elephant, giraffe, rhino, zebra (merged) | **NO — zero labels exist** |

Gazelle was excluded from the heading project for good reason, and the data supports it: gazelle has the
**smallest animals** (median 87 px), the **worst GT stability** (depth jitter 0.120 BL, dims CV 0.162) and
**near-useless motion-yaw agreement** (33.7° in a world ground plane vs a 45° null). So:
**train the detector on all 6 classes; MASK gazelle in the orientation loss.** Note also the heading project
merges zebra while the detector splits plains/grevys — the orientation label is species-agnostic, so this is
harmless, but do not try to map heading `species` onto detector `category_id`.

## ⚠ THE JOIN KEY — a trap that silently loses 99% of labels
`crops.npz` stores **`frame` as the SEGMENT-LOCAL index (0..199)**, while the released `file_path` carries the
**video frame number** (`frame_004210.jpg`). Joining on `frame` matches **265 / 237,505 = 0.1%** and looks
like "the labels don't join". **Join on `image_name`** (the actual filename) instead:

    key = (video, seg_name, int(track.split("::")[-1]), image_name)

With the correct key: **12,762 / 12,762 = 100% of heading labels land on a detector annotation.**

## Measured orientation-label coverage of the detector's training set
| class | annotations | with heading label | coverage |
|---|---:|---:|---:|
| plains_zebra | 62,450 | 1,681 | 2.7% |
| gazelle | 59,379 | **0** | **0.0%** |
| rhino | 46,013 | 5,457 | 11.9% |
| elephant | 42,927 | 3,614 | 8.4% |
| grevys_zebra | 23,048 | 1,182 | 5.1% |
| giraffe | 3,688 | 828 | 22.5% |
| **total** | **237,505** | **12,762** | **5.37%** |

Adding `crops_stand.npz` (4,833 standing labels) takes it to ~17.6 k ≈ **7.4%**. **This is sparse — the
orientation loss MUST be masked, not dense**, and it is an auxiliary term, not the main objective. Sparse
supervision is workable (this is the standard auxiliary-head setup) but it caps how much the orientation head
can learn, and it is the strongest argument for the (currently deprioritised) teacher-distillation pass if the
head turns out to be data-limited rather than bias-limited.

## Two cluster blockers found while preparing the first run (both fail SILENTLY)
1. **Image root.** `datasets.py:131-137` hardcodes `image_root='datasets'` (relative to CWD) and builds
   `file_name = datasets/<group>/<video>/<seg>/frame_X.jpg`. We unzipped to `datasets/wildbox_hf/<group>/...`.
   Fix with symlinks (no data movement):
   ```
   cd /storage2/3DOM/vshukla/repos/ovmono3d/datasets
   for g in elep1 elep2 elep3 gaze1 gira1 gira2 rhin1 rhin2 zebr1 zebr2 zebr3; do ln -sfn wildbox_hf/$g $g; done
   ```
2. **`configs/wildbox/category_meta.json` currently symlinks to `category_meta_wildlife5.json`** (5 classes:
   giraffe, zebra, elephant, rhino, gazelle) while the wildlife6 config declares **6**. `--eval-only` reads
   the meta *next to the config* (`train_net.py:411`). This is the documented footgun that once produced
   silently wrong per-class metrics. Fix:
   `ln -sfn category_meta_wildlife6.json configs/wildbox/category_meta.json`

---

# ✅ E0 VALIDATION PASSED on the cluster (V100, 2026-08-19)
`--eval-only` with `datasets/wildbox_hf/checkpoints/ovmono3d_lift_init5sp_seed0/model_final.pth`
reproduced the published seed0 artefacts essentially exactly:

| metric | published seed0 | cluster run | Δ |
|---|---:|---:|---:|
| 2D AP | 39.614 | **39.623** | 0.009 |
| 3D AP | 12.412 | **12.416** | 0.004 |
| overall NHD | 7.063 | **7.0637** | 0.001 |
| disent xy / z / dims / pose | 2.209 / 5.970 / — / — | **2.211 / 5.970 / 0.776 / 0.547** | ~0 |

Per-class 2D also matches (giraffe 19.29, grevys 28.47, elephant 56.25, plains 44.14, rhino 49.12,
gazelle 40.47). **The full chain is validated: image root symlinks, 6-class category_meta, checkpoint, env,
config.** Eval fits comfortably on a 16 GB V100; runtime ≈ 16 min (the 2D COCO pass dominates at 824 s).

**Depth dominance re-confirmed on this run:** disent_z 5.970 / overall 7.064 = **84.5%** — exactly the
init5sp-seed0 figure the audit measured. xy is only 2.211, i.e. the xy decode works and depth is the problem.

## 🔴 CRITICAL DESIGN CONSEQUENCE THE E0 RUN EXPOSED — read before writing any code
`disent_pose_NHD = 0.547`, the SMALLEST of the four components. That looks like "pose is already solved".
**It is not — the metric cannot see orientation error.** NHD, 3D IoU and BEV AP are all **invariant to a
cuboid's 180° flip**, so a box pointing exactly backwards scores identically to a correct one. This is
precisely why nobody noticed the pose head learns nothing about *direction*.

Two consequences for the implementation:
1. **In cubercnn there is NO α head** (that is DetAny3D). Orientation lives in `bbox_3D_pose` (6D rotation),
   supervised by the annotations' `R_cam` — which carries **arbitrary PCA signs (10.9% frame-to-frame flips)**.
   So the pose target is sign-ambiguous, and the loss cannot teach direction.
2. Therefore **do NOT try to fix orientation by re-signing `R_cam`** for the 5.4% labelled subset — that
   leaves 94.6% of instances still supplying sign-arbitrary targets, i.e. actively conflicting supervision.
   **Add a separate, small, MASKED orientation output** (a sign/α head) trained ONLY on the 12,762 heading-
   labelled instances, leaving the existing pose loss untouched, and grade it with a **flip-SENSITIVE**
   metric (sign/flank accuracy vs the human locks; the label-free flank-switch rate) — never with NHD/BEV,
   which are flip-blind by construction.


## 🎯 SUCCESS CRITERIA for the orientation head — 92.5/97.1 is the CEILING, not the bar
An earlier draft called the heading pipeline's **92.5% sign / 97.1% flank** "the bar to beat". **That is
wrong and would set up a misleading experiment.** Four reasons:
1. **Different inputs.** The pipeline is HANDED the WildBox 3D box, so geometry restricts the answer to
   **4 candidates** and appearance only resolves which end is the head. Our head sees an image, with no box.
2. **It is the TEACHER.** The head is trained on labels this pipeline produced; a distilled student is
   normally capped by its teacher, so "beat the teacher" is not a first-experiment criterion.
3. **It is zebra-only** — 5,542 human-locked instances from 2 videos; not multi-species.
4. **Different runtime shape.** The pipeline is **two-stage**: box -> per-animal crop -> a SECOND network
   forward (the frozen DINOv3 template, ~356 crops/s @224). Our head emits alpha inside the detector's own
   forward pass. Matching ~90% while collapsing two stages into one is a win, not a shortfall.

**⚠ THE LADDER BELOW WAS REPLACED ON 2026-08-19. The 45.7% figure is NOT the student's floor** — it belongs
to the held-out WALKING split under a box-restricted 4-way decision (`METHOD.md:340`). **Use these MEASURED
floors instead** (all computed by fitting the best constant alpha on TRAIN and transferring — an eval-fitted
constant is an ORACLE and is never a valid floor):

| grader | n | **TRAIN-TRANSFERRED constant floor** | notes |
|---|---:|---|---|
| val labels **per species — elephant** | 1,666 (75 tracks) | eval-fitted 77.2 sign / 71.4 flank | **informative** |
| val labels **per species — zebra** | 1,294 (53 tracks) | eval-fitted 74.7 / 72.6 | **informative** |
| val labels **per species — rhino** | 4,500 (80 tracks) | eval-fitted **98.3** / 81.0, R=0.847 | ⚠ **cannot carry a result** |
| val labels, POOLED | 7,460 | **87.9 / 77.4** | ⚠ **NEVER pool** — 60% rhino saturates it |
| val, **track-balanced** (1/track) | **208** | eval-fitted 76.4 / 67.3 | the honest effective n |
| **gold locks (SECONDARY)** | 5,542 (66 tracks) | **35.2 / 34.1** | "labels held out, IMAGES SEEN" |
| ceiling (teacher, zebra only) | 5,542 | 92.5 sign / 94.2 flank all / 97.1 flank on \|sin a\|>=0.35 | not a first-experiment bar |

**Success = held-out sign accuracy beats the TRAIN-TRANSFERRED constant on that exact set by more than the
track-clustered 95% CI**, reported per species and band-resolved by `|sin alpha|`. Design effect ~7x =>
**no gap under ~5 points is real.** A degenerate constant-alpha head passes every criterion the old ladder
stated, which is why the old ladder is withdrawn.

Grade ONLY with flip-SENSITIVE metrics. NHD, 3D IoU and BEV are **exactly invariant to 180 deg flips**
(verified: min BEV IoU **1.000** over 3000 random boxes, 8-corner set identical to 6.3e-16, with a 90 deg
control at IoU 0.333 proving the test has power) and cannot measure orientation at all.
⚠ Also flip-BLIND, so NOT usable as falsifiers: the label-free flank-switch count (bit-identical for the truth
and for a 180-inverted prediction) and the E7 re-ID AUC (a uniform relabel of query and gallery changes no
score). The ONE flip-sensitive multi-species channel is **alpha vs motion-derived heading on held-out videos** —
that is the primary grader.

---

# ★ AUDIT VERDICT + CORRECTED GROUNDWORK (2026-08-19). THIS SECTION SUPERSEDES CONFLICTING TEXT ABOVE.

13-agent adversarial audit (6 dimensions verified, each attacked by a skeptic, then arbitrated).
**VERDICT: PROCEED.** The core bet is well-posed; the join is exact; every E0 number reproduces.
Only TWO blocking issues survived, both pre-flight, both < 1 h. Full record:
`/tmp/.../scratchpad/SYNTHESIS.md` (regenerate from the workflow if lost).

## B1 (BLOCKING) — horizontal flip inverts alpha. Fix before any GPU hour.
`INPUT.RANDOM_FLIP="horizontal"` (`cubercnn/config/config.py:237`, never overridden) with prob 0.5.
alpha = atan2(d.s, d.r), s = up x r (`papersub.py:221,232`). Under a camera-x mirror M: s -> -Ms, so
**alpha -> -alpha: cos alpha (sign/end) SURVIVES, sin alpha (flank) INVERTS on every flipped sample.**
Flank supervision becomes 50/50 self-contradictory and the head drives sin->0. Invisible in the training log.
FIX: in `dataset_mapper.py` HFlip loop, `heading_alpha := -heading_alpha`, placed **OUTSIDE** the
`if annotation['center_cam'][2] != 0:` guard at `:87` (the existing pose-mirror block at `:120-128` is nested
inside it and would silently skip labels).
⚠ RELATED TRAP: cubercnn's own hflip pose mirror is **not** the geometric mirror — it differs from `M R M` by
exactly 180 deg about the box's height axis (a head/tail swap), invisible to every metric in use.
**NEVER anchor alpha to `annotation['pose']`.**

## B2 (BLOCKING) — the gold set overlaps the training labels.
Both lock videos are 100% in `WildBox_train` (725 / 533 hits; **0** in val). Direct key overlap:
`crops n crops_human` = 101, `crops_stand n crops_human` = 41 -> **142 exact keys on 12 of the 66 gold
tracks = 1,105 of 5,542 gold instances (19.9%)**, and heading is near-constant within a track.
FIX (join script only, no repo edit): train labels = `keys(crops) u keys(crops_stand)` MINUS `keys(crops_human)`
MINUS both lock videos. **VERIFIED: this lands at exactly 9,887.** Assert 0 gradient-carrying orientation RoIs
from the two lock videos.
NOTE: no `builtin.py` edit is needed — `WildBox_test` is already whitelisted at `cubercnn/data/builtin.py:52`
and `tools/train_net.py:394-395` registers any name in `DATASETS.TEST`.

## ⚠ MY OWN MEASUREMENT — THE ARBITER'S "PROMOTE THE 7,460 VAL LABELS TO PRIMARY" IS UNSAFE AS STATED
The arbiter recommended the 7,460 val motion+standing labels as the primary held-out grader without computing
their floor. **I computed it. A CONSTANT alpha scores 87.9% sign / 77.4% flank there** (eval-fitted oracle only
88.6/77.4). Cause, measured:

| grader | n | train-transferred constant floor | eval-fitted (oracle) floor | R |
|---|---:|---|---|---:|
| val labels, POOLED | 7,460 | **87.9 sign / 77.4 flank** | 88.6 / 77.4 | 0.641 |
| val, elephant | 1,666 | — | 77.2 / 71.4 | 0.417 |
| val, **rhino** | 4,500 | — | **98.3 / 81.0** | **0.847** |
| val, zebra | 1,294 | — | 74.7 / 72.6 | 0.388 |
| val, **TRACK-BALANCED** (1/track) | **208** | — | **76.4 / 67.3** | 0.405 |
| gold locks | 5,542 | **35.2 / 34.1** | 75.1 / 65.9 | — |

**Rhino is 60% of the val grader and its headings are nearly all identical (R=0.847), so pooled val is
saturated.** Consequences, binding:
1. **NEVER pool the val grader.** Report per species, always beside that species' own constant floor.
2. **Rhino-val cannot carry a result** (98.3% floor). Elephant (75 tracks) and zebra (53 tracks) are informative.
3. **Track-balance or track-cluster everything.** The effective n is **208 tracks**, not 7,460 instances.
4. **The floor is always the TRAIN-TRANSFERRED constant on that exact set** — never 45.7%, never 50%, never an
   eval-fitted constant (that is an oracle). On the gold locks the transferred floor is only **35.2%**, which is
   why the locks remain valuable as a *secondary* grader despite "images seen".

## Phase 0 DECISIONS (now binding)
1. **Grading rule.** Continuous alpha. sign = `|wrap(a_hat - a*)| < 90 deg`; flank = `sign(sin a_hat) == sign(sin a*)`.
   Band-resolve by `|sin a*|`; report against the train-transferred constant on the same set.
2. **Target array = `Y` = `[cos alpha, sin alpha]`, already on disk in all three crop files.** Do NOT use
   `face_alpha[i, y_face]` — that is the heading SNAPPED to the nearest PCA box face.
   **RE-MEASURED BY ME (the audit's figures were wrong — do not reuse them):** it flips the FLANK bit on
   **9.11%** of `crops` and **11.19%** of `crops_stand`, and **0.00%** on `crops_human` — which is exactly why
   this error is invisible on the gold set. The SIGN bit flips **0.00%** everywhere (the audit claimed
   6.30%/3.19%): structurally impossible, since the 4 box faces are 90 deg apart so the nearest face is always
   within 45 deg of the true heading, hence `|wrap(a_snap - a_true)| <= 45 deg < 90 deg`. Measured 0.00% confirms it.
   NET: snapping costs ~9-11% of the FLANK bit and nothing of the sign bit.
3. **Graders:** primary = val labels **per species, track-balanced**; secondary = the 5,542 locks, labelled
   *"labels held out, images seen"*, track-clustered CIs over 66 tracks.
4. **Ceiling text:** `92.5% sign (n=5,542) / 97.1% flank on |sin a|>=0.35 (n=5,281, 95.3% coverage) / 94.2% flank
   on all 5,542`. Stop quoting flank as an independent second result — on this set flank is a deterministic
   consequence of sign.
5. **VGGT ledger.** DELETE "the orientation and visibility targets are not VGGT-derived". The allocentric BASIS
   in which every alpha is expressed (`papersub.py:214`, `Segment.allocentric_basis`) is a per-segment consensus
   over VGGT box rotations (`papersub.py:131-140`), so `up` is a fourth circularity channel. **Runtime fix
   (measured by one auditor, NOT re-derived by the arbiter — verify before relying):** consensus over the
   DETECTOR's own predicted poses agrees to median 1.76 deg (p90 4.19). Needs >=10 boxes pooled per segment;
   the single-frame single-animal case is UNSOLVED.
6. **Number fixes:** seed0 NHD **7.063** (not 7.065); gold set **5,542** everywhere (the 11,084 in the
   supervision table is wrong — the other half has no crop; 5,542 is a clean stride-2 subsample, retention
   exactly 0.500 on all 66 tracks); coverage **5.80% train-only** (the 5.37% used a train+val denominator);
   depth share "84.5-91.8% across trained runs, up to 99.7% on collapsed zero-shot arms" (the four disentangled
   components sum to 134.5% of the total, so it is an ablation ratio, NOT a partition share).
7. **45.7% is real but belongs to the held-out WALKING split under a box-restricted 4-way decision**
   (`METHOD.md:340`). It is NOT the student's floor. Superseded by item 4 of the measurement block above.

## What is FALSIFIABLE, and what is not
Three of the plan's four evaluation channels are provably invariant to a **global heading inversion**:
NHD/BEV/3D AP (180-flip-blind, proven end-to-end: flipping all 118,809 seed0 predictions gives a
**byte-identical** evaluator report, while a 90 deg control moves macro@0.50 from 8.20 to 0.87);
the label-free flank-switch count (bit-identical for truth and for a 180-inverted prediction at 0.0% flank);
and E7 re-ID AUC (a uniform relabel of query and gallery leaves every score unchanged).
**The fourth channel IS flip-sensitive and multi-species: alpha vs motion-derived heading on held-out videos.
Promote it to primary.** Falsifier: held-out sign accuracy fails to beat the train-transferred constant on the
same set by more than the track-clustered 95% CI, band-resolved by `|sin a|`.
Design effect ~7x on the locks => 95% CI ~+/-4.9pp at teacher accuracy. **No student-vs-teacher gap under ~5
points is real.** No power analysis exists for the ladder's internal contrasts — do one before committing 9+ runs.

## CUT / DEFER (decided)
- **CUT E8** (AM3D/CARLA): not on disk, metric geometry on vehicles, out of scope.
- **CUT any Viterbi/temporal-decoding rung.** Already measured: **+0.3%, NULL**, with a mechanism (errors are
  whole-track inversions — 46/72 zebra tracks >80% wrong — and a smoothness prior PRESERVES a coherent wrong
  trajectory). It is on the project's DO-NOT-RETRY list. An auditor recommended adding it by reading the
  module's aspirational docstring; do not.
- **DEFER E6** into E2 as one joint backbone+resolution arm. A DINOv3 patch-16 swap renames FPN outputs
  p2/p3/p4 -> p3/p4/p5 (every `IN_FEATURES` breaks) and needs `SQUARE_PAD` raised with the input.
- **DEFER E5** behind a data rung that runs the ray-OBB / n.v teacher over WildBox. The released
  `visibility`/`truncation` fields are literal CONSTANTS over all 237,505 annotations; nothing generates targets;
  the 512 human labels live on a different data organisation (6 tracks, one side positive 99.6% of the time).
- **MOVE** the uncertainty/confidence head to "the detection-paper arm" — by the flip-invariance argument it
  cannot move any Part-2 number. Keep it: the model realises only 56% of its own geometry under its own ranking.

## The single biggest advance NOT being used
A **frozen monocular RELATIVE-depth foundation model (Depth Anything 3 / MoGe-2 / UniDepthV2) as a conditioning
input to the cube head.** Depth is the entire error budget; relative depth is affine-invariant, so it survives
the out-of-scope-metric-scale constraint exactly where the metric-depth successors (UniMODE 28.2, 3D-MOOD 30.0,
LocateAnything3D **38.90** — not 49.89, an auditor mis-cited the abstract) have nothing to bind to.
Use it as a conditioning INPUT, not as a scored floor: WildBox GT depth is itself VGGT-authored, so scoring a
depth model against it measures agreement with VGGT, not accuracy.

## Herd visibility: HALF the stated purpose is NOT delivered
**Per-animal viewpoint IS delivered** — `viewpoint_of(alpha)` is a pure function of one scalar, so a detector
emitting alpha yields flank/end/strength/usable with **no 3D box, no gravity vector and no VGGT at inference**.
That is a real collapse of the two-stage pipeline into one forward pass.
**Herd visibility is NOT delivered and is not even specified**: every output is per-animal; there is no per-frame
aggregate anywhere — no herd count, no fraction-of-herd-usable, no coverage/duty statistic, no who-blocks-whom
edge list, no metric that scores a FRAME rather than an INSTANCE. Either specify one or drop the word "herd".

---

# VISUALIZATION PLAN — how we verify detection + heading on real images

**Tool:** `ovmono3d/tools/aeroview/viz_heading.py`. Two modes; the second is the one that actually works.

```
# GT only (runs today, no model needed)
python tools/aeroview/viz_heading.py \
    --gt /mnt/d/aeroview/labelled/WildBox_val_paper.json \
    --image-root /mnt/d/3DBOX/papersubdata \
    --video <VIDEO> --n 14 --crops --out /mnt/d/aeroview/viz_<name>

# GT vs a trained model: add
    --preds <run>/eval/inference/iter_final/WildBox_val/instances_predictions.pth --score-thresh 0.25
```

**Why it exists.** NHD / 3D IoU / BEV AP are EXACTLY 180-flip-blind, so a systematic head/tail inversion is
invisible to the entire evaluation stack. The only way to catch it early is to look. Green = GT, magenta =
prediction, **arrowhead = the HEAD**.

**Full-frame mode is for context only.** Animals are ~142 px in 1920x1080, so a heading is not legible at
frame scale. **`--crops` emits a per-animal contact sheet** (zoomed, `--crop-pad` x the box) and that is the
mode to verify with.

**Reading rule.** Arrows with `|sin alpha| < 0.35` are drawn DASHED and titled `END`. Near end-on the flank
bit is degenerate — **never score a left/right disagreement on a dashed arrow as an error.** A real failure
looks like SOLID arrows pointing at tails across many animals.

**Two things the tool does NOT need — this is the point.** No VGGT artefact, no `cameras.json`, no
`tracking_summary.json`. `up` is recovered exactly from the annotation itself: Omni3D stores `dims=[W,H,L]`
with H on the box's local Y, so `R_cam[:,1]` IS the vertical axis — **measured over 2,491 val frames, every
box in a frame agrees to 0.00 degrees.** For predictions the same field is read off the predicted `pose`.

**The projection is the part that is easy to get wrong.** alpha is ALLOCENTRIC, so it does not map to a fixed
image angle — the same alpha points differently depending on where the animal sits in frame. Recover the 3D
direction first, then differentiate the projection:
`r` = horizontal ray to the animal, `s = up x r`, `d = cos(a) r + sin(a) s`, and
`image_dir ~ ( fx (d_x - (p_x/p_z) d_z), fy (d_y - (p_y/p_z) d_z) )`.
Treating alpha as a raw image angle is wrong everywhere except the principal point.

**⚠ THE FIRST VERSION OF THIS TOOL DREW EVERY ARROW BACKWARDS, AND I "VALIDATED IT BY EYE" AND SAID IT WAS
CORRECT. The user spotted it; my eyeballing did not.** Root cause: `up` sign (see above). What makes this the
single most instructive bug in the project — the printed flank label is computed from `sin(alpha)` DIRECTLY and
was always right, while the arrow used the reconstructed basis and was inverted. So the picture disagreed with
its own caption, and every by-eye check I ran simply mis-read which end of a rhino was the head.

**THE OBJECTIVE TEST — always run this instead of eyeballing.** Geometrically `left = up x forward`, so a
visible LEFT flank forces the head to project image-LEFT (for a camera whose up is image-up, near the
principal point). Therefore `sign(u_x)` from the projected arrow MUST agree with the printed flank label:

| up convention | agreement, all 7,460 | agreement, usable \|sin a\|>=0.35 (n=5,358) |
|---|---:|---:|
| `up = -R_cam[:,1]` (**correct**) | **99.8%** | **100.0%** |
| `up = +R_cam[:,1]` (the bug) | 0.3% | 0.0% |

0.0% vs 100.0% — this test is decisive and takes seconds. **Never accept a heading render on eyeballing again.**

**THE 2D BOXES WERE NEVER WRONG** (also investigated, also on user report). Two independent checks:
`bbox` is XYWH and reproduces `bbox2D_tight` exactly (maxdiff 0.0); and the 2D box centre matches the
projected 3D centre to a **median 9.6 px = 6.1% of box size, 95.4% within 25%** over all 66,951 annotations.
The apparent misalignment was a RENDERING CHOICE: only ~11% of val annotations carry a heading label, so
drawing labelled animals alone left most animals in a frame unboxed and the eye landed on an unboxed
neighbour. One case that looked badly offset was a labelled rhino CALF with the unlabelled adult beside it.
Fixed by drawing every other annotated animal dim-blue as context (`--no-context` to disable).

**Still true and still visible:** the rhino degeneracy — nearly every solid arrow in a val rhino sheet points
the same way, which is the R=0.847 concentration that lets a constant score 98.3% on rhino val.

**Use it at three points:** (1) NOW on GT, as a label audit per species; (2) immediately after the first
training run, GT vs prediction on the same frames — look for solid arrows pointing at tails; (3) on the
held-out val species, since a systematic inversion there is exactly what the flip-blind metrics cannot report.

---

# ★ ARCHITECTURE DECIDED (2026-08-20): **GroundCast** — see `GROUNDCAST_DECISION.md` (same dir; backup in /mnt/d/aeroview/plan_backup/)

11-agent research workflow (6 exploration sweeps -> metric adjudication -> 3 biased designs -> arbiter), with
the user's two objections (hidden feet; telemetry sensitivity under zoom: dZ/Z = 6.2%/deg pitch bias at 15.8
deg, 10%/5m terrain) injected and adjudicated. One idea applied twice: **factorise each output along what is
identifiable from where; compute the computable factor exactly; learn only the residual.**
- **Depth = anchor x plane-structure x residual.** The telemetry plane is a DIFFERENTIABLE PRIOR MEAN, not a
  read-off (that is what the objections forced); a per-image anchor head (init = the median(z)=1 convention on
  own detections) learns the gauge; a per-RoI residual absorbs hidden feet / mounds / lying animals.
- **Orientation = axis x sign.** KEY: the boxes carry the body axis MOD pi on ALL 237,505 annotations (PCA
  sign-arbitrary but axis-valid) => dense doubled-angle axis supervision (24x expansion, gated by S1b audit)
  + the 9,887 sparse labels spent ONLY on the sign bit, via an antipodally-tied von Mises mixture.
- **Depth-FM verdict (adjudicated, user objection 4):** REJECTED as scene model / input channel — arithmetic:
  1.24% intra-object depth signal vs >=2% AbsRel at ~9 patches on target; where the plane works the FM cannot
  beat it, where the plane fails is the FM's documented telephoto weakness. FM survives ONLY as a gated
  training-time distillation arm (S2d gate) + a non-circular audit channel. Foundation models carry the
  APPEARANCE side (frozen DINO backbone; DINOv3 sign probe) — geometry is computed, not learned, because
  telemetry makes it exactly known.
- Arbiter verified code itself: GRAZE's bottom-centre claim WRONG (`center_cam` is the VGGT CENTROID,
  prepare_wildbox_dataset.py:65,431,448,515; commit 22a266a converts only released KITTI labels) => the
  (H/2)*n_hat contact lift is required. IMS_PER_BATCH=4 => ~0.9 labelled sign instances/iter.
- **Build: S0 intrinsics+telemetry gates -> S1 label/axis audits -> S2 zero-training precursors (incl. the
  decisive plane-vs-head test and the FM gate) -> S3 depth arm -> S4 orientation arm -> S5 calibration ->
  S6 ranker (gated) -> S7 conditional -> S8 5-seed headline + KABR.** S0-S2 are CPU/cheap and runnable now.
  ~95-130 A40-h total. Falsifiers pre-registered (telemetry join p50>7 deg kills the central claim, etc.).
- First result banked meanwhile: **elephant sign 96.2% vs 78.7% floor, CI [92.7,98.6], 64 tracks, R=0.608**
  (alpha_s0 run, 1 seed); contact sheet visually clean. Zebra +23.1 not resolvable (22 tracks); grevys/rhino
  uninformative by construction. alpha-off control still owed when the A40 frees.

## RESOLVED (2026-08-20): the out-of-plane boxes are CLUSTER LABEL FAILURES, not lying-down animals
Measured (plane through box bottoms per segment, offset in units of own height H): |off|>0.5H — rhino 45.0%,
gazelle 56.7%, plains 29.4%, grevys 23.5%, elephant 15.7%, giraffe 9.1%. Visual inspection of the 18 worst
rhino cases (offsets +5.7H..+9.9H): EVERY one is a STANDING rhino in a dense cluster with the box landed on a
body fragment between animals — none lying down, none on mounds. => GroundCast decision Q2 resolved: these
are pseudo-label failures; MASK them from the loss (m_p); no tilt output; deleting the 6D pose head stands.
Also note gazelle 56.7% — gazelle GT is the least plane-consistent, consistent with its known worst-GT status.
