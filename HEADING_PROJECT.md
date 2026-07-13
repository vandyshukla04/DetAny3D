# 3D animal heading — project state

**READ THIS FIRST.** Written so someone picking this up cold — or a context-compacted assistant —
can continue without re-deriving anything.

**Goal.** Give monocular 3D detection a *3D world-space heading* for each animal → which flank the
camera sees → **side-consistent re-ID** (a zebra's left stripes are a different pattern from its
right, so a left-flank query must not match a right-flank gallery).

**Shape of the solution.**
```
3D box            →  the animal's body axis, in WORLD space   (but UNSIGNED — PCA/SVD signs are junk)
DINOv3            →  which end is the HEAD                    (the one thing geometry cannot give)
camera + gravity  →  world heading → visible flank            (visibility.py, already validated)
```
The camera is **essential** — it is what puts the heading in world space and decides visibility.
Only the box's *rotation* is ever discarded.

---

## 1. Dataset of record: `papersubdata`

`/mnt/d/3DBOX/papersubdata/{elep1,2,3 · rhin1,2 · zebr1,2,3 · gira1,2}` — **`gaze1` excluded** (user).
**60 videos · 305 segments · 4 species.** Per segment: `frame_*.jpg` (1920×1080), `cameras.json`,
`tracking_summary.json`, `kitti_labels/`.

**CUT3R is dropped** (user: *"forget cut3r inventory!! work with the papersubdata only!!"*).

### Verified coordinate facts (do not re-derive)
| | |
|---|---|
| `centers`, `rotation_matrices` | **WORLD** |
| `extrinsic` | 3×4, **world → camera** |
| `cameras.json` intrinsics | **FULL-RES** (fx≈1266, cx=960, cy=540) |
| `tracking_summary.bbox_2d` | **518-space** — ×(width/518)=3.707 to reach full-res |
| world scale | **VGGT-arbitrary** ⇒ every threshold is in **body-lengths**, never metres |
| proof | projecting `centers` with `cameras.json` → (870.8, 577.7); kitti 2D box centre → (870.7, 580.6) |

### ⚠️ THE SCALE TRAP (this bug already cost us once)
Reading `bbox_2d` as full-res cut **every crop from the background**. The model then reported a
confident **"91.8 % flank accuracy"** having never seen an animal. That result is VOID.
`papersub.Segment.scale` now **measures** the ratio (by projecting the 3D centres) and **raises**
on a mismatch. Never hardcode 3.707. Never use `io.py` (WildBox tree) to cut crops.

---

## 2. The core insight — animals label their own heading

A walking animal's **world velocity is its heading**, and unlike the box it is **signed**.
Measured over all 305 segments:

| | |
|---|---|
| **Motion-derived world headings** (`autolabel.py`, after both gates) | **25,554**, zero human annotation |
| per species | elephant 7,229 · rhino 10,914 · zebra 5,740 · giraffe 1,671 |
| **Box-vs-motion agreement** (independent signals!) median cos | **0.984** after gating; **0.953** raw (rhino .979, eleph .969, gira .934, zebra .818) — random = 0.707 |
| tracks that walk | 383 / 1,094 (35 %) |
| **"longest horizontal axis" shortcut is WRONG** | **13 %** of the time (zebra 26 %) → caps any sign-only cue at 87 % |

**LOCOMOTION IS THE TEST SET, NOT THE METHOD.** (User: *"I don't want to depend on locomotion."*)
Nothing at inference depends on the animal moving. These 25 k labels exist to **score** a heading
predictor on 4 species and 60 videos for free. The predictor must work from a **single frame on a
standing animal**.

**Visually verified** (`preview_labels.py` → `/mnt/d/detany3d/heading/labels_preview.jpg`): the green
arrow lands on the **head** — elephant trunks, rhino horns, giraffe necks, and *grazing* zebras with
their heads down. Animals span 80–520 px. **This is the check we skipped last time and paid for.**

---

## 3. Target: the ALLOCENTRIC angle (not an image angle)

A crop determines the animal's orientation **relative to the viewing ray**, not relative to the
image — the same animal at the left and right edge of a frame *looks identical* but projects to
different image angles. Regressing an image angle therefore asks the net to infer what the pixels
do not contain. **That was the flaw in the previous attempt.**

`papersub.Segment.allocentric_basis()` builds `r` = horizontalised camera→animal ray, `s = up × r`;
`alpha = atan2(h·s, h·r)`. Round-trip is unit-tested.

**This is exactly DetAny3D's `alpha` head** — which already exists, already has a live
`loss_3d_alpha`, and is currently fed a **hardcoded `0.0`** (`data_creator/wildbox.py:270`). That is
the Stage-4 drop-in.

---

## 4. Why the box's heading cannot be trusted (all verified in code)

| layer | state |
|---|---|
| GT rotation | raw PCA/SVD fit; **axis signs arbitrary** (12.4 % of consecutive frames flip an axis) |
| GT yaw | **hardcoded `0.0`** — `data_creator/wildbox.py:270` |
| `alpha` head (12 bins over 2π, heading-aware) | trained against that placeholder → learns "yaw ≈ 0" |
| 6D rotation head | **zero gradient** — `chamfer_loss` absent from `wildbox_final.yaml:54` `loss_list`; also 180°-blind |
| metrics (BEV AP, NHD, 3D IoU) | **all flip-invariant** ⇒ **published APs are NOT corrupted**, but no box rotation is usable for front/back |

## 5. What DetAny3D *does* give — the self-contained inference path (agent-verified in code)
| | |
|---|---|
| `pred_K` | **genuinely predicted** (`unidepth.py:611`); `deploy.py:281` already runs on it — **no GT camera needed** |
| `depth_maps` | full **metric depth map**, computed on *every* forward (`wrap_model.py:101`) and currently **discarded**. RANSAC a ground plane on its point cloud → **world `up`** (unprojection exists: `train_utils.py:47-68`) |
| `center_cam` | metric 3D position (`train.py:336`) |
| gravity / up | **absent** — must be derived from the depth map (above) |
| rotation | ❌ broken — **the only thing we throw away** |

⇒ no VGGT, no `cameras.json` at test time.

---

## 6. Dead ends — MEASURED, do not retry
- **GroundingDINO `"head"` prompt** — domain shift; grounds onto the whole animal. Failed at 149 px as well as 69 px.
- **Mask taper / "hindquarters are bulkier"** — view-dependent junk (user was right).
- **Motion at inference** — `velocities` is never populated; grazing animals move 10–25 % of a body length per segment.
- **Image-space temporal smoothing** — image heading is not temporally smooth. Tracker v1 made it *worse* (96.8 → 79.0 %); v2 (doubled-angle) gained nothing.
- **Track-level splits** — LEAKY. A track-split said 96.7 % while the model visibly broke on unseen footage. **Split by VIDEO or nothing.**

---

## 7. Code (`DetAny3D/tools/heading/`)

| file | role |
|---|---|
| `conventions.py` | corners/faces/normals, `FACE_AXIS`, `left_from(up,fwd)=up×fwd`. **Upstream `_FACE_CORNER_INDICES` comments are WRONG**; truth derived from corners: `0:-Y 1:+Y 2:-X 3:+X 4:+Z 5:-Z` |
| `frame.py` | `world_up_from_boxes` (consensus ground normal), `horizontal_faces` (the 4 front candidates), `frame_from_front_face` |
| `papersub.py` | **the loader.** world up, allocentric basis, `scale` (measured!), `crop_box` |
| `autolabel.py` | motion → signed world heading + true front face. **The free test set.** |
| `extract_crops.py` | square-pad **before** resize (a plain resize is *not* a similarity transform — it changes angles). Ships `crops.npz` |
| `preview_labels.py` | **look at the pixels before trusting the numbers** |
| `extract_features.py` | frozen DINOv3 → pooled feats **+ full PATCH GRIDS** (the parts — previously pooled away) |
| `parts.py` | **THE PROBE**: k-means parts → border-touch background rejection → head prototype → 4-way face accuracy (+ `parts_viz.jpg`) |
| `train_head.py` | supervised baseline on the motion labels, **same 4-way metric** → directly comparable |
| `visibility.py` | heading + camera + gravity → visible flank. **Box-free. Validated 614/614.** |
| `model.py` | the MLP, defined once (`HEAD_VERSION` checked on load) |
| `io.py` | ⚠️ **VALIDATION ONLY** — WildBox tree, for the human face-locks. **Never cut crops with it.** |
| `probe.sbatch` | cluster job A→B→C, idempotent, fail-loud |
| `tests/` | `test_papersub.py` covers the allocentric round-trip + the real 518-scale check |

`tools/__init__.py` is **load-bearing** — a `tools` package exists in site-packages and Python
resolves regular packages before namespace packages.

---

## 8. THE OPEN QUESTION (the probe answers it)

> Do DINOv3 patch tokens contain a stable **head part**, and does it pick the right end of the body
> axis — from a **single frame**, with **no motion**?

Scored for free on the walking crops (we already know the answer there). **4-way, chance 25 %.**
Calibration = "which cluster is the head", **one choice per species = 4 bits**, made on calibration
videos and frozen; accuracy reported on **held-out videos**.

- **≫ 25 %** → build the heading on DINOv3 parts. No locomotion dependence at all.
- **~ 25 %** → parts are dead. **Say so and pivot** — as we did with GroundingDINO and the taper cue.

`parts.py` (training-free) and `train_head.py` (supervised on the motion labels) report the **same
4-way number on the same crops**, so they can be put side by side.

**Run:** `sbatch tools/heading/probe.sbatch` (needs `data/heading/crops.npz`).

## 9. After the probe
1. World heading = body axis signed by the DINOv3 cue → `visibility.py` → flank tag.
2. **Human-label check:** the 3 face-lock videos (`zebr3/…0007_V`, `zebr3/…0008_V`, `gira1/…0007_V`)
   are **inside papersubdata but excluded from training**, and are mostly **stationary grazing**
   animals — the exact generalisation that matters.
3. **Label-free physics check** on all 305 segments: a flank may only switch at **low** confidence
   (head-on / tail-on). A switch at high confidence is impossible ⇒ free error detector.
4. **Stage 4 — the "improved DetAny3D":** replace the `0.0` yaw at `data_creator/wildbox.py:270`
   with the derived heading ⇒ the `alpha` head finally gets real signal ⇒ DetAny3D emits heading
   **natively**, no DINOv3 at test time. **3D/BEV AP must not regress** vs the 5-seed baseline
   (**3.97 ± 0.65**).
5. **Re-ID payoff:** `gallery.pt` accuracy **with vs without** side-consistent matching.
