# Animal Heading → side-consistent re-ID  (DetAny3D / WildBox)

**READ THIS FIRST.** Written so someone picking this up cold — or a context-compacted
assistant — can continue without repeating the dead ends or re-learning the traps. Every
number here was measured; the commands to re-measure are given.

---

## 1. Goal

Give each detected animal a **heading**, hence **which flank the camera sees**, so re-ID can
be **side-consistent**. A zebra's left stripes are a *different pattern* from its right, so
matching a left-flank query against a right-flank gallery is a guaranteed miss. The existing
re-ID stack (`dinov3/extract_dinov3_features.py` → per-track DINOv3 crop features +
`gallery.pt`) cannot tell which side it is looking at. **That is the gap.**

---

## 2. THE ARCHITECTURE (current — this is the important part)

```
   2D box  ──►  crop  ──►  FROZEN DINOv3  ──►  small MLP  ──►  heading angle IN THE IMAGE
                                                                        │
                CAMERA (extrinsic+intrinsic)  ───────────────►  lift 2D angle into 3D
                GRAVITY (up)                  ───────────────►  (heading is horizontal)
                                                                        │
                                                                        ▼
                                              world heading  ──►  left = up × forward
                                                                        │
                                                                        ▼
                                       which flank faces the camera  =  THE re-ID TAG
```

**The camera is load-bearing in two places**: it lifts the image angle into a world direction,
*and* it decides visibility (the ray from animal to camera). Note a 3D direction projects to a
**different image angle depending on where the animal sits in the frame** — so the perspective
must be inverted properly, not treated as "image angle == world azimuth". `visibility.py` does
this.

**What we DROPPED: the 3D bounding box** (its rotation `R_cam` and dims). It was only ever a
scaffold for the 2D→3D lift, and it is the one piece built on sand (§4). **Measured: the
box-free path agrees with the box-derived flank on 614/614 = 100%.**
*We did NOT drop the camera — the camera is essential.*

---

## 3. RESULTS (all held out BY TRACK, never by frame)

**Frozen DINOv3 ViT-L/16 (2048-d = CLS ‖ mean-pooled patches) → 512-unit MLP → (cos θ, sin θ).
13 of 68 tracks held out (2400 test / 8785 train).**

```
median angular error :   4.0 deg
180-deg flips (>90)  :   3.1%          <- the failure that matters (it inverts the flank)
FLANK accuracy       :  96.8%          <- the re-ID number   (chance = 50%)
after track-vote     : 100.0%  (13/13) <- but fragile: assumes the animal never turns
tracks fully backwards:  0/13
```

**Cross-species evidence (the strongest result).** The single giraffe track (100 frames, the
only non-zebra) is held out entirely — the model **never sees a giraffe in training**:
* with the buggy background crops (§4) it was **100% backwards**;
* with correct crops it is **0.0% flipped — perfect**.
That is the proof the model looks at the *animal*, not at a scene shortcut.

**Where the remaining 3% lives** (`plot_tracks.py` run-length analysis):
* 6 tracks **clean**; 6 tracks **scattered** (flip runs of 1–3 frames) → filterable noise;
* **1 track** (`DJI_..._0007_V/seg1 track 4`): 56 flips in runs up to **20 frames** →
  *confidently backwards for ~1–2 s*. A filter cannot fix that. Needs inspection
  (`plot_crops.py`) — likely sustained head-on views where head-vs-tail is not visible.

**Geometry (the scaffold, independently verified):** on 11,084 human-labelled instances,
`front ∈ our 4 horizontal candidates` = 100%, `TOP` from geometry = 100%,
`left = up × forward` = 100%. **17 tests green.**

---

## 4. ⚠️ TRAPS THAT ALREADY BIT US (do not re-fall)

### 4a. The 518-vs-1920 scale trap  ← *cost us a completely fake result*
VGGT runs at long-side **518**. So `cameras.json` intrinsics **and**
`tracking_summary.json`'s `bbox_2d` are in **518×294** space, while the frames on disk are
**1920×1080**. Nothing announces this — `cameras.json` *reports* `image_width: 518`, so it
looks self-consistent. (The SAM3 masks, meanwhile, ARE full-res — that's how it was caught.)

Cropping with an un-rescaled `bbox_2d` cuts a patch **3.7× too small in the top-left corner**
— **pure background**. And it fails **silently**: a model trains happily on it and even scores
well, because the wrong crop location is still a deterministic function of the animal's true
position, so it learns **scene/drone geometry**. It reported **"7.0° / 91.8% flank" having
never seen an animal.** *That number was void.*

**Fixed once, at the source: `io.py::_frame_scale`. DO NOT REMOVE IT.**
Verified: projected 3D box centre now lands 24 px from the SAM3 mask centroid; crop animal
size went 69 px → **256 px**; `crops.npz` 136 MB → **346 MB** (background compresses to
nothing; a zebra doesn't).

### 4b. Ambiguous track keys
`seg1` exists in **several videos**. Grouping or selecting by the bare string `seg1` merges
different animals, and made a failing track look clean. Always key on the **full segment
path**; `plot_*.py` now refuse to guess and print `<video>/<seg>`.
(A GT-smoothness check that grouped this way "found" 30% broken labels — it was my own bug.
Correctly grouped, **GT jumps are 1.11%**: the labels are fine, and the 3% flips are real
model errors.)

### 4c. DetAny3D's rotation is untrained (documented, NOT fixed — agreed)
* `chamfer_loss` is **absent** from `wildbox_final.yaml:54`'s `loss_list`, so the 6D rotation
  head gets **zero gradient** — yet it is what eval/export ships as the box `pose`
  (`train.py:193`, `train.py:310`).
* Even if trained, chamfer is a permutation-invariant 8-corner set distance → **180°-blind**.
* The GT yaw is a hardcoded **`0.0`** (`data_creator/wildbox.py:270`), and `alpha` — the one
  head that IS heading-aware (12 bins over 2π) — is trained against it (`train.py:108-110`,
  loss at `:182`), so it learns "yaw ≈ 0" for everything.
* The GT rotation itself has **arbitrary PCA/SVD signs** — 12.4% of consecutive frames flip an
  axis (`vggt/demo_viser_tracking.py:675`).
* **No metric can see any of this**: BEV AP uses a convex hull, NHD a symmetric corner
  Hausdorff — both 180°-invariant. So **the published APs are NOT corrupted**, but **no box
  rotation here can be trusted for front/back.** This is why §2 drops the box.

---

## 5. WHAT DOES NOT WORK (measured — do not retry)

* **GroundingDINO "head" prompting.** On aerial-oblique footage it grounds the noun onto the
  **whole animal**: the 100%-of-crop box scored highest for *every* prompt (`head.`,
  `animal head.`, `zebra head.`, `head of a zebra.`). Abstained 3/6, wrong 3/3. **Domain
  shift, not resolution** — failed at 149 px as well as 69 px. Code deleted.
* **Mask-shape taper** ("head end narrower, hindquarters bulkier"). 97% on ONE segment, but it
  is a function of viewing pitch/yaw, not anatomy. Does not generalise.
* **Motion direction.** `velocities` are never populated, and differencing `centers` gives
  nothing: the animals **graze** — ~10–25% of a body length across a whole 200-frame segment.
* **Temporal smoothing in IMAGE space** (`track.py`). Took flank **96.8% → 96.7%** (no gain).
  Why: image-space heading **is not smooth** — the drone moves, and near head-on views the
  projected heading is ill-conditioned, so the angle legitimately swings. Smoothing a
  non-smooth signal blurs it. **The physically smooth quantity is WORLD azimuth** — so
  temporal filtering belongs in `visibility.py`/`predict.py` (which have the camera), not in
  the feature-space eval (`features.npz` carries no geometry).
  *(An earlier neighbour-chained de-flipper was far worse — 96.8% → 79.0% — because a bad seed
  frame cascaded through the whole track. The doubled-angle version in `track.py` is correct
  in principle and passes synthetic tests; it just has nothing to gain in image space.)*

---

## 6. WHAT DOES WORK: the labels already existed

The VGGT annotator produced human face labels that were **never wired into anything**:
* `track_face_locks.json` — track-level `{front, top, left}` → face id (8 segments)
* `manual_labels.json` — **per-frame**, all six faces (26 segments)

**Do not train on a face id** — face ids rename themselves whenever the PCA signs flip. We
convert each label to an **image-space heading angle**:
```
GT angle = atan2( project(front_face_centre) − project(box_centre) )
```
a property of the *picture*, not of the box's bookkeeping. **Verified:** the human face id
flips with the sign chaos while our derived angle stays smooth (**1.11% jumps**).

**Two rotation sets exist — never mix them** (`io.py` forces you to choose):
* `raw` = `vggt_results/tracking_summary.json` — per-frame PCA signs. **WildBox was built from these.**
* `canonical` = `vggt_results/annotations/tracking_summary.json` — annotator sign-aligned.
  **The face-locks index into THESE.** Validation must use `canonical`.

---

## 7. DATA

**Trainable labels: 11,185 samples / 68 tracks — 11,085 zebra + 100 giraffe.**

Stranded (worth ~1,200 more samples, 4 species): **rhino (12 tracks, 921) + elephant (3, 286)**
live in the **CUT3R tree** (`/mnt/d/3DBOX/Results_13_04_26_CUT3R/.../corrected_labels/semantic_faces/`),
which has a `camera/` **directory** and **four competing `tracking_summary.json`** files
(`root`, `botsort`, `bytetrack`, `retracked`). **I refused to guess which one the labels key
to** — a wrong guess is silently wrong supervision. Resolve by matching box counts / track ids
to the label keys. **This is the cheapest next win: 4 species, zero new labelling.**

Animal size in WildBox (max side, px): giraffe 692 · plains_zebra 228 · elephant 250 ·
rhino 139 · grevys_zebra 99 · gazelle 98. **The labelled zebras are the small end (median 69 px)** —
i.e. the validation set is the *hardest* slice, not a representative one.

---

## 8. WHERE THINGS RUN (the parity trap)

* **Human annotations + frames exist ONLY on local `/mnt/d/3DBOX`.** There are **no**
  `semantic_faces/` annotations anywhere on the cluster (checked).
* **DINOv3 exists ONLY on the cluster** (`/storage3/3DOM/vshukla/dinov3`, env `envs/dinov3`).
* ⇒ **Ship the crops, not the dataset**: `extract_crops.py` locally → one `crops.npz`
  (346 MB, JPEG-encoded inside the npz) → `scp` → cluster runs DINOv3.

Unlabelled WildBox on the cluster:
`/storage3/3DOM/vshukla/sam3/wd_data/wildbox/{archive/*,data*}/WildBox_vggtv1/WildBox/<vid>/<seg>/`
zebra sets: `archive/data2023KABRZebras` (30 segs), `archive/202401KZebras` (9), `archive/data202307KZebras` (4).

**GPUs:** `gpu-A40` is saturated. **`gpu-1080` node7 has a DEAD GPU** (nvidia-smi lists 7 of 8)
→ `Error 101: invalid device ordinal`. **Use `gpu-V100` (node8)** — works. Avoid `gpu-K80`
(Kepler `sm_37`; PyTorch 2.x dropped support). matplotlib is NOT in the `dinov3` env.

---

## 9. FILES

```
tools/heading/
  conventions.py     corner/face geometry — SINGLE SOURCE OF TRUTH.
                     NOTE: upstream's face table is WRONG (vggt/build_canonical_atlas.py:47
                     calls faces 0-3 "+X,-X,-Y,+Y"; from the corners they are -Y,+Y,-X,+X).
                     Correct: 0:-Y 1:+Y 2:-X 3:+X 4:+Z 5:-Z
  frame.py           world-up + front-face -> forward/up/left   (verified 100%)
  visibility.py      *** flank from heading + CAMERA + gravity, NO 3D box (100%, 614/614) ***
  io.py              loaders; raw-vs-canonical; **the 518->1920 rescale (§4a)**
  dataset.py         face label -> image-space heading angle (the training target)
  build_manifest.py  [CLI] scan all annotations -> manifest.json
  extract_crops.py   [CLI, LOCAL/CPU] crops -> crops.npz     (square-pad: a plain resize
                                                              changes ANGLES = training on a lie)
  extract_features.py[CLI, CLUSTER/GPU] frozen DINOv3 -> features.npz
  train_head.py      [CLI] small head + per-track flip breakdown + track-vote
  track.py           temporal filter — NO GAIN in image space (see §5)
  plot_tracks.py     [CLI] per-track heading vs GT; scattered-vs-contiguous run lengths
  plot_crops.py      [CLI] whole-track filmstrip, arrows drawn on each crop
  tests/             17 tests, zero-dependency: python -m tools.heading.tests.test_frame
```
**`tools/__init__.py` is load-bearing** — a `tools` package exists in site-packages, and Python
resolves regular packages before namespace ones, so without it `import tools.heading` binds to
the wrong package.

---

## 10. RUN BOOK

```bash
# LOCAL (annotations + frames live here)
python -m tools.heading.build_manifest --root /mnt/d/3DBOX --out data/heading/manifest.json
python -m tools.heading.extract_crops  --manifest data/heading/manifest.json --out data/heading/crops.npz
#   -> also copied to D:\detany3d\heading\crops.npz for scp from Windows

# CLUSTER (DINOv3 lives here)
srun --partition=gpu-V100 --gres=gpu:1 --mem=32G --cpus-per-task=4 --time=00:30:00 --pty bash
conda activate /storage3/3DOM/vshukla/envs/dinov3
python -m tools.heading.extract_features --crops data/heading/crops.npz --out data/heading/features.npz --device cuda
python -m tools.heading.train_head  --features data/heading/features.npz            # honest eval
python -m tools.heading.plot_tracks --features data/heading/features.npz            # where it fails
python -m tools.heading.plot_crops  --features data/heading/features.npz --crops data/heading/crops.npz \
                                    --track '<FULL::key>' --out data/heading/track.jpg   # why it fails
python -m tools.heading.train_head  --features data/heading/features.npz --all-data --save data/heading/head.pt
```

---

## 11. NEXT STEPS (in value order)

1. **Scale up the experiment.** Run `visibility.py` over MANY unlabelled tracks (the zebra sets
   in §8) and check the flank tag is temporally stable and matches eyeballing. This is the
   "does the heading descriptor actually tell us what part of the animal we see" test.
2. **Unstick the CUT3R rhino/elephant labels** (§7) → 4 species, no new labelling.
3. **World-space temporal filtering** inside `visibility.py` (image-space failed, §5). Should
   erase the scattered flips → flank ~98.7%+, while still handling a turning animal (unlike
   the track-vote).
4. **Inspect `seg1 track 4`** with `plot_crops.py` — is it head-on (information limit) or a
   real blind spot? Decides whether a better cue (dense patch tokens instead of mean-pooled;
   SAM3 mask as a channel) is worth it.
5. **Fold into DetAny3D**: replace the `0.0` yaw at `wildbox.py:270` with the predicted heading
   so `loss_3d_alpha` (already live, already 180°-aware) finally trains. One model emits
   box + heading, no DINOv3 at inference.
6. **Prove the point**: re-ID accuracy on the existing `gallery.pt` **with vs without**
   side-consistent matching. That is the number that shows heading buys something.

## 12. Standing lesson
The first result (91.8%) looked great and was **entirely fake**. It was the **giraffe** — the
one datum that didn't fit — that exposed it. Distrust clean numbers; check the outlier.
