# Animal Heading for DetAny3D → side-consistent re-ID

**Status doc. Read this first.** It is written so that someone (or some context-compacted
assistant) picking this up cold can continue without repeating the dead ends. Every claim
here is measured, and the measurements are named so you can re-run them.

---

## 1. The goal

Make monocular 3D detection emit a **semantically meaningful animal heading** (which way
the animal faces → **which flank the camera sees**), so re-ID can be **side-consistent**.

Wildlife re-ID is viewpoint-dependent: a zebra's left stripes are a *different pattern*
from its right. Matching a left-flank query against a right-flank gallery is a guaranteed
miss. The existing re-ID stack (`dinov3/extract_dinov3_features.py` → per-track DINOv3
crop features + `gallery.pt`) has no idea which side it is looking at. **That is the gap.**

The deliverable is: **DetAny3D outputs box + faces**, i.e. front / back / left / right / top.

---

## 2. The diagnosis: the heading signal is absent end-to-end

All verified in code, not assumed:

| Layer | State | Evidence |
|---|---|---|
| GT rotation | **No heading.** Raw PCA/SVD fit; singular-vector signs arbitrary. **12.4% of consecutive frames in a track flip an axis** — impossible for a real animal. | `vggt/demo_viser_tracking.py:675` |
| GT yaw scalar | **Hardcoded `0.0` placeholder** | `detect_anything/datasets/data_creator/wildbox.py:270` |
| `alpha` head (12 bins over 2π — *is* heading-aware) | **Actively trained against that placeholder.** `gt_alpha = -atan2(cx−cx_K, fx) + 0.0` → pure viewing-ray geometry, zero orientation content. Learns "yaw ≈ 0" for everything. | `train.py:108-110`, loss at `train.py:182` |
| 6D rotation head | **Zero gradient** — its only loss (`chamfer_loss`) is absent from `wildbox_final.yaml:54`'s `loss_list`. Also 180°-blind (permutation-invariant corner set). Yet it is what gets exported as the box `pose`. | `train.py:193`, `train.py:310` |
| Metrics | **Cannot see heading.** BEV AP uses a convex hull; NHD a symmetric corner Hausdorff. Both invariant to 180° flips. | `bev_ap_eval.py:61`, `class_agnostic_eval.py:41` |

⇒ **Published AP is not corrupted** (every metric is flip-invariant), **but no box rotation
in this project can be trusted for front/back.** Heading must be *re-derived*, not read off
the box.

**The good news:** DetAny3D *already has the head we need*. `alpha` is 12 bins over a full
2π (a 180° flip lands in a different bin and IS penalised) and `loss_3d_alpha` is already
live. Nobody ever gave it a heading to learn. **The final fix is data-side, not
architectural.**

---

## 3. What is SOLVED: the geometry (exact)

`tools/heading/{conventions,frame}.py`. **17 tests green** (`python -m tools.heading.tests.test_conventions`, `..test_frame`).

**The upstream face table is WRONG.** `vggt/build_canonical_atlas.py:47` labels faces 0-3 as
`+X,-X,-Y,+Y`. Derived from the actual corner coordinates they are **`-Y,+Y,-X,+X`**. Faces
4/5 are right. We derive the table from the corners so the mistake cannot propagate.
Correct mapping: **`0:-Y  1:+Y  2:-X  3:+X  4:+Z  5:-Z`**.

Measured on the human annotations (**11,084 instances, 8 segments, 66 tracks**):

```
the true front is among our 4 horizontal candidates : 100.00%
up / TOP, from geometry alone                       : 100.00%
left = up x forward, given the front face           : 100.00%
```

⇒ **The geometry is exact, and the ENTIRE problem is one bit: which of the 4 horizontal
faces is the head end.** Left/right is then *derived* (`left = up × forward`), never guessed.

**Do NOT reintroduce a "body axis = longest horizontal axis" heuristic.** It was tried; it
is wrong 5.6% of the time and wrong in *every frame* for 3 tracks whose PCA box is genuinely
wider than long. The cue picks the FACE directly, which fixes axis and sign in one step.

---

## 4. What DOES NOT WORK (measured — do not retry)

* **GroundingDINO "head" prompting.** On aerial-oblique drone footage it grounds the noun
  onto the *whole animal*: the 100%-of-crop box scored highest for **every** prompt tried
  (`head.`, `animal head.`, `zebra head.`, `head of a zebra.`). Abstained 3/6, wrong 3/3.
  **Domain shift, not resolution** — it failed at 149 px as well as at 69 px. Prompt/threshold
  tuning will not fix it. (Code deleted; finding preserved in `tools/heading/__init__.py`.)
* **Mask-shape taper** ("head end narrower, hindquarters bulkier"). 97% on ONE segment, but it
  is a function of viewing pitch/yaw, not anatomy. A cue that works for one camera geometry
  is not a cue.
* **Motion direction.** `velocities` in `tracking_summary.json` are never populated, and
  finite-differencing `centers` yields nothing: the animals are **grazing** — they travel
  ~10–25% of a body length across a whole 200-frame segment, with displacement barely
  correlated with the body axis.

---

## 5. What DOES work: the labels already existed

The VGGT annotator produced human face labels that were **never wired into anything**:

* `track_face_locks.json` — track-level `{front, top, left}` → face id (8 segments)
* `manual_labels.json` — **per-frame**, all six faces (26 segments)

**Do not train on a face id.** Face ids rename themselves whenever the PCA signs flip. We
convert each label to an **image-space heading angle** — the direction, in the crop, from the
animal's centre toward its head:

```
GT angle = atan2( project(front_face_centre) − project(box_centre) )
```

That is a property of the *picture*, not of the box's bookkeeping. **Verified:** the human
face id flips with the sign chaos while our derived angle stays **smooth (1.1% jumps >90°)**
— i.e. the conversion provably undoes it.

**Two rotation sets exist. Never mix them** (`io.py` makes you choose):
* `raw` = `vggt_results/tracking_summary.json` — per-frame PCA signs. **WildBox was built from these.**
* `canonical` = `vggt_results/annotations/tracking_summary.json` — sign-aligned by the annotator. **The face-locks index into THESE.** Validation must use `canonical`.

---

## 6. ⚠️ THE 518-vs-1920 TRAP (this bit me; it will bite you)

**VGGT runs at a reduced resolution (long side 518).** So `cameras.json` stores intrinsics
**and** `tracking_summary.json` stores `bbox_2d` in **518 × 294** space — while the frames on
disk are **1920 × 1080**. Nothing announces this: `cameras.json`'s `image_width/height` fields
*say* 518×294, so they look self-consistent and you assume they describe the JPEG. **They do
not.** (The SAM3 masks, meanwhile, ARE full-res.)

Consuming `bbox_2d` against the full-res frame crops a patch ~**3.7×** too small in the
top-left corner — **pure background**. And it fails **silently**: the crop is a valid image, a
model trains happily on it, and it can even *score well* by latching onto the correlation
between crop location and scene geometry.

**It did exactly that.** A head trained on those crops reported *"7.0° median error / 91.8%
flank accuracy"* **while having never seen an animal.** That number is void.

Fixed once, at the source: `io.py::_frame_scale` rescales intrinsics and `bbox_2d` to the
frame's true resolution at load time. **Do not remove it.** Verified after the fix:
projected 3D box centre lands **24 px** from the SAM3 mask centroid.

**Any result produced before this fix is void and must be re-run.**

---

## 7. The approach (current)

```
DetAny3D  →  3D box (center, dims, R)      [UNCHANGED — the 5-seed results stand]
                   ↓
            animal crop (full-res!)
                   ↓
            DINOv3  (FROZEN — no fine-tuning)
                   ↓
            small MLP head  →  image-space heading angle (cos θ, sin θ)
                   ↓
            geometry (exact, §3)  →  front/back/left/right/top
                   ↓
            FLANK tag  →  side-consistent re-ID gallery
```

Frozen DINOv3 + a small head is **data-efficient**, which is the right regime for ~11k labels.
No backbone retraining, no 15-hour runs.

**Correctness details that are NOT cosmetic:**
* **Square-pad crops before resize.** The target is an ANGLE; a non-uniform resize is not a
  similarity transform — it *changes angles*. Stretching would train the head on a lie.
* **Split by TRACK, never by frame.** Adjacent frames are near-duplicates; a frame split leaks
  the answer and reports a beautiful, meaningless number.
* **FLANK accuracy is the headline metric**, not mean angular error. A 20° error is harmless;
  a **180° flip** inverts the flank and poisons the gallery. Report the flip rate.

---

## 8. Files

```
tools/heading/
  conventions.py       corner/face geometry — SINGLE SOURCE OF TRUTH (corrects upstream)
  frame.py             world-up + front-face -> forward/up/left   (exact; 100%)
  io.py                loaders; raw-vs-canonical; **the 518->1920 rescale**
  dataset.py           face label -> image-space heading angle
  build_manifest.py    [CLI] scan all annotations -> manifest.json
  extract_crops.py     [CLI, LOCAL/CPU] crops -> crops.npz  (data is only on /mnt/d)
  extract_features.py  [CLI, CLUSTER/GPU] frozen DINOv3 -> features.npz
  train_head.py        [CLI] small head; per-track flip breakdown; track-vote
  predict.py           [CLI, GPU] render heading + flank on an UNLABELLED segment
  tests/               17 tests, zero-dependency (`python -m tools.heading.tests.test_frame`)
```

**`tools/__init__.py` is load-bearing** — an unrelated `tools` package exists in site-packages,
and Python resolves regular packages before namespace packages, so without it
`import tools.heading` silently binds to the wrong one.

---

## 9. Where things run, and the data-parity trap

* **Human annotations + frames exist ONLY on the local `/mnt/d/3DBOX`.** There are **no**
  `semantic_faces/` annotations anywhere on the cluster (checked).
* **DINOv3 exists ONLY on the cluster** (`/storage3/3DOM/vshukla/dinov3`, env `envs/dinov3`).
* ⇒ We **ship the crops, not the dataset**: `extract_crops.py` locally → one `crops.npz`
  (~136 MB, JPEG-encoded inside the npz) → `scp` → cluster does DINOv3.

Cluster data trees (note the differing shapes):
* annotated: `.../WildBox_sam3-vggtv1_processed/WildBox/<vid>/<seg>/` (has `sam3_masks/`, `vggt_results/annotations/`)
* unlabelled: `/storage3/3DOM/vshukla/sam3/wd_data/wildbox/{archive/*,data*}/WildBox_vggtv1/WildBox/<vid>/<seg>/`
  — zebra sets: `archive/data2023KABRZebras` (30 segs), `archive/202401KZebras` (9), `archive/data202307KZebras` (4)

**GPU note:** `gpu-A40` is usually saturated. `gpu-1080` node7 has a **dead GPU** (nvidia-smi lists
7 of 8) → `Error 101: invalid device ordinal`. **Use `gpu-V100` (node8)** — it works. Avoid
`gpu-K80` entirely (Kepler `sm_37`; PyTorch 2.x dropped support).

---

## 10. Run book

```bash
# LOCAL (data lives here)
python -m tools.heading.build_manifest --root /mnt/d/3DBOX --out data/heading/manifest.json
python -m tools.heading.extract_crops  --manifest data/heading/manifest.json --out data/heading/crops.npz
scp data/heading/crops.npz fbk-cluster:/storage3/3DOM/vshukla/DetAny3D/data/heading/

# CLUSTER (DINOv3 lives here)
srun --partition=gpu-V100 --gres=gpu:1 --mem=32G --cpus-per-task=4 --time=00:30:00 --pty bash
conda activate /storage3/3DOM/vshukla/envs/dinov3
python -m tools.heading.extract_features --crops data/heading/crops.npz --out data/heading/features.npz --device cuda
python -m tools.heading.train_head --features data/heading/features.npz              # honest eval
python -m tools.heading.train_head --features data/heading/features.npz --all-data --save data/heading/head.pt
python -m tools.heading.predict --segment <UNLABELLED_SEG> --head data/heading/head.pt \
       --out data/heading/vis/zebra --every 20 --device cuda
```

---

## 11. Results

| run | status |
|---|---|
| Geometry vs human locks | **100%** on 11,084 instances. **Trustworthy.** |
| Heading head, 1st attempt | *"7.0° / 91.8% flank"* — **VOID**: trained on background crops (§6). |
| Heading head, corrected crops | **PENDING** — the number to get. |

The first run's per-track breakdown is still informative about *method*: flips were scattered
(≤3% on most zebra tracks) → **a track-level majority vote works** (an animal's head does not
swap ends mid-track). The one 100%-flipped track was the **giraffe** — the only non-zebra, and
(being a single track) it landed entirely in *test*, so the model had **never seen a giraffe**
and inverted it. Whether that survives the crop fix is unknown.

---

## 12. Next steps

1. **Re-run everything with corrected crops** (§10). The animal is now ~230 px in the crop
   instead of ~62 px, so expect a *different* (and finally meaningful) number.
2. **Species gap.** Trainable labels are **11,085 zebra + 100 giraffe**. Rhino (12 tracks, 921
   instances) and elephant (3 tracks, 286) labels **already exist** but are stranded in the
   **CUT3R tree** (`/mnt/d/3DBOX/Results_13_04_26_CUT3R/.../corrected_labels/semantic_faces/`),
   which has a `camera/` directory and **four competing `tracking_summary.json`** files
   (`root`, `botsort`, `bytetrack`, `retracked`). **I refused to guess which one the labels key
   to** — a wrong guess is silently wrong supervision. Resolving this = 4 species for free.
3. **Track-level vote** into `predict.py` (one heading per animal, not per frame).
4. **Then fold into DetAny3D**: replace the `0.0` yaw placeholder at `wildbox.py:270` with the
   predicted heading and let `loss_3d_alpha` finally train. One model emits box + heading, no
   DINOv3 at inference.
5. **Prove the point**: re-ID accuracy on the existing `gallery.pt` **with vs without**
   side-consistent matching. That is the number that shows the heading actually buys something.

## 13. Known bug, documented not fixed (agreed)

`chamfer_loss` is missing from `wildbox_final.yaml:54`, so DetAny3D's 6D rotation head gets
**zero gradient** during WildBox fine-tuning, while still being what eval/export ships as the
box `pose`. Not a blocker (we ignore the box rotation entirely) and it does **not** corrupt the
published APs (all flip-invariant), but it is real.
