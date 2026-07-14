# 3D animal heading — WHERE WE ARE  *(compaction-proof handoff)*

**READ THIS FIRST.** Written so someone picking this up cold — or a context-compacted assistant —
can continue without re-deriving anything. Companion docs: **`METHOD.md`** (the full report: method,
formulae, dataset, every measured number, the progression including the failures) and the generated
**`REPORT.md`** (numbers read straight from `results.json`).

---

## 1. The goal

Give every detected animal a **3D heading in the world**, and from it the **viewpoint tag** re-ID
needs: *"we are seeing this animal's LEFT flank, 0.85 broadside, slightly from behind."*
Wildlife re-ID is side-dependent (a zebra's left stripes ≠ its right), so a left-flank query against
a right-flank gallery is a guaranteed miss.

**Constraint held throughout: no human orientation labels.**

## 2. The method, in one box

```
LOCOMOTION  names the head      free & signed, but only while the animal WALKS (14% of images)
DINOv3      transports the name frozen, dense; asked ONLY for front-vs-back
GEOMETRY    places it in world  body axis, ground plane, LEFT/RIGHT (= up x forward), the lift
```
1. A walking animal's world velocity **is** its heading → free labels.
2. The **template** = the mean DINOv3 profile along the body axis (rump→head, 5 bins), **per species**.
   **Nothing is trained.** It is an average.
3. For a new box: **geometry proposes the axis**, appearance **disposes the sign** — score the two
   ends of the body axis against the template.
4. α (the allocentric angle) → the **viewpoint tag**.

⚠️ **The heading is QUANTISED to the box's horizontal face normals.** We resolve *which end of an
axis*; we do **not** regress a free angle. Never call it "continuous" — a reviewer will catch it.

## 3. Key formulae

```
r = horizontalised camera→animal ray;  s = up × r;  (r, s, up) right-handed
α = atan2(h·s, h·r)                    h = cos α·r + sin α·s        (allocentric angle)

left = up × h = cos α·s − sin α·r ,  camera lies at −r from the animal ⇒
    left·(−r) =  sin α    →  sin α > 0 : LEFT flank visible ; |sin α| = how broadside
       h·(−r) = −cos α    →  cos α < 0 : we see its FACE (else REAR)
    |sin α| → 0           →  head-on: NO flank exists. Abstaining is the honest answer.
```
*Verified against the independent construction (`left = up×forward`, `forward·to_camera`) over 200
random cameras: exact 200/200.* Flank accuracy is only ever reported where **|sin α| ≥ 0.35**.

## 4. Data — WildBox (`/mnt/d/3DBOX/papersubdata`, gazelles excluded)

**60 videos · 305 segments · 52,443 frames · 1,115 tracks · 177,973 animal images** (3.4 animals/frame).

| | eleph | giraffe | rhino | zebra | total |
|---|---:|---:|---:|---:|---:|
| animal images | 42,930 | 3,688 | 45,853 | 85,502 | **177,973** |
| **WALKING** → free label | 7,229 | 1,671 | 10,914 | 5,740 | **25,554** (14%) |
| crops kept (stride 2) | 3,614 | 828 | 5,457 | 2,863 | **12,762** |
| → build the template | 587 | 187 | 285 | 440 | **1,499** |
| → **held out, walking: TESTED** | 722 | 30 | 4,157 | 859 | **5,768** |
| standing crops produced | 2,211 | 103 | 1,741 | 778 | **4,833** |
| → training videos: **NOT USED** | 1,650 | 93 | 1,216 | 470 | **3,429** |
| → **held out, standing: TESTED** | 561 | 10 | 525 | 308 | **1,404** |

**Only 14% of animal images carry a free label** (the walking ones). The other **86% stand still** —
that is the premise, and what the transfer test probes.
Drone geometry: **median 18° elevation** (p10 9°, p90 30°) — **aerial-OBLIQUE, not top-down**. That
obliquity is what makes a flank visible at all.

## 5. RESULTS (held out by VIDEO; L24/token/224; SAM masks; nothing trained)

| setting | n | med err | @45° | sign | flank* |
|---|---:|---:|---:|---:|---:|
| random sign | 5768 | 80.4° | 46.9% | 46.9% | — |
| **locomotion only (oracle)** | 5768 | **9.1°** | 100% | 100% | — | ← **the FLOOR the boxes impose** |
| appearance, oracle axis | 5767 | 9.4° | 93.5% | 93.5% | — |
| **FULL METHOD** | 5767 | **9.8°** | 87.4% | **87.4%** | **95.0%** |

Per species (sign): rhino 91.9 · elephant 77.3 · zebra 73.8 · giraffe 100 (n=30).
`acc@45 ≡ sign` — the only >45° errors *are* sign flips. **We are 0.7° above the oracle floor.**

**TRANSFER (standing animals):** sign **87.4% → 69.1%** (−18.3), med err 9.8° → 15.2°,
but **flank 95.0% → 89.4%** (only −5.6). Per species (standing): zebra **87.0%** (*improves*),
rhino 69.5, elephant **58.3** (*worst*), giraffe 100 (n=10).

**ABLATION:** `− centring` 87.3 (**no effect**) · `− SAM mask` 86.6 · `− geometric axis prior` 83.6.

## 6. THE THREE OPEN QUESTIONS

1. **Where do the sign errors live?** `experiments.py` now bins sign/flank by **|sin α|**. If they
   pile into `|sin α| < 0.35`, the method is **guessing where it should be abstaining** — and the
   honest headline becomes *"87% on the ~80% of frames where the geometry is readable, abstaining on
   the rest"*, which is stronger than a flat 87%. **Only 1 crop in 5,768 currently abstains, which
   is suspiciously few.** ← RUN THIS FIRST.
2. **Elephant collapses on standing animals** (77.3 → 58.3) while zebra *improves*. Not a uniform
   "standing is hard" effect. Unexplained.
3. **The committed grazer is still untested.** Bridges only capture *brief* stops (median 14 frames).
   The animal that never walks in a segment — the real 86% — has never been evaluated.

## 7. Measured dead ends — DO NOT RETRY

| | |
|---|---|
| GroundingDINO `"head"` prompt | domain shift; grounds onto the whole animal |
| mask taper / "hindquarters are bulkier" | view-dependent |
| **image-space** temporal smoothing | made it **worse** (96.8 → 79.0%). The drone moves, so smoothing the projected angle smooths the **camera**. |
| track-level splits | **leaky**: 96.7% while the model visibly broke. **Split by VIDEO or nothing.** |
| k-means over DINOv3 patches | 16.6% — **retracted as an invalid negative** (global k-means, 14×14 grid, last layer, centroids not correspondence) |
| **Viterbi / temporal decoding** | **+0.3%. NULL.** Errors are **whole-track inversions** (46/72 zebra tracks >80% *wrong*), and a smoothness prior is *designed to preserve* a coherent trajectory. Fixed at the source by L24/token instead. |

## 8. Two priors that were WRONG, and it mattered

| assumed | measured |
|---|---|
| a **mid-layer** wins (parts peak mid-network) | the **LAST layer (24)** wins |
| the **`key`** facet wins (Amir et al., *Deep ViT Features*) | the **`token`** facet wins |

**This was the entire zebra result:** `L12/key 52% · L24/key 52% · L12/token 52% · L24/token 79–81%`.
Only the last-layer token output sees zebras at all. (DINOv3's Gram anchoring targets the **final**
dense features; Amir et al. studied the *original* DINO.) **224 ≈ 448** → resolution is free.

## 9. Silent bugs found (every one produced a *plausible* number)

| bug | it reported | fix |
|---|---|---|
| `bbox_2d` (518-space) read as full-res | **91.8% flank** from a model that had never seen an animal | `Segment.scale` derives it; `check_scale()` asserts against projected 3D centres |
| checkpoint recorded no split | **94.3%** where the truth was **79.8%** | `split_fingerprint()` in every ckpt; evaluators `assert_matches()` or die |
| **DC domination** in the score | 4-way collapsed to **39.4%** while the cue was 83.7% | centre profile+template → score the **gradient**. Margins 0.005 → **0.155** (predicted ≈0.2 *before* the run) |
| `_final_norm` outside `no_grad` | the token facet **never ran** | whole forward under `no_grad` |
| `output_hidden_states=True` | materialised **25** hidden states (~7.7 GB/batch) to use one | hooks capture only requested layers |
| **`np.load` → lazy `NpzFile`** | **4,113 ms per crop** — every `z[k][i]` re-inflated the whole 259 MB array. **~6.6 h to score.** THIS was the slowness, not the GPU. | `load_npz()` materialises once; **every** consumer routed through it |
| `${STAND:+...}` tested the VARIABLE, not the FILE | would have silently produced a report **with no transfer test** | test the file; shout if absent |

**Lesson recorded:** I mis-diagnosed the perf 4× before profiling — and then trusted a profile that
had `crops.batch()` *outside* the timing loop. **A profile that omits a stage is a guess with numbers
attached.**

## 10. Code (`tools/heading/`)

| file | role |
|---|---|
| `papersub.py` | the loader. `ground_basis`, `allocentric_basis`, `azimuth_of`, **`scale` (measured, asserts)** |
| `autolabel.py` | motion → signed world heading. **The free labels.** |
| `bridges.py` | `WALK→STAND→WALK` → **standing** labels (self-validating: before/after must agree within 30°) |
| `descriptors.py` | frozen DINOv3. `grids()` (hooks, any layer/facet), `foreground()`, `axis_profile()` |
| `template.py` | **THE METHOD.** `Accumulator` → `AxisTemplate.score_faces` → `choose()`. `centred=False` for the ablation. |
| `viewpoint.py` | α → the re-ID tag (flank / broadside / face / rear) |
| `experiments.py` | **the paper table** + ablation + transfer + the |sin α| bands |
| `cropset.py` | the ONE accessor for `crops.npz`; `load_npz()` |
| `masks.py` | SAM3 instance masks; the join is **asserted** (centroid vs own 2D box) |
| `paper_fig.py` / `cam_fig.py` | 16 two-row figures / 16 camera+heading figures |
| `run_all.sh` / `final_report.py` | one folder, one scp; **REPORT.md is GENERATED from results.json** |
| `dataset_stats.py` | counts WildBox → `dataset.json` (committed; the cluster has no papersubdata) |
| `tests/` | 27 tests, all silent-failure guards |

## 11. Run it

```bash
# cluster
git pull origin wildbox_detany3d
OUT=data/heading/REPORT_v2 bash tools/heading/run_all.sh
# needs data/heading/{crops.npz, crops_stand.npz}; without the latter the TRANSFER TEST is skipped
# and REPORT.md says so in a banner at the top.
```
`crops.npz` (259 MB) and `crops_stand.npz` (100 MB) are built **locally** (papersubdata is on `/mnt/d`,
DINOv3 is on the cluster) and scp'd up. Rebuild with `autolabel.py` → `extract_crops.py`.

## 12. Not yet done

- **DetAny3D integration.** `data_creator/wildbox.py:270` feeds the `alpha` head a hardcoded `0.0`.
  The head already exists (12 bins over 2π, live loss) — it is simply starved. Feed it our heading.
  **3D/BEV AP must not regress** vs the 5-seed baseline **3.97 ± 0.65**.
- **Human face-locks** (11,084 grazing-zebra instances, 2 videos, permanently held out).
  ⚠️ Join **BY WORLD DIRECTION, NEVER BY FACE INDEX** — papersubdata carries **raw** rotations, the
  locks index **canonical** ones, and **16.9% differ by 180°**. A naive index join inverts head/tail
  on one in six and reports a believable ~83%.
- **The re-ID payoff.** Gallery accuracy with vs without side-consistent matching.
