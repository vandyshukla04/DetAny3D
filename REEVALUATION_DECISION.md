All context is in hand — the surveys, the plan's Phase A measurements, the run-2/run-3 record, and the GroundCast adjudication. The decision follows.

# AEROVIEW RE-EVALUATION DECISION — 2026-08-20

**Verdict in one line: STAY on OVMono3D and run the already-specced run 3 (dims-only token + per-image anchor head); adopt MoGe-2 only as a gated, one-afternoon anchor probe and non-circular audit channel; DetAny3D stays parked as provisioned insurance with a named, falsifiable activation trigger; DINOv3 clean-slate and full assembly are rejected now.**

---

## 1. DECISION MATRIX

Checks: (1) single image in → 3D box + heading out; (2) heading co-equal; (3) elegance = one frozen FM + small heads; (4) gauge honesty (affine currency, no metric binding); (5) compute fit (part-time A40 + V100-16GB, offline, 3-seed affordable); (6) evidence at ~140 px / 15.8° oblique aerial.

| Vehicle | 1 Single-img | 2 Heading co-equal | 3 Elegance | 4 Gauge | 5 Compute | 6 Aerial evidence | Migration (wks) | What transfers | What we lose |
|---|---|---|---|---|---|---|---|---|---|
| **A. STAY: OVMono3D + run-3 (GroundCast S3)** | YES (running) | YES — proven: elephant sign 96.2% CI [92.7,98.6], stable 3 runs | PASS — one frozen DINOv2 + heads; run-3 is a ~2-file diff that *deletes* the shared-trunk token | NATIVE — trained in label gauge; anchor-free z 1.09% = floor | PROVEN — 2.1 GB/batch-8, 2.6 h/15k on V100; 3-seed cheap | The ONLY vehicle with measured WildBox evidence at all (full scorecards) | **0** | Everything | Nothing; residual risks are the 560 px ceiling and the unmeasured single-frame anchor floor |
| **B. DetAny3D migration** | YES (promptable, = our oracle-2D protocol) | Plausible — native 12-bin alpha+residual slot exists, but WildBox yaw target is currently the literal constant 0.0 (`wildbox.py:270`); unproven on our labels | FAIL-ish — three stitched FMs (SAM-H + DINOv2-L + UniDepth) + adapters (survey B's own caveat) | Metric pretrain, but surgery done locally: depth_loss severed, NHD best-scale 0.10 → 0.94 after fine-tune | WEAK — A40-only (~2 it/s b1, ~6 h/epoch), V100 infeasible, DDP broken → 3-seed ≈ 27 h/arm on a contended queue | None published; our own smoke is the only aerial evidence for any Regime-B system | 2–4 to first honest scorecard | Labels/graders, oracle-2D jsons (wired), evaluator glue (green), alpha recipe (slot exists) | V100 trainability, cheap multi-seed rigor, one-FM elegance; our proven 2D detector goes redundant |
| **C. Clean-slate frozen DINOv3 ViT-L + light heads** | YES by design | Recipe transfers (alpha proven on frozen-DINO features) but everything rebuilt | PUREST on paper | PASS — gauge-free representation | OK for features (precompute → V100 head training; disk budget unmeasured) | NONE for the claim that matters: 66.1 frozen-COCO is the 7B + Objects365-pretrained DETR decoder; no frozen-ViT-L 3D result exists; SAT variant is nadir-ortho, wrong regime. Real argument is patch budget: ~3 → 6.5–9 patches across a 142 px animal at 1536–1920 | 4–8 (patch-14→16 replumb or new DETR head, gated weights, license read) | Labels/graders/evaluator/telemetry, gauge, alpha pattern | Working 2D + trained FPN/RPN/cube head, all measured baselines, months of recipe knowledge; D0 explicitly ranked confidence calibration AHEAD of any backbone swap |
| **D. Assembly (frozen 2D + geometry FM + tiny readouts)** | Yes with MoGe | **FAIL** — PCA/box-fit heading is axis-ambiguous; the trained alpha head is needed regardless, so assembly strictly *adds* mechanisms | FAIL — literature verdict 3× negative: PatchNet/DD3D (transform, not representation, is the win — our stack is already the evolved form), OVM3D-Det needed erosion+LLM repair, 2601.03617 (mask-sampling hurts) | MoGe affine is honest, but VGGT-online replicates the GT authoring pipeline → circular | Fine (MoGe MIT, 331M, V100-OK) | Zero; "flying points" worst exactly at small objects | 2–3 to a literature-predicted dead end | Its ONE idea survives as row-A's gated anchor probe | Trained lift quality, heading co-equality, non-circularity. Also **already adjudicated locally**: GroundCast rejected the depth-FM as scene input (1.24% intra-object signal vs ≥2% AbsRel at ~9 patches); FM survives only as gated distillation + audit |
| **E. 3D-MOOD (strongest surfaced alternative)** | YES | Unknown — orientation representation unverified (survey B open question) | FAIL — carries its own 2D detector (ours is solved+frozen) | Metric-bound; the depth-severing surgery is unproven there | FAIL as published — batch 128 over 8 GPUs/multi-node; mmdet offline stack | None | 3–5 | Labels/graders/evaluator only | Promptability (breaks the oracle-2D cross-arch protocol), plus everything row B loses |

Not vehicles: **LocateAnything3D** = monitor only (metric number-token gauge, NC weights, rotation explicitly ranked least learnable — the anti-co-equal design; its 49.89 AP3D conflates architecture with 1.74M training images). **MoGe-2, MapAnything, GeoCalib, SAM3/SAM3D, Pi3/DA3** = components (anchor probe, pitch recovery, label hygiene, future pseudo-label regeneration), not vehicles.

---

## 2. THE ANCHOR QUESTION — answered

**The single best mechanism across all four regimes is the one already specced for run 3: the per-image anchor head — `z = z̃·exp(δ_img)`, δ_img regressed from pooled P5 + frame-global geometry (log fx_tel, pitch), zero-init so it starts at the median(z)=1 convention on own detections.** The surveys converge on it independently: Regime B found NO published per-frame arbitrary-gauge scale mechanism anywhere (the slot would be novel in any vehicle); Regime C derived exactly this shape (global-context head + renormalization prior, structurally unable to repeat run-2's pollution); Regimes A/D propose the FM route only as the escalation.

Why it wins on local measurement, not vibes:
- Phase A: the model's per-frame anchor error is **sd 0.0456 vs a TRUE within-segment drift of only 0.0203** — the trivial convention (δ=0, i.e., per-frame median renorm) already roughly halves the anchor error before any learning; the learnable residual is the real 2.03% drift.
- The drift's dominant driver is a **known input**: fx_tel varies 1.4–2.6× *within* videos (zoom). The head reads log fx_tel directly — the cause of the drift is an exact telemetry scalar, which is why a 331M geometry FM is probably unnecessary for this scalar.
- It is a per-frame mechanism for a per-frame quantity (the run-2 law), enters only the z path, backbone frozen — it cannot reproduce the −6.6 2D-recall FPN pollution.
- Cost: ~2-file diff, ONE run judged against the existing ctrl, 2.6 h/15k on a V100 — off the contended A40 entirely.

**Escalation ladder if run 3 under-delivers, strictly ordered and gated:** (i) add the telemetry plane-depth prior scalar to the anchor head's input (computed, free); (ii) cached MoGe-2 frame-level ground statistics (near-field-weighted plane fit in its affine gauge) concatenated as one more global input — MIT, 331M, V100-runnable, all 59,598 frames cacheable in ~1–2 A40-hours (survey A), and consistent with GroundCast's pre-registration of the FM as a *gated* arm only. Each rung must beat the previous rung on measured anchor error to earn its dependency (survey D's own rule, and the standing "delete more than it adds").

**Failure modes:** (a) irreducible per-segment floor — part of a frame's anchor depends on *other frames'* depths; measure it from GT z-stats (one CPU script) BEFORE interpreting any run-3 shortfall; (b) circularity — GT is VGGT-authored, so the head can learn VGGT's per-frame biases; grade on the label-sane subset + human-audited frames, and keep the audit channel MoGe-family, not VGGT; (c) median-of-detections bias in dense herds — gate with grade_detection_sane; (d) aerial domain gap of any FM rung (AerialMetric documents it, and a fine-tune recipe exists) — validate on the 23 telemetry videos first.

---

## 3. RECOMMENDATION — with the two-way door

**Stay and finish run 3. Adopt MoGe-2 as one bounded probe. Nothing migrates now.**

**NOW (cheap, reversible, ~1 week, mostly CPU/V100):**
1. **Run 3 exactly as specced** (dims-only token + anchor head), ONE run vs the existing ctrl, judged on the label-sane grader (recall@.5, z raw, z anchor-free, dims, xy) + grade_orientation. Success shape: raw z moves from 2.67% materially toward the 1.09% floor (plan target NHD_z 5.88 → ~3–4) with 2D recall and the 0.159 dims gain held, orientation stable.
2. **CPU script: single-frame anchor identifiability floor** from GT z-stats — how much of the 2.03% true drift is visible in one frame under the per-segment gauge. This number is the yardstick that makes run 3 interpretable and the migration trigger honest.
3. **MoGe-2 afternoon probe** (V100): correlate its plane-fit frame scale against true anchor drift on the 23 telemetry videos; simultaneously stand it up as the non-circular audit channel GroundCast pre-registered. Adoption rule: it enters training ONLY if run 3 leaves a gap ≥ ~1.5× the measured floor AND the probe's residual correlation explains that gap.
4. **GeoCalib check** on the same 23 videos (bar: 0.73° p50; existing fallback consensus 1.76°) to unlock pitch for the 40 gimbal-less _V videos. Tiny, permissive licenses.
5. If run 3 is positive: **3-seed it on V100s**, then promote the **uncertainty head supervised by realised IoU** (D0: confidence discards ~half the model's AP — the single largest AP lever anywhere in this decision, and it is vehicle-internal).

**COMMIT only after these measurements:**
- Anchor closed + confidence fixed + AP3D/BEV beats 13.17/8.68 beyond seed noise → the incumbent is confirmed; proceed down the GroundCast ladder (axis×sign orientation arm, S4–S8) and stop re-evaluating vehicles.
- **Migration trigger (pre-registered, falsifiable):** if after run 3 + the confidence fix, z sits at the measured identifiability floor but AP3D/BEV still does not beat baseline beyond noise — i.e., the bottleneck is demonstrably the lift head/resolution, not the anchor — **activate DetAny3D** (pipeline already green). Week-one deliverable: WildBox oracle-2D fine-tune scored by the same evaluator + label-sane grader, with its native alpha head supervised by the 9,887 motion labels (replace the yaw=0.0 stub) and graded by grade_orientation. Falsifiable: it must beat the incumbent's detection scorecard AND match 96.2% alpha within CI by week two, or it dies and the finding is "the incumbent is the frontier for this regime." Decide DDP-debug vs 27 h/arm before starting.
- **DINOv3 is not a migration, it is the parked resolution arm:** before any backbone swap, run the cheap version of its one real argument — raise the incumbent's input res 560 → ~900–1120 and measure. Only a large, resolution-blocked gain plus saturated anchor+confidence re-opens the DINOv3 question (D0's own ordering). The DINOv2 register-variant check is near-free; do it whenever convenient.
- **Full assembly: rejected permanently** (three independent literature negatives + heading axis-ambiguity + GroundCast's local adjudication).

---

## 4. PERMANENT CAPITAL vs DISPOSABLE

**Permanent regardless of vehicle:**
- The label/grader stack: build_heading_labels (9,887 / 7,460 / 5,542), grade_orientation with per-species floors, grade_detection_sane, viz_heading, the 359-item human audit (95% of flagged off-plane boxes = junk).
- Telemetry cache (63/63, both DJI dialects), the K_tel convention, and the finding that json K is VGGT's per-segment fiction (deployment truth vs GT gauge).
- The evaluator: bev_ap_eval (validated to reproduce seed0 exactly), oracle-2D jsons, the 13.17/8.68 baseline protocol, the 23-video gimbal validation set (0.73° p50 bar).
- The measured laws: 1.09% anchor-free floor; anchor sd 4.56 vs true drift 2.03; depth = 85–99% of 3D error; D0 (confidence discards ~half the AP; trivial+oracle-ranking ties the published model); run-2's negative (per-RoI tokens pollute shared features — per-frame quantities need per-frame mechanisms); the dims-token mechanism (angular size + true fx → proportions, −30%).
- The alpha result and recipe (96.2%, 3-run stable) and the axis×sign factorization (mod-π axis on all 237,505 boxes + sparse sign bits).
- The gauge doctrine and GroundCast's pre-registered falsifiers.

**Disposable:**
- Cube R-CNN plumbing and trained weights (FPN/RPN/RoI, the confidence head as implemented — already scheduled for replacement), the SQUARE_PAD-560 choice, the run-2 token formulation.
- VGGT as anything but frozen history (labels are the capital; future regeneration could use Pi3/DA3 offline, optional).
- The DetAny3D clone + WildBox glue: a zero-maintenance option, not capital.

---

## 5. OPEN QUESTIONS FOR THE USER

1. **Is there a submission deadline, and how far out?** If < ~8 weeks, the migration trigger can never fire in time; the plan compresses to run-3 → confidence → orientation arm → multi-seed headline, and rows B/C leave the matrix entirely.
2. **Paper-only or releasable tool/checkpoint (any commercial angle)?** Decides license weight: MoGe MIT / DetAny3D Apache are clean; DINOv3's custom license ("Built with DINOv3", derivative clauses) and VGGT's NC original checkpoint need a real read only if artifacts ship.
3. **Weekly A40 budget and whether cluster admins will help debug the NCCL DDP hang.** Determines whether the DetAny3D insurance is real (27 h/arm × 3 seeds) and whether GroundCast's ~95–130 A40-h ladder fits the season; if the A40 stays scarce, everything V100-runnable gets priority automatically.
4. **Do you want the uncertainty/confidence head promoted immediately after run 3, ahead of any FM probing?** D0 says it is worth roughly half the model's AP — larger than the entire anchor gap in AP terms; both are incumbent-internal, so this changes sequencing and how fast 13.17/8.68 falls, not the vehicle.
5. **Provenance of the 5,542 human locks: seeded from VGGT boxes (sign-corrected only) or drawn independently?** If seeded, circularity control for any FM-derived signal leans entirely on the 23 telemetry videos, which raises the evidential bar for the MoGe rung and for any claim that run 3 "closed" the anchor.
6. **Does the 15.8° median elevation hide a low tail (< 8°)?** A grazing tail is where plane-fit and ground-statistics mechanisms die first; its size decides whether the anchor head needs the median-animal fallback alignment designed in from the start.