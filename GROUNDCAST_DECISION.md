Both verifications resolved a live discrepancy between the designs. Now I can write the decision.

---

# THE ARCHITECTURE DECISION — GroundCast (adopted spine, with named grafts)

## 1. THE PROBLEM, RESTATED CORRECTLY (5 sentences)

One monocular model must emit, for every animal in a single oblique aerial frame (median 15.8° camera elevation, ~186 m standoff, two exactly-known optical lenses whose digital zoom is a geometrically inert crop-and-resample that scales fx by a known factor and adds zero depth information), a 3D box graded against 12.4 AP3D / 8.68 BEV@0.50 / 7.063 NHD and a heading graded flip-sensitively per species against the train-transferred constant floor — two co-equal headlines forced by the theorem that NHD/IoU/BEV are exactly 180°-flip-invariant. Intrinsics are exactly known from telemetry, gravity is telemetered from the gimbal, and the label gauge is a per-segment scale pinned to median depth exactly 1.000 (sd(log)=0.0000 over 82 segments), so exactly one scalar per segment is unidentifiable and everything else — angles, ratios, ordering, within-frame depth structure, and the per-frame anchor up to a measured sd(log)=0.0203 drift — is a legitimate target. The camera is affine at object scale (1.24% depth extent, ~1.75 px), so the RoI crop carries no depth and RoIAlign deletes the two extrinsic cues (image row, apparent size) — the verified mechanism behind depth being 84.5% of 3D error — while the same limit makes allocentric α a near-sufficient statistic of crop appearance: the regime is worst-possible for appearance-based depth and near-best for appearance-based orientation. The measured ladder factorises the depth failure into within-frame structure (which plane geometry supplies at 0.67–1.7% vs the learned head's 2.04% — IF the raw-telemetry normal is good to <7°, which is unmeasured and is the gate) times a per-frame anchor the model gets wrong by 2.2× more than the anchor actually drifts (80% of that variance recoverable, and learnable because the gauge is pinned); orientation fails because the 1−cos loss has a vanishing gradient exactly at the 180° flip and 94.2% of instances lack the sign label — while the boxes carry the axis mod π on 100% of 237,505 annotations. Adjudicating the sweeps: the binding constraints are (i) the depth parameterisation (not backbone capacity, not virtual depth — moot in the ground gauge — not BEV lifting, degenerate at 2–13° HFOV), (ii) the sign bit's label sparsity plus pixel starvation (resolution binds HERE, not on depth), and (iii) VGGT label circularity, which caps measurement, not learning, and is handled by independent audit channels — with the ranking head a measured near-null (AUC 0.583 vs 0.597) until the oracle gap is decomposed.

## 2. THE SPINE: GroundCast

**GroundCast is the spine because it is the only design whose depth mechanism is factorised exactly along the measured error ladder — geometry supplies the within-frame structure, a learned per-image head supplies the anchor, a learned per-RoI residual supplies what geometry provably cannot (hidden feet, terrain, lying-down animals) — so the user's objections (1) and (2) degrade it gracefully instead of killing it. GroundFrame and GRAZE both stake everything on a hard closed-form read-off whose two known failure modes (contact bias, telemetry-pitch/terrain sensitivity at cot(15.8°)=3.53×) are precisely the objections raised; GroundCast makes the plane a differentiable prior mean rather than a constraint, which is why it survives a marginal telemetry gate that would kill the others.**

**Grafted in:**
- From **GroundFrame**: the contact-visibility logit with visibility-weighted contact loss (the explicit handling of the 21.3% crowding population); h-by-convention `h_conv = 1/median_i[d_z,i/(n̂·d_i)]` — demoted from "the anchor" to the anchor head's zero-parameter initialisation and regulariser; the analytic von Mises κ prior; the three-arm axis sub-ablation (parameterisation vs supervision); the S2b substitution-diagnosis experiment; the flank statistic correction |sin α|·cos ε; the dims chart (log H, log L/H, log W/L) with H load-bearing in the decode.
- From **GRAZE**: the antipodally-tied von Mises mixture NLL on labelled instances (max gradient exactly at the flip; subsumes a bare sign BCE and lets sparse labels also sharpen axis and κ); the explicit geometric Jacobian in the graph (∂λ/∂ψ = −λ cot ψ visible to the optimiser, not buried in a regression head); the VIRTUAL_DEPTH dissolution argument; deleting the alpha trunk as an ablation axis, not a default.
- **Adjudicated by code this session**: GRAZE's claim that WildBox training labels are bottom-centre is **WRONG** — commit 22a266a converts centroid→bottom-centre only for the *released papersub KITTI labels*; `tools/prepare_wildbox_dataset.py` builds `center_cam` from the VGGT box **centroid** (lines 65, 431, 448, 515). GroundFrame's contact target — projection of X_centre_gt − (H_gt/2)·n̂ — is the correct construction, the contact-to-centre offset is real in the training gauge, and this keeps the explanation of the post-hoc substitution failure coherent. Also verified: IMS_PER_BATCH is 4 (so ~0.9 labelled sign instances/iter, not 1.9), and the alpha-trunk quarantine comment exists verbatim at cube_head.py:63-67.

**Discarded:** GroundFrame's anchor-by-convention-only (inherits detection recall bias; the learned anchor subsumes it and the convention survives as its init); GRAZE's "no z anywhere" purity (dies on objection 1 — a wrong contact ray is not a small perturbation); GRAZE's FPN-level perspective-field channels (a second injection point for the same information the RoI token carries — parsimony; revisit only if the RPN shows small-object recall problems); GroundCast's own retained-6D-pose-for-tilt (over-parameterised for a measured ~1-DOF manifold; deleted — R is built from n̂ and α, the lying-down contingency goes to the label audit, not the architecture).

## 3. THE ARCHITECTURE

```
single image ──► frozen DINOv2 ViT-L (backbone.net, ~300M, FROZEN)
                    │
                    ▼
              trainable FPN ──► RPN ──► 2D box head        [all unchanged; ~15-20M trained]
                    │                      │
                    │            per-RoI RoIAlign features
                    │                      │
   TELEMETRY (computed, 0 params):         │
   K_tel = diag(fx,fx,1), fx=(f_eq/36)·W·dzoom  (convention gate S0)
   u_g = world-up from gimbal (pitch θ, roll φ)│
                    │                      │
   geometric token τ (10 scalars/RoI): ────┤   [ray d to box bottom-centre (3), u_g (3),
       COMPUTED, concatenated to           │    sin ψ ray depression, log fx, log(h_px/fx), log dzoom]
       flattened RoI features              │    = computed Perspective Field sampled at RoI
                    │                      ▼
                    │            GROUND CUBE HEAD (replaces CubeHead; shared fc trunk,
                    │            separate alpha trunk DELETED — sharing is an ablation axis)
                    │            outputs 11 scalars/RoI (was 15):
                    │              (du,dv) contact offset | visibility logit v
                    │              log H, log L/H, log W/L
                    │              axis director a=(cos2α,sin2α) | sign logit s | log κ (residual off analytic prior)
                    │              δ  (per-RoI log-depth residual)
                    │            + quality q (gated, S6)
                    ▼
   per-image ANCHOR head a_img: pooled P5 + telemetry token → MLP 256→64→1 (~17k params),
       initialised/regularised at log h_conv (the median(z)=1 convention imposed on own detections)
                    │
                    ▼
   DIFFERENTIABLE DECODE (closed form, no free depth):
       p = bottom-centre + (du·w, dv·h);  d_p = normalize(K_tel⁻¹[p,1])
       X_contact = exp(a_img + δ) · d_p / (−u_g·d_p)          ← anchor × plane-structure × residual
       X_centre  = X_contact + (H/2)·u_g                      ← the offset the post-hoc patch missed
       e1 = normalize(d_p−(d_p·u_g)u_g); e2 = u_g×e1
       α = ½·atan2(a₂,a₁) + π·[s<0];  ĥ = cosα·e1 + sinα·e2;  R = [ĥ | u_g×ĥ | u_g]
                    ▼
   per-animal 3D box (X_centre, (L,W,H), R) + heading α + calibrated von Mises posterior
   → standard Omni3D evaluator; 12.4/8.68/7.063 directly comparable
   → planner outputs analytic: flank = sinα·cosε, per-face visibility n_face·d<0 (0 params)
```

**DELETED:** bbox_3D_center_depth (free z), 6D pose head, Kendall uncert_sf and LOSS_W_POSE/LOSS_W_JOINT, `feature_generator_alpha` separate trunk. Net new trained parameters < 50k on top of the existing FPN/heads; the model gets smaller.

**Losses (unitary scalarisation; all gauge-invariant or gauge-pinned):**
- L_rpn, L_2d — unchanged.
- **L_contact** = smoothL1((du,dv) − (du,dv)*), RoI-normalised; GT contact = projection of X_centre_gt − (H_gt/2)·n̂ (verified centroid convention). **Dense on all 237,505 annotations, occluded feet included** — the target is where contact IS, not where it appears, so the hidden-feet bias is supervised away. Down-weighted by predicted visibility. w=1.0.
- **L_vis** = BCE(v, "GT contact pixel inside own SAM3 mask and not inside a neighbour's"). Label-time masks only. w=0.2.
- **L_z** = L1(log z_pred − log z_gt) at the centre; gradients reach a_img (frame term), δ (residual), and (du,dv) through the explicit Jacobian. Plus **L_δ** = 0.1·δ² so the residual fires only when appearance warrants — this is what keeps the plane the prior mean and stops δ re-becoming a free z head. w=1.0.
- **L_anchor** = 0.25·|a_img − log h_conv| — tethers the learned anchor to the convention solution.
- **L_dims** = smoothL1 on (log H, log L/H, log W/L), residuals to per-species priors REFIT on gauge-consistent stats. w=1.0.
- **L_axis** = von Mises NLL on the doubled angle with κ, α_gt from GT box yaw mod π — **dense, 100% of annotations**, gated by the audit mask m_p (long/mid > 1.3 AND axis within 20° of segment plane; drops the measured 13.4%/rhino 33.9% impossible poses). w=1.0.
- **L_orient** = −log[σ(s)·vM(α_gt; α, κ) + (1−σ(s))·vM(α_gt; α+π, κ)] — antipodally-tied mixture, **masked to the 9,887 labelled instances (5.80%)**, normalised by FIXED expected count (not the realised ~0.9/iter at IMS_PER_BATCH 4), with RepeatFactorTrainingSampler keyed on heading-label presence targeting 8–16 labelled RoIs/iter. Maximal gradient exactly at the confident flip. w=1.0.
- log κ is a residual off the analytic prior κ₀ ∝ (sin²ε + cos²α(1−sin²ε))⁻¹, ε from telemetry.
- **L_q** = BCE(q, Rel-AP3D scale-aligned 3D IoU, grid (0.3,3.0,28) already configured) on ALL proposals including negatives, conditioned on τ. w=0.5. **Built only if S2c says rank dominates the oracle gap.**
- **L_distill** (ablation arm only, behind the S2d FM gate) = MiDaS SSI loss on a dense FPN depth head vs frozen DAv2-relative, deleted at inference.

**Sign head input:** a 224-px native-resolution re-crop through the SAME frozen DINOv2 (second RoIAlign at source resolution, ~256 tokens/animal ≈ 1.3× current token cost) — the sign is semantic asymmetry, the pixels exist (dzoom put up to 6.6× on target; the 518-resize discards them), and the 92.5% template evidence lives at large-crop resolution. FPN-features-only is the ablation cell. Mirror equivariance: inference-time TTA (predict on crop + mirror, map α→−α, average circular distributions; flip-disagreement = free epistemic uncertainty); architectural tying is a cheap later cell.

**How the 5.8% mask is handled — the complete answer:** structurally, the mask now gates one bit and one concentration (the axis is at 100% via L_axis — a ~24× supervision expansion, licensed by gate S1b); statistically, sampler + fixed-count normalisation fix the ~0.9-instance/iter variance; honestly, the ~208-independent-track effective sample size forbids any high-capacity angular head — the sign head is one logit.

## 4. WHY IT IS ELEGANT

One idea, applied twice: **factorise each output along what is identifiable from where, compute the computable factor exactly, and let the network learn only the factor that is provably its own.** Depth = anchor × plane-structure × residual, where telemetry-plus-known-K supplies the structure in closed form (h cancels in every ratio; rel_alt never enters), the pinned gauge makes the anchor a learnable one-scalar-per-image quantity initialised at its own convention, and the residual absorbs exactly the population — hidden feet, mounds, lying animals — that geometry provably cannot see. Orientation = axis × sign, where the flip-invariance theorem says the boxes carry the axis mod π exactly and carry no sign at all, so the dense labels supervise the continuous factor and the sparse labels are spent entirely on the one bit that is an appearance classifier forever. Nothing is patched: the contact offset, the (H/2)·n̂ centre lift, the analytic κ, the planner's flank output, and the label auditor are all read-offs of the same two geometric objects (the plane and the quotient), and the model ends with fewer heads, fewer output scalars, and fewer loss terms than the baseline it replaces.

## 5. ORDERED BUILD PLAN

**S0 — CONVENTIONS + TELEMETRY GATE (no GPU, 1–2 days, runnable this week).**
(a) Settle fx convention (35mm-equiv vs physical, width vs diagonal): SRT/EXIF dump of one wide + one tele frame; the size-consistency regression (predict pixel size = fx·L_species/(rel_alt/sin|pitch|) over 237,505 anns; correct convention → slope 1, no dzoom/rel_alt residual trend). Files: new `tools/aeroview/verify_intrinsics.py`. (b) Join 404 SRTs frame-accurately; measure per-frame angle between telemetry normal and GT-fitted normal; fit horizon on wide frames to settle pitch-absolute-vs-body and roll logging. **PASS p50<3° (→1.29% relerr), MARGINAL 3–7°, FAIL >7°** (learned head is 2.04%). (c) Reconcile the 5.878-vs-5.08 NHD_z discrepancy; confirm segment-disjoint split. FAIL consequence: plane cast demoted to token+anchor+residual with free structure — the design survives; the central-mechanism claim dies and is reported as such.

**S1 — LABEL + AXIS AUDITS (no GPU, 1–2 days).** (a) Re-derive alpha labels under K_tel; report shift on 5,542 human locks; re-run zero-training pipeline (92.5/97.1) with corrected K — the label fix ships regardless. (b) Score GT box yaw mod π vs human axis on locks: **PASS >90% within 15°** — licenses the 24× expansion; decompose the 10.9% same-track axis flips into 180° flips (harmless) vs 90° swaps (gate L_axis if swaps >5%). (c) Out-of-plane audit per species; visually inspect 50 flagged (PCA failure → mask; lying-down → contingency output); emit m_p. (d) Contact audit: ~300 hand-marked crops; GT-contact bias vs mask bottom on hidden-feet cases; verify δ_gt = log(z_gt/z_plane) correlates with occlusion cues (residual learnability) — **this adjudicates objection (1) with numbers**. (e) Train/val heading overlap at track and video level. Files: `tools/aeroview/audit_labels.py`, edits to `tools/preflight_orientation`.

**S2 — ZERO-TRAINING PRECURSORS + AUDIT CHANNELS (no GPU + one GPU afternoon).** (a) Fit z = h/(n̂·d) with one free scalar per frame using TELEMETRY normals; compare NHD_z vs trained head — the decisive pre-experiment. (b) Redo the post-hoc substitution WITH the (H/2)·n̂ offset and GT dims/rotation held fixed — separates offset failure from NHD Hungarian coupling. (c) Decompose the 24.31-vs-43.03 oracle gap into rank/recall/duplication — gates S6. (d) **FM GATE (objection 4)**: frozen DAv2-relative on tele-lens val frames; within-frame Spearman vs GT and vs the plane ordering, stratified by dzoom; PASS for distillation only if it adds ordering information where the plane is weak (steep pitch, occluded contacts). (e) Sign probe: small classifier on frozen DINOv3 features of the 9,887 labelled crops vs the 92.5% template, stratified by crop size — gates the per-frame sign head's ceiling. (f) Build the three VGGT-independent audit channels (telemetry range at contact, mask occlusion order, SfM on static ground) on 2–3 held-out segments; standing rule: no depth claim on GT-agreement alone.

**S3 — ARM A: DEPTH (3 paired seeds, ~10.5 h + eval).** Token + contact head + anchor×plane×residual decode + K_tel labels; orientation heads untouched. Files: `cubercnn/modeling/roi_heads/cube_head.py`, `roi_heads.py`, `dataset_mapper.py`, `prepare_wildbox_dataset.py` (K_tel), config. Ablation cells: free-z same-everything (isolates the parameterisation); δ≡0 (isolates the residual and directly measures hidden-feet absorption). Primary: NHD_z, NHD_xy, BEV@0.25/0.50, plus the three independent channels. Independently ablatable: yes.

**S4 — ARM B: ORIENTATION (3 paired seeds, ~10.5 h).** Doubled-angle axis dense (m_p-gated) + antipodal mixture + native-res sign crop + sampler + fixed-count; delete 6D pose and the alpha trunk. Sub-ablation: (A) current cos/sin head, (B) doubled-angle sparse-only, (C) doubled-angle dense — pre-registered: axis C≫B>A while sign ≈ equal across all three (the flip theorem says axis supervision cannot touch the sign; if sign moves too, that theorem's application is wrong and it is a finding). Independently ablatable: yes.

**S5 — CALIBRATION + EQUIVARIANCE (inference-only first, then 1 arm).** Mirror TTA (free); analytic κ prior with residual head; retire |sinα|≥0.35 for the model's posterior; risk-coverage curves (accuracy at 100/80/60/40% coverage) + PIT circular calibration per |sinα| decile.

**S6 — ARM C: RANKER (3 seeds, ONLY if S2c says rank dominates).** q head on scale-aligned IoU, all proposals, conditioned on τ; score = q.

**S7 — CONDITIONAL (only on diagnosed shortage).** If S4's sign is starved: telemetry-propagated α along straight tracks (self-validating: propagate end-to-end, report disagreement vs segment length), then entropy-filtered pseudo-labelling with the per-species collapse guard. If S2d passed: DAv2 SSI distillation arm; kill if neither task moves beyond 2.83.

**S8 — HEADLINE + TRANSFER.** 5 seeds on the winning config (pipeline exists, commit 7e2d80a). Presentation order per the metric decision: unchanged Primary-1 table first, flip-invariance lemma with its two in-repo proofs, Primary-2 with per-species floors and track-bootstrapped CIs, diagnostic panel (disentangled NHD per species, 26.3% NHD coverage, identifiability ladder, ranking gap, label-reliability row), 8-bin external comparison vs Sun et al.'s 75%. KABR: 20-minute feasibility check FIRST (full frames? crop offsets? focal/dzoom fields in telemetry?) — decides whether KABR grades the mechanism or only the orientation head; state which up front.

Total GPU: ~95–130 A40-hours across arms before the 5-seed headline.

## 6. TWO-TASK SCORECARD PER STEP

| Step | Detection (AP3D/BEV/NHD) | Orientation (sign/flank vs floor) | Could hurt the other? How detected |
|---|---|---|---|
| S0–S2 | none (measurement) | none (label fix may shift targets — reported before grading) | n/a — but S1a CHANGES the orientation target; publish the shift first |
| S3 depth | NHD_z 5.88→**3.0–4.0** if gate ≤3° (only →4.5–5.0 if marginal); BEV@0.50 8.68→**10.5–14**; AP3D@0.25 +2–4. Honest uncertainty: the 0.67% used an oracle normal; the gate bounds it | ~0 direct; small positive via shared FPN possible, not claimed | **Yes**: contact-driven xy could regress NHD_xy (2.199, band −0.3/+0.5) on the 21.3% crowded population — detected via disentangled NHD_xy + 2D AP, paired seeds; δ≡0 cell attributes it |
| S4 orientation | pre-registered: NHD_xy moves (≤−0.3), NHD_z does NOT; if NHD_z moves it is seed noise until 3 seeds say otherwise. Deleting 6D pose: pose-NHD may shift on the 13.4% tilted-label population (label defect, reported per species) | axis: strong, near-immediate (24× supervision; zero-training already 97.1% flank). Sign: **the genuine unknown** — target macro margin >+10 pts over transferred floor on ≥4/6 species, CIs excluding zero; may not beat the 92.5% cross-instance template | **Yes**: RepeatFactor sampler = video-level domain shift on detection — detected by the sampler on/off cell's detection scorecard; disqualification rule: detection regression beyond 2.83 kills the orientation claim |
| S5 | untouched (inference-time) | calibration/coverage improves; mirror TTA kills any shadow cue — compare strong- vs weak-shadow frames | no |
| S6 ranker | BEV@0.25 **+0–8** — honestly wide; measured near-null is a live outcome | none | no; drop rather than tune if null |
| S7 | distillation: both-task move required or killed | pseudo-labels: sign +0–5; floor-amplification risk — guarded by per-species pseudo-label distribution vs the transferred constant | yes (distillation wall-clock >15% = kill) |

## 7. FALSIFIERS AND CHEAPEST EXPERIMENTS

1. **Telemetry join p50 > 7°** → the plane cast loses to the head it replaces; the central mechanism is dead (design survives in reduced form, claim does not). Cheapest: S0b, one day, no GPU.
2. **S2a: the one-scalar-per-frame telemetry plane fails to beat the trained head's within-frame structure** → the "geometry supplies structure" premise is wrong before any training. One afternoon.
3. **S2b: substitution still degrades NHD with the offset and GT dims/rotation handled** → NHD's coupling is deeper than the offset story; S3's predicted conversion of depth into metric wins will not happen; re-scope. One afternoon.
4. **S1b: box-yaw-mod-π vs human axis <90% within 15°** → the 24× expansion dies; L_axis reverts to sparse and the factorisation keeps only its gradient-profile argument.
5. **S2e: frozen-feature sign probe cannot approach 92.5%** → per-frame sign is capped; the honest fallback is a frozen template head at inference — less elegant, stated now.
6. **S3 ablation: free-z matches plane-cast** → the plane is not the mechanism, the token was; report it that way.

## 8. DELIBERATELY NOT DOING

Metric depth or rel_alt as model input (h cancels; the one unidentifiable scalar stays unestimated). Re-enabling VIRTUAL_DEPTH (in the ground gauge there is no focal-scaled regression target left to normalise — the two sweeps that argued for it are answered by dissolution, not rebuttal). BEV lifting (frustum degenerates to a line at 2–13° HFOV). Tiling/SAHI/query detectors (the bottleneck is the 1920→518 resize; the native-res sign crop is 3× cheaper per token-on-target). Any geometry FM at inference, as input channel, or as pseudo-LiDAR (1.24% intra-object signal vs ≥2% AbsRel, ~9 patches — arithmetic; FM survives only as the gated training-time distillation arm and the non-circular cross-check, per objection 4's honest verdict: the plane supplies structure the FM cannot beat where the plane works, and where the plane fails is exactly the FM's documented telephoto weakness — so the FM is a gated regulariser, never the scene model). 6D rotation / matrix Fisher (1-DOF target). Gradient surgery / GradNorm / PCGrad / Kendall homoscedastic weighting (two tasks, one 1-D; per-sample κ subsumes it). Any temporal dependency at inference. Replacing or reweighting the detection headline (the unidentifiability premise is measured false; depth ordering is near-saturated and stays a secondary diagnostic). Chasing the oracle gap as a headline (measured near-null; decompose first). Shadows, SMAL rendering, Orient Anything V2, animal-keypoint FMs, MonoCon head/tail keypoint heatmaps — each logged as a gated fallback for a diagnosed sign shortage (S7), none in the core: each is a plausible independent lever and together they are the pile of fixes the user has twice rejected. GRAZE's FPN-level perspective-field channels (second injection point of the token's information). GroundCast's retained 6D pose (deleted). The alpha-trunk quarantine as a default (verbatim comment at cube_head.py:63-67 keeps "the promise to leave the existing losses untouched" — the framing the goal statement bans; it becomes the S4 ablation axis).

## 9. OPEN QUESTIONS FOR THE USER

1. **Which airframe(s) shot the 404 SRTs, and can you produce one checkerboard (or known-baseline) image per lens at dzoom=1?** Yes → fx convention settled decisively; no → the S0a consistency regression is the fallback and a residual ~4.7% width-vs-diagonal ambiguity may persist into every ray.
2. **From your field knowledge: do these videos contain a meaningful population of lying-down / mound-standing animals, or are the 13.4% out-of-plane boxes almost certainly PCA failures?** Label defect → loss mask (as designed); real behaviour → a 1-DOF tilt output must be added back and deleting the 6D pose head partially reverses.
3. **If the per-frame sign head cannot match the 92.5% frozen-template accuracy, is keeping a frozen DINOv3 template head at inference acceptable, or does "one elegant model" forbid it?** Forbid → S7 label expansion becomes mandatory rather than conditional, and the end-on band ships as a calibrated "unknown".
4. **Is the KABR transfer meant to grade the full mechanism (needs raw frames + telemetry with focal/dzoom fields, unverified) or only the orientation head (mini-scenes suffice)?** Determines S8 scope and what the paper promises; the 20-minute check answers feasibility but not intent.
5. **Is ~95–130 A40-hours of arms before the 5-seed headline acceptable (~2–3 weeks of queue), or should cells be cut?** Cut order if constrained: S6 ranker first (measured near-null), then S5's architectural-tying cell (TTA stays, it is free), never S3/S4's isolation cells — without them no mechanism claim survives review.
6. **Can you fund ~1–2 annotator-days for the S1 audits (300-crop contact visibility, 50 out-of-plane inspections, human sign-agreement on end-on crops)?** No → the hidden-feet bias and the end-on learnability question stay model-inferred rather than measured, and objection (1)'s adjudication is weaker.