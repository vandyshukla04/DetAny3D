# Locomotion-Anchored 3D Heading for Wildlife, with Zero Orientation Labels

## The problem

Monocular 3D detection is starting to work on wild animals: given a single aerial frame, models such as
DetAny3D place a metric 3D box around an elephant or a zebra. But a box is not a *pose*. It tells you where
the animal is and roughly how large it is — **it does not tell you which way the animal is facing.**

For wildlife re-identification, that omission is fatal. Re-ID is **side-dependent**: a zebra's left flank
carries a completely different stripe pattern from its right. Matching a left-flank query against a
right-flank gallery entry is not a hard match — it is a *guaranteed miss*. A re-ID system that cannot tell
which side of the animal it is looking at is silently comparing two different pictures and calling them a
failure. What it needs is a **viewpoint tag**: for each detection, which flank does the camera actually see?

That tag reduces to one geometric quantity: the animal's **3D heading in the world**.

## Why the heading cannot simply be read off the box

The obvious move is to read the orientation from the 3D box. It does not work, for two independent reasons,
both of which we verified in code rather than assumed:

1. **The detector's rotation head is untrained.** In DetAny3D, the 6-D rotation head receives *zero gradient*
   — its only loss (`chamfer_loss`) is absent from the training config — and the heading-aware `alpha` head is
   trained against a **hardcoded `0.0`** yaw placeholder. The box orientation it emits is therefore noise. It
   was never penalised for being wrong, because no metric can see it: 3D IoU, BEV AP and NHD are all
   invariant to a 180° flip. The published accuracies are not corrupted — but they are also not evidence that
   the orientation means anything.

2. **The ground-truth orientation has no heading either.** The boxes are fitted by PCA/SVD, whose axis *signs*
   are arbitrary. Empirically, 12.4% of consecutive frames of the *same animal* flip an axis. The box gives a
   body **axis**, but never a **direction**.

So the heading has to be *re-derived*. And there are no heading annotations to learn it from — labelling
orientation on tens of thousands of aerial animal crops is exactly the cost we are trying to avoid.

## The idea: three signals, each covering the others' blind spot

```
   LOCOMOTION   ──►  NAMES the head      free, exact, signed  │ but only while the animal WALKS
   DINOv3       ──►  TRANSPORTS the name  head↔head, rump↔rump │ but cannot name it, and is mirror-blind
   3D BOX + CAM ──►  PLACES it in the world  axis, ground, L/R │ but the sign is arbitrary
```

**Locomotion names it.** An animal that is walking is, to an excellent approximation, walking *forwards*. Its
world-space velocity — recovered for free from the tracker's own 3D trajectories and the camera poses — **is**
its heading. Unlike the box, this direction is **signed**. Across the dataset this yields **25,554 exact 3D
heading labels with zero human annotation**. Crucially, this is *not* an assumption we take on faith: the
motion direction and the box's body axis are two entirely independent measurements, and they agree at a
median **cos = 0.98**. Each corroborates the other, and their agreement doubles as a purity filter.

**But locomotion is not the method — it is the teacher.** Only 16% of frames show a walking animal; **84% are
standing or grazing**, and those are the ones we must actually serve. A method that requires motion at
inference is just a tracker.

**DINOv3 transports it.** Self-supervised dense features match *semantically corresponding parts* across
instances: a head patch on one zebra matches a head patch on another. This is DINOv3's headline property, and
the one it is engineered for. So locomotion names the head **once**, on the walking animals, and dense
correspondence carries that name to every other frame — including the standing ones. Nothing has to
*generalise*; it only has to **match**, and a grazing animal's head patches still look like head patches.

Notably, dense features are **mirror-blind**: a left flank and a right flank are near-mirror images, and the
descriptors cannot separate them. That is not a limitation here, because we never ask appearance for
left/right. We ask it for exactly one bit — **front vs back**, which is *not* a mirror pair — and take
left/right from geometry (`left = up × forward`).

**Geometry places it in the world.** The 3D box supplies the body axis and the ground plane; the camera lifts
the appearance decision out of the image and into world coordinates, where it can finally be compared across
frames. This is also what makes **temporal accumulation** valid: an animal's heading is coherent in the
*world*, but every frame observes it from a different viewpoint, pose and light — so the per-frame errors are
largely *independent* and cancel when integrated along a track. A mediocre per-frame cue becomes a confident
per-track answer.

## WildBox as a proxy for the detector's output

We develop and evaluate on **WildBox** — 60 aerial videos, 305 segments, four species (elephant, rhino, zebra,
giraffe) — and we use its 3D boxes and camera poses as a **stand-in for what a monocular 3D detector emits at
inference.**

This substitution is deliberate and specific. A detector must supply three things for our pipeline:

| requirement | DetAny3D | WildBox (proxy) |
|---|---|---|
| camera intrinsics | ✅ `pred_K` — genuinely predicted; the standalone demo already runs on it | ✅ `cameras.json` |
| ground plane / gravity | ✅ derivable from its dense metric depth map (computed on every forward, currently discarded) | ✅ consensus ground normal |
| 3D position | ✅ `center_cam` | ✅ per-frame world centres |
| **body axis** | ❌ **untrained rotation head** | ✅ from the box |

Everything except the **body axis** is already available from the detector. WildBox therefore stands in for
precisely the one component DetAny3D cannot yet provide — which lets us isolate and answer the heading
question without first fixing the rotation head, and without conflating "does the heading idea work?" with
"is this particular detector's rotation any good?".

The substitution is also the point of the final step. DetAny3D **already contains the right receptacle**: its
`alpha` head predicts the allocentric (observation) angle in 12 bins over 2π, with a live loss — it is simply
being fed a constant. Our pipeline produces exactly that quantity, at scale, with no human labels. Feeding it
back closes the loop: the detector learns to emit heading **natively**, end-to-end, with no DINOv3 and no
tracker at test time. WildBox is the scaffold we build on; DetAny3D is what we hand the result back to.

## What this buys

A per-frame **3D world heading** for every tracked animal — walking or standing — and from it the visible
flank, attached to every re-ID gallery entry. Trained on **zero orientation annotations**, on four species,
across sixty videos.
