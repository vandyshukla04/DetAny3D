"""Animal heading + box-face semantics for WildBox -> side-consistent re-ID.

Gives each detected animal a canonical frame (forward / up / left), hence which of the
box's faces the camera sees -- the viewpoint tag re-ID needs, because a zebra's left
stripes are a different pattern from its right, so a left-flank query must not be matched
against a right-flank gallery.

    conventions.py       corner/face geometry -- the single source of truth
    frame.py             box axes -> up, front face -> forward / up / left
    io.py                loaders (tracking_summary, cameras, human face annotations)
    dataset.py           human face labels -> image-space heading angle (the target)
    build_manifest.py    [CLI] scan all annotations -> training manifest
    extract_features.py  [CLI, GPU] frozen DINOv3 features on the labelled crops
    train_head.py        [CLI] small head: DINOv3 features -> heading -> FLANK

THE PROBLEM, AND WHY IT COLLAPSES TO ONE BIT
--------------------------------------------
The 3D box gives three axes, but their SIGNS are meaningless: they come from a PCA/SVD
fit whose singular-vector signs are arbitrary, and 12.4% of consecutive frames flip one.
Nothing downstream can see this -- BEV AP, NHD and 3D IoU are all invariant to a cuboid's
180-degree flips -- so the error is invisible to every existing metric.

Measured against the human annotations (11,084 instances, 8 segments, 66 tracks):

    the true front is among our 4 horizontal candidates : 100%
    up / TOP, from geometry alone                       : 100%
    left = up x forward, given the front face           : 100%

So the geometry is exact and the ENTIRE problem is one bit: which of the 4 horizontal
faces is the head end. Everything else is derived, never guessed.

WHAT DOES NOT WORK (measured -- do not retry these)
---------------------------------------------------
* **GroundingDINO "head" prompting.** On aerial-oblique drone footage it grounds the noun
  onto the *whole animal*: the 100%-of-crop box scored highest for every prompt tried
  ("head.", "animal head.", "zebra head.", ...). It abstained on 3/6 and was wrong on 3/3.
  This is domain shift, not resolution -- it failed at 149px as well as at 69px. Prompt
  and threshold tuning will not fix it.
* **Mask-shape taper** ("the head end is narrower, the hindquarters bulkier"). Scored 97%
  on ONE segment, but it is a function of viewing pitch/yaw, not anatomy, so it does not
  generalise. A cue that only works for one camera geometry is not a cue.
* **Motion direction.** `velocities` in tracking_summary are never populated, and
  finite-differencing `centers` gives nothing usable: the animals are grazing, travelling
  ~10-25% of a body length across a whole 200-frame segment, with displacement barely
  correlated with the body axis.

WHAT DOES WORK: the labels already exist
----------------------------------------
The VGGT annotator produced 13k+ human face labels (`track_face_locks.json`,
`manual_labels.json`) which were never wired into anything. We convert them to an
IMAGE-SPACE HEADING ANGLE -- not a face id, because face ids rename themselves whenever
the PCA signs flip -- and train a small head on FROZEN DINOv3 features. Verified: the
human face id flips with the sign chaos while our derived angle stays smooth (1.1%
jumps), i.e. the conversion provably undoes it.
"""
