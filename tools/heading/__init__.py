"""Training-free animal heading + box-face semantics for WildBox.

Derives, per detection, a canonical animal frame (forward / up / left) and labels
which faces of the 3D box the camera actually sees -- the viewpoint tag needed for
side-consistent re-identification.

Layering (each depends only on the ones above it):

    conventions.py   corner/face geometry -- the single source of truth
    frame.py         box axes -> up, body axis, forward  (head-sign is INJECTED)
    faces.py         face labelling + camera visibility
    cues/            head-sign cues: GroundingDINO, DINOv3 prototype
    io.py            loaders (tracking_summary, cameras, face locks)
    validate.py      scoring against the human face locks
    stats.py         dataset viewpoint-coverage aggregation

The head-sign cue is injected rather than imported, so the entire geometry half is
testable on CPU with no model weights -- see tools/heading/tests/.
"""
