"""The head-cue interface.

Geometry has reduced the heading problem to exactly one question:

    which of the 4 horizontal faces of this box is the animal's FRONT?

A cue answers that (with a confidence) from image evidence. Cues are injected into the
pipeline rather than imported by it, so the geometry half stays testable on CPU with no
model weights -- and so cues can be swapped, ensembled, or ablated independently.

A cue may ABSTAIN (return an empty dict) whenever it has no evidence -- e.g. GroundingDINO
finds no head on a distant, occluded animal. Abstention is a first-class outcome: the
track-level resolver would rather have silence than a coin flip, because a confidently
wrong front face flips the left/right flank label and poisons the re-ID gallery.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from tools.heading.conventions import face_centers_world
from tools.heading.io import Camera

__all__ = ["CueContext", "HeadCue", "projected_face_centers"]


@dataclass(frozen=True)
class CueContext:
    """Everything a cue may look at for ONE (track, frame) box instance."""
    image: np.ndarray            # full RGB frame, HxWx3 uint8
    bbox_2d: np.ndarray          # animal box in image coords, (x1, y1, x2, y2)
    center: np.ndarray           # box centre  (world)
    dims: np.ndarray             # box extents (world/local)
    R: np.ndarray                # box rotation; columns are the box axes (world)
    up: np.ndarray               # signed ground-up (world)
    camera: Camera
    candidates: list[int]        # the 4 horizontal faces -- the cue must pick one
    class_name: str = "animal"

    @property
    def crop_box(self) -> tuple[int, int, int, int]:
        """bbox_2d clamped to the image, as ints."""
        h, w = self.image.shape[:2]
        x1, y1, x2, y2 = self.bbox_2d
        return (
            int(max(0, np.floor(x1))), int(max(0, np.floor(y1))),
            int(min(w, np.ceil(x2))), int(min(h, np.ceil(y2))),
        )


@runtime_checkable
class HeadCue(Protocol):
    """Scores each candidate face by how likely it is to be the animal's front."""

    name: str

    def score(self, ctx: CueContext) -> dict[int, float]:
        """face id -> score (higher = more front-like). Empty dict = abstain.

        Scores need not be normalised; the resolver only compares them within a frame
        and uses the best-vs-second margin as the confidence.
        """
        ...


def projected_face_centers(ctx: CueContext) -> dict[int, np.ndarray]:
    """Candidate faces -> their centre projected into the image, in pixels.

    Faces whose centre falls behind the camera are dropped: they cannot be the front,
    and projecting them would produce a mirrored point that silently scores well.
    """
    centers_w = face_centers_world(ctx.center, ctx.dims, ctx.R)
    out: dict[int, np.ndarray] = {}
    for f in ctx.candidates:
        p_cam = ctx.camera.world_to_cam(centers_w[f])[0]
        if p_cam[2] <= 1e-6:
            continue
        out[f] = ctx.camera.project(p_cam)[0]
    return out
