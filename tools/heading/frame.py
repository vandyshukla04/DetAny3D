"""Derive an animal's canonical frame (forward / up / left) from its 3D box.

The box rotation gives three *axes*, but their signs are meaningless (PCA/SVD sign
ambiguity -- 12.4% of consecutive frames flip one). So we re-derive the frame:

    up          <- the box axis aligned with the ground normal, signed toward the drone
    front face  <- one of the 4 faces whose normal is horizontal   [INJECTED head cue]
    forward     <- that face's outward normal, flattened into the ground plane
    left        <- up x forward                                    (never guessed)

WHY THE CUE PICKS A FACE, NOT A SIGN
------------------------------------
An earlier design first chose the body axis ("the longer of the two horizontal axes")
and left the cue to pick only its sign. Measured on 11,084 real box instances that
heuristic was wrong 5.6% of the time -- and for 3 whole tracks it was wrong in *every*
frame, because their PCA box is genuinely wider than it is long. A bad box fit makes
"longest axis" unrecoverable, so the heuristic is deleted: the cue picks the front
FACE directly from the 4 horizontal candidates, fixing axis and sign in one step.

MEASURED ON THE HUMAN LOCKS (11,084 instances, 8 segments, 66 tracks)
---------------------------------------------------------------------
    the locked front is among our 4 horizontal candidates : 100.00%
    up / TOP derived from geometry alone                  : 100.00%
    left = up x forward, given the front face             : 100.00%

=> the geometry is exact, and the ONLY unknown is the front face. All residual error
   belongs to the cue, which is where it can actually be measured and improved.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tools.heading.conventions import (
    OPPOSITE_FACE,
    face_index_for_direction,
    face_normals_world,
    left_from,
)

__all__ = [
    "CanonicalFrame",
    "world_up_from_boxes",
    "sign_up_toward_camera",
    "horizontal_faces",
    "frame_from_front_face",
    "face_map",
]


@dataclass(frozen=True)
class CanonicalFrame:
    """Orthonormal animal frame, in the same coordinates as the box rotation."""
    forward: np.ndarray     # unit: nose direction (in the ground plane)
    up: np.ndarray          # unit: dorsal direction
    left: np.ndarray        # unit: the animal's own left
    front_face: int         # face id chosen as the front

    @property
    def right(self) -> np.ndarray:
        return -self.left


def world_up_from_boxes(
    rotations: np.ndarray, *, iters: int = 5, max_candidates: int = 256, seed: int = 0
) -> np.ndarray:
    """Consensus ground normal (UNSIGNED) from a set of box rotations.

    Every animal stands on the same ground plane, so exactly one axis of each box is
    (+/-) parallel to the ground normal. We look for the direction `n` maximising
    ``mean_i max_c |R_i[:, c] . n|`` -- the direction that is an axis of as many boxes
    as possible -- then refine by sign-aligned averaging.

    A plain second-moment (sum of outer products) does NOT work here: each box's three
    axes are orthonormal, so they sum to the identity and carry no directional
    information. Measured: picks the correct vertical axis on 11,084/11,084 instances.
    """
    R = np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3)
    if not len(R):
        raise ValueError("no rotations given")
    axes = np.transpose(R, (0, 2, 1))                       # axes[i, c] = R_i[:, c]
    flat = axes.reshape(-1, 3)

    rng = np.random.default_rng(seed)
    cand = flat if len(flat) <= max_candidates else flat[
        rng.choice(len(flat), size=max_candidates, replace=False)
    ]
    align = np.abs(np.einsum("icd,kd->ick", axes, cand))    # (N, 3, K)
    n = cand[int(np.argmax(align.max(axis=1).mean(axis=0)))]
    n = n / np.linalg.norm(n)

    for _ in range(iters):                                  # sign-aligned refinement
        d = np.einsum("icd,d->ic", axes, n)
        pick = np.argmax(np.abs(d), axis=1)
        chosen = axes[np.arange(len(axes)), pick]
        chosen = chosen * np.sign(d[np.arange(len(d)), pick])[:, None]
        m = chosen.mean(axis=0)
        norm = float(np.linalg.norm(m))
        if norm < 1e-9:
            break
        n = m / norm
    return n


def sign_up_toward_camera(
    up_unsigned: np.ndarray, box_center: np.ndarray, camera_center: np.ndarray
) -> np.ndarray:
    """Resolve the +/- of the ground normal.

    The drone is always ABOVE the animals, so 'up' is whichever sign points from the
    animal toward the camera.
    """
    v = np.asarray(camera_center, dtype=np.float64) - np.asarray(box_center, dtype=np.float64)
    s = float(np.dot(v, np.asarray(up_unsigned, dtype=np.float64)))
    if abs(s) < 1e-12:
        raise ValueError("camera lies in the animal's horizontal plane; cannot sign up")
    return np.asarray(up_unsigned, dtype=np.float64) * np.sign(s)


def horizontal_faces(R: np.ndarray, up: np.ndarray) -> list[int]:
    """The 4 faces whose outward normal is (near) horizontal -- the front candidates.

    Top and bottom are excluded: an animal's head is never on the dorsal or ventral
    face. Verified: the human-locked front is among these 4 on 11,084/11,084 instances.
    Returned most-horizontal first.
    """
    up = np.asarray(up, dtype=np.float64)
    normals = face_normals_world(R)
    return sorted(normals, key=lambda f: abs(float(np.dot(normals[f], up))))[:4]


def frame_from_front_face(R: np.ndarray, up: np.ndarray, front_face: int) -> CanonicalFrame:
    """Build the animal frame from the chosen front face. `up` must already be signed."""
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)

    forward = face_normals_world(R)[int(front_face)]
    forward = forward - np.dot(forward, up) * up            # flatten into the ground plane
    n = float(np.linalg.norm(forward))
    if n < 1e-9:
        raise ValueError(f"face {front_face} is vertical; it cannot be the front")
    forward = forward / n

    return CanonicalFrame(
        forward=forward,
        up=up,
        left=left_from(up, forward),
        front_face=int(front_face),
    )


def face_map(R: np.ndarray, frame: CanonicalFrame) -> dict[str, int]:
    """Semantic name -> face id, directly comparable with `track_face_locks.json`."""
    top = face_index_for_direction(R, frame.up)
    left = face_index_for_direction(R, frame.left)
    return {
        "front": frame.front_face, "back": OPPOSITE_FACE[frame.front_face],
        "top": top, "bottom": OPPOSITE_FACE[top],
        "left": left, "right": OPPOSITE_FACE[left],
    }
