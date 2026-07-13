"""Single source of truth for the 3D-box corner / face conventions.

Every other module in `tools.heading` imports its geometry constants from here.
Nothing else in the package is allowed to hardcode a corner index or a face id.

WHY THIS FILE EXISTS
--------------------
Three different corner orderings coexist across the repos, and the upstream
comments for the face table are WRONG. Getting this wrong silently inverts every
result, so the conventions are *derived from the corner coordinates* and pinned
by tests (see tests/test_conventions.py), never copied from a comment.

THE AUTHORITY
-------------
`BBox3D.get_corners()` (vggt/annotator_tool.py:82) — byte-identical to
`vggt_corners_world()` (ovmono3d/tools/prepare_wildbox_dataset.py:173). The human
face-lock annotations (`track_face_locks.json`) are keyed to this ordering::

    dims = (dx, dy, dz)                 # extents along the box's local X, Y, Z
    corner i =  (+/-dx/2, +/-dy/2, +/-dz/2)   in the order below
    world    =  R @ corner_local + center     # R's COLUMNS are the box axes

CORRECTION TO UPSTREAM
----------------------
`_FACE_CORNER_INDICES` (vggt/build_canonical_atlas.py:47) labels faces 0-3 as
+X/-X/-Y/+Y. That is wrong: read off the corner coordinates, faces 0/1 are the
-Y/+Y pair and faces 2/3 are the -X/+X pair. Faces 4/5 (+Z/-Z) are correct. We
derive `FACE_NORMAL` from the corners so the mistake cannot propagate.

EMPIRICAL VALIDATION (66/66 human-locked tracks, 8 segments)
------------------------------------------------------------
* `front`/`top`/`left` always land on three *distinct* axis pairs.
* `top` is always face 0 -> the up axis and its sign are already canonical
  (VGGT's `align_bbox_to_ground_plane` fixed them).
* ``left == up x forward`` holds for every track (right-hand rule).

=> The box axes carry: X = front/back, Y = top/bottom, Z = left/right.
=> Only ONE bit is genuinely unknown per track: which end of the X (length) axis
   is the head. Left/right is then *derived*, never guessed.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "CORNERS_LOCAL_UNIT", "FACE_CORNERS", "FACE_NORMAL_LOCAL", "OPPOSITE_FACE",
    "SEMANTIC_NAMES", "corners_of", "face_normals_world", "face_centers_world",
    "left_from", "face_index_for_direction", "derive_all_face_names",
]

# --- corners -----------------------------------------------------------------
# Signs of (x, y, z) for each of the 8 corners, in BBox3D.get_corners() order.
CORNERS_LOCAL_UNIT = np.array([
    [-1, -1, -1],  # 0
    [+1, -1, -1],  # 1
    [+1, +1, -1],  # 2
    [-1, +1, -1],  # 3
    [-1, -1, +1],  # 4
    [+1, -1, +1],  # 5
    [+1, +1, +1],  # 6
    [-1, +1, +1],  # 7
], dtype=np.float64)

# --- faces -------------------------------------------------------------------
# Face id -> its 4 corner indices. Ids match track_face_locks.json.
FACE_CORNERS: dict[int, list[int]] = {
    0: [0, 1, 5, 4],
    1: [2, 3, 7, 6],
    2: [0, 3, 7, 4],
    3: [1, 2, 6, 5],
    4: [4, 5, 6, 7],
    5: [0, 1, 2, 3],
}

# Outward unit normal of each face in the box's LOCAL frame, DERIVED from the
# corner coordinates (a face's corners all share one coordinate; its centroid,
# normalised, is the outward normal). Do not hand-edit -- fix the corners instead.
FACE_NORMAL_LOCAL: dict[int, np.ndarray] = {
    f: np.round(
        CORNERS_LOCAL_UNIT[idx].mean(axis=0)
        / np.linalg.norm(CORNERS_LOCAL_UNIT[idx].mean(axis=0))
    )
    for f, idx in FACE_CORNERS.items()
}
# => {0: -Y, 1: +Y, 2: -X, 3: +X, 4: +Z, 5: -Z}

# Face id -> which LOCAL axis (0=x, 1=y, 2=z) it is normal to. Opposite faces share an axis.
# Use this to index `dims`: the box extent along face `f` is `dims[FACE_AXIS[f]]`. Dotting a
# face's *world* normal against `dims` is meaningless -- `dims` is in the local frame.
FACE_AXIS: dict[int, int] = {f: int(np.argmax(np.abs(n))) for f, n in FACE_NORMAL_LOCAL.items()}

OPPOSITE_FACE: dict[int, int] = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}

SEMANTIC_NAMES = ("front", "back", "top", "bottom", "left", "right")


# --- geometry ----------------------------------------------------------------
def corners_of(center: np.ndarray, dims: np.ndarray, R: np.ndarray) -> np.ndarray:
    """8x3 corners in the frame `R`/`center` are expressed in (world or camera).

    `R`'s columns are the box's local axes; `dims` are the full extents along them.
    """
    local = CORNERS_LOCAL_UNIT * (np.asarray(dims, dtype=np.float64) / 2.0)
    return (np.asarray(R, dtype=np.float64) @ local.T).T + np.asarray(center, dtype=np.float64)


def face_normals_world(R: np.ndarray) -> dict[int, np.ndarray]:
    """Outward unit normal of each face, rotated into the box's parent frame."""
    R = np.asarray(R, dtype=np.float64)
    return {f: R @ n for f, n in FACE_NORMAL_LOCAL.items()}


def face_centers_world(center: np.ndarray, dims: np.ndarray, R: np.ndarray) -> dict[int, np.ndarray]:
    """Centroid of each face in the parent frame."""
    c = corners_of(center, dims, R)
    return {f: c[idx].mean(axis=0) for f, idx in FACE_CORNERS.items()}


def left_from(up: np.ndarray, forward: np.ndarray) -> np.ndarray:
    """The animal's LEFT direction. Verified against 66/66 human-locked tracks.

    Right-hand rule: left = up x forward. `right` is simply -left.
    """
    left = np.cross(np.asarray(up, dtype=np.float64), np.asarray(forward, dtype=np.float64))
    n = np.linalg.norm(left)
    if n < 1e-9:
        raise ValueError("up and forward are parallel; cannot form a frame")
    return left / n


def face_index_for_direction(R: np.ndarray, direction: np.ndarray) -> int:
    """Face whose outward normal best matches `direction` (same frame as `R`).

    This is how a semantic direction (e.g. our derived `forward`) is converted
    into a face id comparable with `track_face_locks.json`. Keeping the frame
    (vectors) as the primary representation -- and only projecting to a face id
    here -- is what makes us robust to the raw-vs-canonicalised `R` difference
    between WildBox's `R_cam` and the annotator's rotations.
    """
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    normals = face_normals_world(R)
    return max(normals, key=lambda f: float(np.dot(normals[f], d)))


def derive_all_face_names(lock: dict[str, int]) -> dict[str, int]:
    """{front, top, left} -> all six names, by opposite-face inference.

    Mirrors vggt/build_canonical_atlas.py:62 so validation compares like with like.
    """
    full = dict(lock)
    for a, b in (("front", "back"), ("top", "bottom"), ("left", "right")):
        if a in lock and b not in lock:
            full[b] = OPPOSITE_FACE[lock[a]]
    return full
