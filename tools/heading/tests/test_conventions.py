"""Pins the box corner/face conventions.

These tests are the guard rail for the whole package: the upstream face-table
comments are wrong (faces 0-3 mislabelled), three corner orderings coexist across
the repos, and a silent convention error would invert every downstream number
without failing anything. So we assert the geometry *and* re-derive it against the
real human annotations.

Dependency-free on purpose -- runs with any Python, on the cluster too:

    python -m tools.heading.tests.test_conventions      # from the DetAny3D root

(pytest will also collect these if it happens to be installed.)
"""
from __future__ import annotations

import glob
import json

import numpy as np

from tools.heading.conventions import (
    FACE_CORNERS,
    FACE_NORMAL_LOCAL,
    OPPOSITE_FACE,
    corners_of,
    derive_all_face_names,
    face_centers_world,
    face_index_for_direction,
    left_from,
)

AXIS_NAME = {
    (-1, 0, 0): "-X", (1, 0, 0): "+X",
    (0, -1, 0): "-Y", (0, 1, 0): "+Y",
    (0, 0, -1): "-Z", (0, 0, 1): "+Z",
}
LOCK_GLOB = "/mnt/d/3DBOX/**/semantic_faces/track_face_locks.json"


class Skip(Exception):
    """Raised to skip a test when its data isn't on this machine."""


def _locks() -> list[str]:
    return [p for p in glob.glob(LOCK_GLOB, recursive=True) if "RECYCLE" not in p]


def _locked_tracks():
    for path in _locks():
        for tid, lock in json.load(open(path)).items():
            if {"front", "top", "left"} <= set(lock):
                yield path, tid, lock


# --- the corrected face table ------------------------------------------------
def test_face_normals_contradict_the_upstream_comments():
    """Upstream calls faces 0-3 '+X,-X,-Y,+Y'. Derived from the corners they are
    -Y,+Y,-X,+X. This test exists so nobody 'fixes' us back to the wrong table."""
    got = {f: AXIS_NAME[tuple(int(v) for v in n)] for f, n in FACE_NORMAL_LOCAL.items()}
    assert got == {0: "-Y", 1: "+Y", 2: "-X", 3: "+X", 4: "+Z", 5: "-Z"}, got


def test_each_face_is_planar_and_normal_points_outward():
    """A face's 4 corners share one coordinate, and its normal points away from the
    centre. (Upstream winding is inconsistent, so we never trust cross-products.)"""
    for f, idx in FACE_CORNERS.items():
        c = corners_of(np.zeros(3), np.array([2.0, 4.0, 6.0]), np.eye(3))[idx]
        n = FACE_NORMAL_LOCAL[f]
        const_axis = int(np.argmax(np.abs(n)))
        assert np.allclose(c[:, const_axis], c[0, const_axis]), f"face {f} not planar"
        assert float(np.dot(c.mean(axis=0), n)) > 0, f"face {f} normal points inward"


def test_opposite_faces_are_antiparallel():
    for f, o in OPPOSITE_FACE.items():
        assert np.allclose(FACE_NORMAL_LOCAL[f], -FACE_NORMAL_LOCAL[o])


# --- frame algebra -----------------------------------------------------------
def test_left_is_up_cross_forward_and_is_orthonormal():
    up, fwd = np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    left = left_from(up, fwd)
    assert np.allclose(left, np.cross(up, fwd))
    for a, b in ((up, fwd), (up, left), (fwd, left)):
        assert abs(float(np.dot(a, b))) < 1e-9


def test_left_from_rejects_degenerate_frame():
    try:
        left_from(np.array([1.0, 0, 0]), np.array([1.0, 0, 0]))
    except ValueError:
        return
    raise AssertionError("expected ValueError for parallel up/forward")


def test_face_index_for_direction_roundtrips_under_rotation():
    """Rotating the box must not change which face a *body-fixed* direction picks."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1
        for f, n_local in FACE_NORMAL_LOCAL.items():
            assert face_index_for_direction(R, R @ n_local) == f


def test_face_centers_lie_at_half_extent():
    dims = np.array([2.0, 4.0, 6.0])
    for f, c in face_centers_world(np.zeros(3), dims, np.eye(3)).items():
        axis = int(np.argmax(np.abs(FACE_NORMAL_LOCAL[f])))
        assert np.isclose(abs(c[axis]), dims[axis] / 2.0)


# --- the empirical guarantee (real human annotations) ------------------------
def test_human_locks_satisfy_our_conventions():
    """The load-bearing claim: across EVERY human-locked track,
      (a) front/top/left land on three distinct axis pairs, and
      (b) left == up x forward.
    If this fails, our frame algebra disagrees with the ground truth."""
    if not _locks():
        raise Skip("face-lock annotations not on this machine")
    pairs = {frozenset((0, 1)), frozenset((2, 3)), frozenset((4, 5))}
    total = 0
    for path, tid, lock in _locked_tracks():
        total += 1
        fr, tp, lf = lock["front"], lock["top"], lock["left"]
        assert {frozenset((f, OPPOSITE_FACE[f])) for f in (fr, tp, lf)} == pairs, (path, tid)
        predicted = left_from(FACE_NORMAL_LOCAL[tp], FACE_NORMAL_LOCAL[fr])
        assert np.allclose(predicted, FACE_NORMAL_LOCAL[lf]), f"left != up x forward: {tid} in {path}"
    assert total >= 60, f"expected ~66 locked tracks, found {total}"


def test_only_the_front_sign_is_unknown():
    """Empirically: `top` is always face 0 (VGGT ground-alignment canonicalised UP),
    and left/right follows from the cross product. So exactly ONE bit per track --
    the front sign -- is genuinely unknown. This is the whole problem statement."""
    if not _locks():
        raise Skip("face-lock annotations not on this machine")
    tops = {lock["top"] for _p, _t, lock in _locked_tracks()}
    fronts = {lock["front"] for _p, _t, lock in _locked_tracks()}
    assert tops == {0}, f"UP is not canonical after all: {tops}"
    assert fronts == {2, 3}, f"front should be the +/-X pair only: {fronts}"


def test_derive_all_face_names_matches_locks():
    if not _locks():
        raise Skip("face-lock annotations not on this machine")
    for _path, _tid, lock in _locked_tracks():
        full = derive_all_face_names(lock)
        assert full["back"] == OPPOSITE_FACE[lock["front"]]
        assert full["bottom"] == OPPOSITE_FACE[lock["top"]]
        assert full["right"] == OPPOSITE_FACE[lock["left"]]


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = skipped = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Skip as e:
            print(f"  SKIP  {t.__name__}  ({e})")
            skipped += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
