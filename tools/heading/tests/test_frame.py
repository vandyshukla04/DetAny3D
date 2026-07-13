"""Pins the canonical-frame derivation.

The synthetic tests check the algebra; the empirical ones re-measure, against the real
human face-locks, the three guarantees the design rests on:

    the locked front is among our 4 horizontal candidates : 100%
    up / TOP derived from geometry alone                  : 100%
    left = up x forward, given the front face             : 100%

If any of these ever drops below 100%, geometry has started leaking error into what is
supposed to be a pure cue problem -- which is exactly the bug the earlier
"longest horizontal axis" body-axis heuristic caused (5.6% wrong; 3 tracks wrong in
every frame). Do not reintroduce it.

    python -m tools.heading.tests.test_frame        # from the DetAny3D root
"""
from __future__ import annotations

import numpy as np

from tools.heading.conventions import face_index_for_direction, face_normals_world
from tools.heading.frame import (
    frame_from_front_face,
    horizontal_faces,
    sign_up_toward_camera,
    world_up_from_boxes,
)
from tools.heading.io import (
    find_face_locks,
    load_face_locks,
    load_segment,
    segment_dir_for_locks,
)

LOCK_ROOT = "/mnt/d/3DBOX"


class Skip(Exception):
    """Raised to skip a test when its data isn't on this machine."""


def _rand_rotation(rng: np.random.Generator) -> np.ndarray:
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    return R


# --- synthetic ---------------------------------------------------------------
def test_world_up_recovers_a_shared_vertical_axis():
    """Boxes that share a ground normal but are rotated about it must all agree."""
    rng = np.random.default_rng(0)
    up = np.array([0.0, 0.0, 1.0])
    Rs = []
    for _ in range(50):
        th = rng.uniform(0, 2 * np.pi)
        c, s = np.cos(th), np.sin(th)
        # columns: two horizontal axes + the shared vertical one (in varying slots)
        Rs.append(np.column_stack([[c, s, 0], [-s, c, 0], up]))
    got = world_up_from_boxes(np.array(Rs))
    assert abs(abs(float(np.dot(got, up))) - 1.0) < 1e-6


def test_sign_up_points_from_animal_toward_camera():
    up = np.array([0.0, 0.0, 1.0])
    signed = sign_up_toward_camera(up, box_center=np.zeros(3), camera_center=np.array([0, 0, 10.0]))
    assert np.allclose(signed, up)
    signed = sign_up_toward_camera(up, box_center=np.zeros(3), camera_center=np.array([0, 0, -10.0]))
    assert np.allclose(signed, -up)


def test_horizontal_faces_excludes_top_and_bottom():
    rng = np.random.default_rng(1)
    for _ in range(20):
        R = _rand_rotation(rng)
        up = R[:, 1]                                   # make column 1 the vertical axis
        cands = horizontal_faces(R, up)
        assert len(cands) == 4
        normals = face_normals_world(R)
        for f in cands:
            assert abs(float(np.dot(normals[f], up))) < 0.9, "a near-vertical face slipped in"


def test_frame_is_orthonormal_and_forward_is_horizontal():
    rng = np.random.default_rng(2)
    for _ in range(20):
        R = _rand_rotation(rng)
        up = R[:, 1]
        for f in horizontal_faces(R, up):
            fr = frame_from_front_face(R, up, f)
            assert abs(float(np.dot(fr.forward, fr.up))) < 1e-9
            assert abs(float(np.dot(fr.left, fr.up))) < 1e-9
            assert abs(float(np.dot(fr.forward, fr.left))) < 1e-9
            assert np.allclose(fr.right, -fr.left)
            for v in (fr.forward, fr.up, fr.left):
                assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-9


def test_vertical_face_cannot_be_the_front():
    R = np.eye(3)
    up = np.array([0.0, 1.0, 0.0])
    vertical = face_index_for_direction(R, up)
    try:
        frame_from_front_face(R, up, vertical)
    except ValueError:
        return
    raise AssertionError("a vertical face was accepted as the front")


# --- empirical: the three guarantees, re-measured on the human locks ----------
_CACHE: list[tuple[np.ndarray, np.ndarray, dict[str, int]]] | None = None


def _scored() -> list[tuple[np.ndarray, np.ndarray, dict[str, int]]]:
    """(R, signed_up, lock) for every human-locked box instance.

    Memoised: the segments live on a slow NTFS mount, so loading them once per test
    (8 segments x ~200 frames x ~12 tracks of JSON) made the suite take >10 minutes.
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    locks = find_face_locks(LOCK_ROOT)
    if not locks:
        raise Skip("face-lock annotations not on this machine")

    out: list[tuple[np.ndarray, np.ndarray, dict[str, int]]] = []
    for lp in locks:
        seg = load_segment(segment_dir_for_locks(lp), rotations="canonical")
        up_un = world_up_from_boxes(np.concatenate([t.rotations for t in seg.tracks.values()]))
        for tid, lock in load_face_locks(lp).items():
            tr = seg.tracks.get(tid)
            if tr is None:
                continue
            for i, fidx in enumerate(tr.frames):
                cam = seg.cameras.get(int(fidx))
                if cam is None:
                    continue
                cam_c = -cam.extrinsic[:, :3].T @ cam.extrinsic[:, 3]
                out.append((tr.rotations[i],
                            sign_up_toward_camera(up_un, tr.centers[i], cam_c),
                            lock))
    _CACHE = out
    return out


def test_locked_front_is_always_among_the_four_candidates():
    n = bad = 0
    for R, up, lock in _scored():
        n += 1
        if lock["front"] not in horizontal_faces(R, up):
            bad += 1
    assert n > 10000, f"expected ~11k instances, got {n}"
    assert bad == 0, f"{bad}/{n} instances: the true front was not a candidate"


def test_top_and_left_are_exact_given_the_front_face():
    n = bad_top = bad_left = 0
    for R, up, lock in _scored():
        n += 1
        fr = frame_from_front_face(R, up, lock["front"])
        if face_index_for_direction(R, fr.up) != lock["top"]:
            bad_top += 1
        if face_index_for_direction(R, fr.left) != lock["left"]:
            bad_left += 1
    assert bad_top == 0, f"{bad_top}/{n} wrong TOP -- geometry is leaking error"
    assert bad_left == 0, f"{bad_left}/{n} wrong LEFT -- the cross-product rule broke"


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
