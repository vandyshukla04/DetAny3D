"""Tests for the papersubdata loader and the allocentric frame.

    python -m tools.heading.tests.test_papersub

Zero dependencies beyond numpy. The tests that need the real dataset SKIP when it is absent,
so this suite runs on the cluster too.

The allocentric round-trip is the one that must be exactly right: it is the bridge between
what DINOv3 can see (the animal's orientation relative to the viewing ray) and the WORLD
heading we actually want. A sign error here would be invisible in the loss and catastrophic
in the output.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.heading.papersub import Camera, Segment, Track, iter_segments

ROOT = Path("/mnt/d/3DBOX/papersubdata")


class Skip(Exception):
    """Raised to skip a test when its data isn't on this machine."""


def _rand_rotation(rng):
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return R * (-1 if np.linalg.det(R) < 0 else 1)


def _toy_segment(rng, up=np.array([0.0, 0.0, 1.0])):
    """One animal, one camera above it, on a flat world with a known `up`."""
    R = _rand_rotation(rng)
    R[:, 2] = up                                     # make one box axis the vertical
    R[:, 0] -= np.dot(R[:, 0], up) * up
    R[:, 0] /= np.linalg.norm(R[:, 0])
    R[:, 1] = np.cross(up, R[:, 0])

    centre = rng.normal(size=3) * np.array([5, 5, 0])
    cam_c = centre + np.array([3.0, -2.0, 20.0])     # drone: above and off to one side
    Rc = _rand_rotation(rng)
    E = np.concatenate([Rc, (-Rc @ cam_c)[:, None]], axis=1)

    cam = Camera(0, E, np.array([[1200.0, 0, 960], [0, 1200.0, 540], [0, 0, 1]]),
                 1920, 1080, "frame_000000.jpg")
    tr = Track("0", np.array([0]), centre[None], np.array([[2.0, 1.0, 1.5]]), R[None],
               np.array([[0.0, 0, 1, 1]]))
    return Segment(Path("."), "zebr1", "vid", "seg1", {"0": tr}, {0: cam})


# --------------------------------------------------------------------------------------
def test_camera_centre_round_trips():
    rng = np.random.default_rng(0)
    for _ in range(20):
        c = rng.normal(size=3) * 10
        R = _rand_rotation(rng)
        cam = Camera(0, np.concatenate([R, (-R @ c)[:, None]], axis=1), np.eye(3), 1, 1, "x")
        assert np.allclose(cam.center, c, atol=1e-9), cam.center


def test_projection_matches_manual_pinhole():
    rng = np.random.default_rng(1)
    seg = _toy_segment(rng)
    cam = seg.cameras[0]
    p = rng.normal(size=(5, 3))
    pc = cam.world_to_cam(p)
    uv = (pc @ cam.K.T)[:, :2] / (pc @ cam.K.T)[:, 2:3]
    got = cam.project(p)
    ok = np.isfinite(got).all(axis=1)
    assert np.allclose(got[ok], uv[ok], atol=1e-9)


def test_allocentric_round_trip_is_the_identity():
    """world direction -> alpha -> world direction must be a no-op, under any camera.

    This is the bridge between DINOv3 (which sees the animal relative to the viewing ray) and
    the world heading. A sign slip here is silent in the loss and fatal in the output.
    """
    rng = np.random.default_rng(2)
    for _ in range(200):
        seg = _toy_segment(rng)
        tr = seg.tracks["0"]
        up = seg.up_at(tr, 0)

        d = rng.normal(size=3)
        d -= np.dot(d, up) * up                      # any HORIZONTAL world direction
        n = np.linalg.norm(d)
        if n < 1e-6:
            continue
        d /= n

        back = seg.dir_of_alpha(tr, 0, seg.alpha_of(tr, 0, d))
        assert np.allclose(back, d, atol=1e-8), f"{back} != {d}"


def test_allocentric_basis_is_orthonormal_and_horizontal():
    rng = np.random.default_rng(3)
    for _ in range(50):
        seg = _toy_segment(rng)
        r, s, up = seg.allocentric_basis(seg.tracks["0"], 0)
        for v in (r, s):
            assert abs(np.linalg.norm(v) - 1) < 1e-9
            assert abs(np.dot(v, up)) < 1e-8, "the basis must lie in the ground plane"
        assert abs(np.dot(r, s)) < 1e-8
        assert np.allclose(np.cross(r, s), up, atol=1e-8), "r x s must be +up (right-handed)"


def test_alpha_zero_means_walking_directly_away_from_the_camera():
    """Anchors the sign convention. alpha=0 <=> heading along +r, and r points camera->animal,
    so the animal is walking AWAY. Without this, `alpha` could be off by pi and nothing else
    in the suite would notice."""
    rng = np.random.default_rng(4)
    seg = _toy_segment(rng)
    tr = seg.tracks["0"]
    r, _, _ = seg.allocentric_basis(tr, 0)
    assert abs(seg.alpha_of(tr, 0, r)) < 1e-9
    assert abs(abs(seg.alpha_of(tr, 0, -r)) - np.pi) < 1e-9


def test_four_horizontal_face_candidates():
    rng = np.random.default_rng(5)
    for _ in range(20):
        seg = _toy_segment(rng)
        faces = seg.horizontal_face_dirs(seg.tracks["0"], 0)
        assert len(faces) == 4, f"expected 4 front candidates, got {len(faces)}"
        up = seg.up_at(seg.tracks["0"], 0)
        for d in faces.values():
            assert abs(np.dot(d, up)) < 1e-8
            assert abs(np.linalg.norm(d) - 1) < 1e-9


# ---- against the real dataset (skipped if it isn't mounted) ---------------------------
def test_real_segments_pass_the_518_scale_check():
    """`Segment.scale` MEASURES the bbox_2d -> full-res ratio by projecting the 3D centres.
    If any real segment breaks the convention we assume, it must raise -- not silently hand
    back a crop of the grass, which is what happened last time."""
    if not ROOT.is_dir():
        raise Skip(f"{ROOT} not mounted")
    n = 0
    for seg in iter_segments(ROOT):
        s = seg.scale                                # raises on a mismatch
        want = next(iter(seg.cameras.values())).width / 518
        assert abs(s - want) / want < 0.06, f"{seg.key}: scale {s:.3f} vs {want:.3f}"
        n += 1
        if n >= 30:
            break
    if n == 0:
        raise Skip("no segments found")


def test_real_crop_boxes_land_inside_the_frame():
    if not ROOT.is_dir():
        raise Skip(f"{ROOT} not mounted")
    n = 0
    for seg in iter_segments(ROOT):
        cam = next(iter(seg.cameras.values()))
        for tr in seg.tracks.values():
            for i in range(0, len(tr), max(1, len(tr) // 5)):
                b = seg.crop_box(tr, i)
                assert b[2] > b[0] and b[3] > b[1], f"{seg.key}: degenerate box {b}"
                assert b[0] > -cam.width and b[2] < 2 * cam.width, f"{seg.key}: box off-frame {b}"
        n += 1
        if n >= 10:
            break
    if n == 0:
        raise Skip("no segments found")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = skipped = failed = 0
    print(f"running {len(tests)} tests\n")
    for t in tests:
        try:
            t()
            print(f"  ok    {t.__name__}")
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
