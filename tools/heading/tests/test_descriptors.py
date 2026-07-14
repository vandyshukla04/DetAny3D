"""Tests for the dense-feature layer and the anatomical template. No GPU, no model, no data.

    python -m tools.heading.tests.test_descriptors

These cover the pieces that would fail SILENTLY -- producing a plausible number rather than an
exception. The reversal identity in particular is load-bearing: `AxisTemplate.score_faces`
computes only 2 profiles for 4 hypotheses, deriving the other two by reversing. If that identity
broke, half the scores would be quietly wrong and every downstream table would still look fine.
"""
from __future__ import annotations

import numpy as np

from tools.heading.descriptors import axis_profile, foreground
from tools.heading.template import _match, opposite_slot


def test_foreground_keeps_the_centre_and_drops_the_corners():
    """The crop is cut from the animal's own 2D box, so the centre is animal and (after
    square-padding) the corners are grass. The mask must exploit exactly that."""
    gh = gw = 14
    D = 16
    g = np.zeros((gh, gw, D))
    g[..., 0] = 1.0                                     # background direction
    cy, cx = gh // 2, gw // 2
    g[cy - 4:cy + 4, cx - 4:cx + 4] = 0
    g[cy - 4:cy + 4, cx - 4:cx + 4, 1] = 1.0            # animal direction

    fg = foreground(g)
    inside = fg[cy - 3:cy + 3, cx - 3:cx + 3].mean()
    corners = np.concatenate([fg[:2, :2].ravel(), fg[:2, -2:].ravel(),
                              fg[-2:, :2].ravel(), fg[-2:, -2:].ravel()]).mean()
    assert inside > 0.95, f"animal centre was masked out ({inside:.2f})"
    assert corners < 0.05, f"grass corners were kept as foreground ({corners:.2f})"


def test_foreground_never_returns_an_empty_mask():
    """A degenerate crop (all patches identical) must fall back to 'everything', not to nothing --
    an empty mask would make every downstream profile silently zero."""
    g = np.zeros((14, 14, 8))
    g[..., 0] = 1.0
    assert foreground(g).sum() > 0


def test_reversing_the_hypothesis_exactly_reverses_the_profile():
    """LOAD-BEARING. score_faces derives the opposite hypothesis' score by reversing the profile
    instead of recomputing it. If this identity fails, 2 of the 4 scores are wrong -- silently."""
    rng = np.random.default_rng(0)
    g = rng.normal(size=(14, 14, 16))
    g /= np.linalg.norm(g, axis=-1, keepdims=True)
    fg = np.ones((14, 14), bool)
    tail, head = np.array([0.15, 0.5]), np.array([0.85, 0.5])

    p_fwd, c_fwd = axis_profile(g, fg, tail, head, bins=5)
    p_rev, c_rev = axis_profile(g, fg, head, tail, bins=5)
    assert np.allclose(p_fwd, p_rev[::-1], atol=1e-6)
    assert np.array_equal(c_fwd, c_rev[::-1])


def test_profile_resolves_a_gradient_along_the_body():
    """If the bins cannot distinguish rump from head, there is no head/tail cue to find."""
    gh = gw = 14
    D = 16
    g = np.zeros((gh, gw, D))
    for c in range(gw):                                 # a left->right anatomical gradient
        g[:, c, min(c * D // gw, D - 1)] = 1.0
    p, cnt = axis_profile(g, np.ones((gh, gw), bool),
                          np.array([0.15, 0.5]), np.array([0.85, 0.5]), bins=5)
    assert cnt.sum() > 0
    adj = [float(p[b] @ p[b + 1]) for b in range(4)]
    assert max(adj) < 0.9, f"profile bins are indistinguishable: {adj}"


def test_profile_is_empty_when_the_animal_is_exactly_end_on():
    """Head and tail project to the same point: the axis has no image extent, so there is no
    profile. It must return zero evidence rather than divide by zero."""
    g = np.zeros((14, 14, 8))
    g[..., 0] = 1.0
    p, cnt = axis_profile(g, np.ones((14, 14), bool),
                          np.array([0.5, 0.5]), np.array([0.5, 0.5]), bins=5)
    assert cnt.sum() == 0 and np.allclose(p, 0)


def test_opposite_slot_uses_face_ids_not_projected_distance():
    """Projected face centres collapse when the animal is end-on -- exactly when we most need the
    pairing to be right. So the opposite face comes from the face IDs, which cannot degenerate."""
    face_ids = np.array([2, 3, 4, 5])                   # 2<->3 and 4<->5 are the opposite pairs
    assert opposite_slot(face_ids, 0) == 1
    assert opposite_slot(face_ids, 1) == 0
    assert opposite_slot(face_ids, 2) == 3
    assert opposite_slot(face_ids, 3) == 2


def test_match_ignores_empty_bins_rather_than_averaging_noise_into_them():
    """An occluded or off-frame slice of the animal must not vote."""
    D = 16
    T = np.eye(5, D, dtype=np.float32)
    prof = np.eye(5, D, dtype=np.float32)
    full = _match(prof, np.array([10, 10, 10, 10, 10]), T)
    holey = _match(prof, np.array([10, 0, 10, 0, 10]), T)
    assert abs(full - 1.0) < 1e-6
    assert abs(holey - 1.0) < 1e-6, "empty bins polluted the score"
    assert np.isnan(_match(prof, np.zeros(5, dtype=int), T)), "no evidence must yield NaN"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    print(f"running {len(tests)} tests\n")
    for t in tests:
        try:
            t()
            print(f"  ok    {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
