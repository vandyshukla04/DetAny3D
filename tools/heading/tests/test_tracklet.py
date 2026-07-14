"""Tests for the world-azimuth track decoder. No GPU, no model, no data.

    python -m tools.heading.tests.test_tracklet

The decoder has to do two OPPOSITE things, and it is easy to build one that does only one of them:

    a GENUINE TURN must SURVIVE          (over-smoothing destroys a real turning animal)
    a SINGLE-FRAME FLIP must be REMOVED  (an animal cannot swap ends between two frames)

A majority vote gets the second and fails the first. A plain smoother gets neither. Both failure
modes look fine in aggregate accuracy on a dataset of grazing animals that barely turn -- which is
exactly the trap the earlier track-vote fell into (it scored 100%, and was wrong by construction).
"""
from __future__ import annotations

import numpy as np

from tools.heading.tracklet import viterbi_azimuth, wrap


def _two_candidates(az_true, noise_flip, *, margin=1.0, seed=0):
    """A track where the box offers 2 hypotheses: the true heading and its 180-degree reverse.

    `noise_flip[t]` = the per-frame appearance cue is WRONG at frame t (it prefers the reverse).
    Returns (cand_az, scores) ready for the decoder.
    """
    T = len(az_true)
    cand = np.stack([az_true, wrap(az_true + np.pi)], axis=1)          # (T, 2)
    scores = np.zeros((T, 2))
    for t in range(T):
        scores[t, 1 if noise_flip[t] else 0] = margin                  # the cue's preference
    return cand, scores


def test_a_single_frame_flip_is_removed():
    """The cue is confidently WRONG on one frame. The animal cannot have spun 180 degrees and back
    between frames, so the decoder must overrule it."""
    T = 20
    az = np.full(T, 0.3)                                   # a straight-walking animal
    flip = np.zeros(T, dtype=bool)
    flip[10] = True                                        # one bad frame

    cand, sc = _two_candidates(az, flip, margin=1.0)
    raw = sc.argmax(1)
    dec = viterbi_azimuth(cand, sc, turn_std=np.radians(12))

    assert raw[10] == 1, "the test is not exercising anything -- the raw cue was already right"
    assert (dec == 0).all(), f"the impossible single-frame flip survived: {dec}"


def test_a_genuine_turn_survives():
    """The animal really does turn around over ~30 frames. The decoder must FOLLOW it, not smooth
    it away -- this is precisely what a track-level majority vote gets wrong by construction."""
    T = 60
    az = np.concatenate([np.full(15, 0.0),
                         np.linspace(0.0, np.pi, 30),      # a real, gradual 180-degree turn
                         np.full(15, np.pi)])
    az = wrap(az)
    flip = np.zeros(T, dtype=bool)                         # a perfect cue -- nothing to correct

    cand, sc = _two_candidates(az, flip, margin=1.0)
    dec = viterbi_azimuth(cand, sc, turn_std=np.radians(12))

    assert (dec == 0).all(), "a real turn was smoothed away -- the decoder is over-constrained"
    # and the decoded heading must actually rotate through ~180 degrees
    got = cand[np.arange(T), dec]
    assert abs(abs(float(wrap(got[-1] - got[0]))) - np.pi) < 0.2


def test_evidence_accumulates_against_a_noisy_cue():
    """THE POINT OF THE WHOLE MODULE. A cue that is only 65% right per frame, but whose errors are
    INDEPENDENT, must be decoded almost perfectly once the track is long enough -- because a flip
    is expensive and the majority of frames pull the other way."""
    rng = np.random.default_rng(0)
    T = 80
    az = np.full(T, -1.1)
    flip = rng.random(T) < 0.35                            # 65% correct per frame

    cand, sc = _two_candidates(az, flip, margin=1.0)
    raw_acc = float((sc.argmax(1) == 0).mean())
    dec = viterbi_azimuth(cand, sc, turn_std=np.radians(12))
    dec_acc = float((dec == 0).mean())

    assert 0.55 < raw_acc < 0.8, f"the raw cue should be mediocre, got {raw_acc}"
    assert dec_acc > 0.95, f"evidence did not accumulate: raw {raw_acc:.2f} -> decoded {dec_acc:.2f}"


def test_a_bigger_frame_gap_permits_a_bigger_turn():
    """Frames are not uniformly spaced (the tracks are strided). A long gap must allow a larger
    turn -- otherwise the decoder would reject a perfectly ordinary rotation just because two
    samples happened to be far apart in time."""
    az = np.array([0.0, 0.0, 0.9, 0.9])                    # a ~50-degree change at step 2
    cand, sc = _two_candidates(az, np.zeros(4, dtype=bool))

    tight = viterbi_azimuth(cand, sc, turn_std=np.radians(12), dt=np.array([1, 1, 1]))
    loose = viterbi_azimuth(cand, sc, turn_std=np.radians(12), dt=np.array([1, 30, 1]))
    assert (tight == 0).all() and (loose == 0).all()       # both follow the cue here

    # the cost of that same turn must be strictly lower when the gap is longer
    turn = float(wrap(az[2] - az[1]))
    c_tight = 0.5 * (turn / np.radians(12)) ** 2
    c_loose = 0.5 * (turn / (np.radians(12) * np.sqrt(30))) ** 2
    assert c_loose < c_tight / 10


def test_unusable_frames_do_not_break_the_path():
    """A frame where the cue is unusable (-inf everywhere, e.g. the animal is exactly end-on) must
    be bridged by the prior, not crash and not derail the track."""
    T = 12
    az = np.full(T, 0.2)
    cand, sc = _two_candidates(az, np.zeros(T, dtype=bool))
    sc[5] = -np.inf                                        # no evidence at all in this frame

    dec = viterbi_azimuth(cand, sc, turn_std=np.radians(12))
    assert (dec == 0).all(), f"an evidence-free frame derailed the decode: {dec}"


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
