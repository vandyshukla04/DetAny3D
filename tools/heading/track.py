"""Temporal heading tracking: turn per-frame guesses into one coherent trajectory.

!! STATUS: DOES NOT HELP IN IMAGE SPACE. Measured on held-out tracks:
!!     per-frame        : 4.0 deg / 3.1% flips / 96.8% flank
!!     + this tracker   : 3.8 deg / 3.7% flips / 96.7% flank   (i.e. no gain)
!!
!! WHY (this is the useful part -- do not repeat the mistake):
!! The synthetic tests below all pass, because they assume the underlying signal is
!! temporally smooth. **In IMAGE space it is not.** The drone moves, so an animal's
!! image-space heading changes even when the animal does not; and near head-on views the
!! projected heading vector is short and ill-conditioned, so the angle legitimately swings.
!! Smoothing a signal that is not smooth does not denoise it -- it blurs it.
!!
!! The physically smooth quantity is the animal's heading AZIMUTH IN WORLD SPACE: that is
!! what cannot change abruptly. Converting image-angle -> world azimuth needs the camera
!! geometry, which `features.npz` does not carry (only track/frame/species) -- but
!! `predict.py` HAS it. **So world-space tracking belongs at inference (predict.py), not in
!! the feature-space eval.** Until then, the honest per-frame number stands.

THE PHYSICAL PRIOR
------------------
An animal cannot swap its head and tail between two consecutive frames. A ~180-degree
reversal between adjacent frames is therefore **provably a model error**. A *gradual* turn,
accumulated over many frames, is real and must survive.

THE DOUBLED-ANGLE TRICK (this is the whole idea)
------------------------------------------------
Work in **2*theta**. A 180-degree flip maps theta -> theta+pi, so 2*theta -> 2*theta+2pi,
which is *the same angle*. **The flip noise simply disappears.** So:

    1. double the raw per-frame headings  -> the flips vanish; what remains is the animal's
       body AXIS over time, which is genuinely smooth,
    2. unwrap + smooth that axis,
    3. halve it back  -> a continuous heading trajectory, correct up to ONE global sign,
    4. resolve that single sign by majority vote against the raw predictions.

There is **no sequential propagation**, so a single bad frame cannot cascade -- which is
exactly what killed the first attempt at this (a naive neighbour-chained de-flipper seeded on
frame 0; if frame 0 was one of the ~3% flipped frames it inverted the whole track, and image-
space angles are *not* smooth near head-on views, so it "corrected" good frames too. Measured:
it took flank accuracy from 96.8% DOWN to 79.0%. Do not reintroduce it.)

WHY NOT A TRACK-LEVEL MAJORITY VOTE
-----------------------------------
It forces ONE heading for the whole track. It scores 100% on grazing zebras only because they
barely turn; it is wrong *by construction* for an animal that turns around mid-track. The
doubled-angle method keeps a per-frame heading, so a turning animal is handled.
"""
from __future__ import annotations

import numpy as np

__all__ = ["circular_smooth", "track_headings"]


def circular_smooth(angles: np.ndarray, window: int = 9) -> np.ndarray:
    """Moving average on the unit circle (never average raw angles: 359 and 1 average to 180)."""
    a = np.asarray(angles, dtype=np.float64)
    if len(a) < 2 or window < 2:
        return a
    window = min(window, len(a))
    k = np.ones(window) / window
    c = np.convolve(np.cos(a), k, mode="same")
    s = np.convolve(np.sin(a), k, mode="same")
    return np.arctan2(s, c)


def track_headings(
    frames: np.ndarray,
    angles: np.ndarray,
    conf: np.ndarray | None = None,
    *,
    window: int = 9,
) -> np.ndarray:
    """Temporally coherent per-frame heading for ONE track. Angles in radians.

    See the module docstring: double -> unwrap -> smooth -> halve -> one global sign.
    Returns per-frame headings in the caller's original ordering.
    """
    frames = np.asarray(frames)
    a = np.asarray(angles, dtype=np.float64)
    if len(a) < 3:
        return a

    order = np.argsort(frames)
    raw = a[order]
    w = np.ones(len(raw)) if conf is None else np.asarray(conf, dtype=np.float64)[order]

    # 1-2. In doubled space the 180-deg flips vanish; unwrap + smooth the body AXIS.
    #
    # NOTE: after np.unwrap the sequence is CONTINUOUS (it may run well outside [-pi, pi]),
    # so it must be smoothed as a plain signal. Passing it through a circular smoother would
    # re-wrap it via arctan2 and destroy the unwrapping -- which annihilates any genuine turn
    # (measured: a real 180-deg turn collapsed to a 3-deg span). Edge-pad so the box filter
    # does not drag the endpoints toward zero.
    doubled = np.unwrap(2.0 * raw)
    if window >= 2 and len(doubled) >= 2:
        w_eff = min(window, len(doubled))
        pad = w_eff // 2
        padded = np.pad(doubled, pad, mode="edge")
        doubled = np.convolve(padded, np.ones(w_eff) / w_eff, mode="valid")[: len(raw)]

    # 3. Halve back -> a continuous heading, correct up to ONE global sign (theta vs theta+pi).
    traj = doubled / 2.0

    # 4. Resolve that single sign by confidence-weighted majority against the raw predictions.
    agree = float(np.sum(w * np.cos(traj - raw)))
    if agree < 0:
        traj = traj + np.pi

    traj = np.arctan2(np.sin(traj), np.cos(traj))       # wrap to [-pi, pi]

    out = np.empty_like(traj)
    out[order] = traj
    return out
