"""Temporal heading tracking: turn per-frame guesses into one coherent trajectory.

THE PHYSICAL PRIOR
------------------
An animal cannot swap its head and tail between two consecutive frames. So a ~180-degree
reversal between adjacent frames is **provably a model error**, never a real motion. A
*gradual* turn, accumulated over many frames, is real and must be preserved.

That single observation is what separates this from the two naive alternatives:

* **Per-frame independent** (what we had): 2.8% of frames are flipped. Each flip inverts the
  LEFT/RIGHT flank tag and would poison a re-ID gallery.
* **Track-level majority vote**: forces ONE heading for the whole track. It scored 100% on the
  grazing zebras only because they barely turn. It is *wrong by construction* for any animal
  that turns around mid-track, which is exactly the interesting case.

So we do neither. We walk the track, flip only the physically-impossible reversals (judged
against the running, already-corrected neighbour -- not a global anchor, so a slow turn
accumulates correctly), then smooth.
"""
from __future__ import annotations

import numpy as np

__all__ = ["correct_flips", "circular_smooth", "track_headings"]


def correct_flips(
    angles: np.ndarray, conf: np.ndarray | None = None, *, max_jump_deg: float = 90.0
) -> np.ndarray:
    """Remove impossible ~180-degree reversals from a time-ordered heading sequence.

    `angles` in radians, time-ordered. Comparison is against the running CORRECTED heading,
    so a genuine gradual turn accumulates and survives; only a jump larger than
    `max_jump_deg` in a single step -- which no animal can perform -- is treated as a flip
    and corrected by adding pi.

    We seed from the most confident frame and propagate outward in both directions, so one
    bad frame at the start cannot poison the whole track.
    """
    a = np.asarray(angles, dtype=np.float64).copy()
    n = len(a)
    if n < 2:
        return a
    conf = np.ones(n) if conf is None else np.asarray(conf, dtype=np.float64)

    thresh = np.radians(max_jump_deg)

    def wrap(x: np.ndarray | float) -> np.ndarray | float:
        return np.arctan2(np.sin(x), np.cos(x))

    seed = int(np.argmax(conf))

    for i in range(seed + 1, n):                       # forward
        if abs(wrap(a[i] - a[i - 1])) > thresh:
            a[i] = wrap(a[i] + np.pi)
    for i in range(seed - 1, -1, -1):                  # backward
        if abs(wrap(a[i] - a[i + 1])) > thresh:
            a[i] = wrap(a[i] + np.pi)
    return a


def circular_smooth(angles: np.ndarray, window: int = 5) -> np.ndarray:
    """Moving average on the unit circle (never average raw angles -- 359 and 1 average to 180)."""
    a = np.asarray(angles, dtype=np.float64)
    if len(a) < 2 or window < 2:
        return a
    k = np.ones(window) / window
    c = np.convolve(np.cos(a), k, mode="same")
    s = np.convolve(np.sin(a), k, mode="same")
    return np.arctan2(s, c)


def track_headings(
    frames: np.ndarray,
    angles: np.ndarray,
    conf: np.ndarray | None = None,
    *,
    max_jump_deg: float = 90.0,
    window: int = 5,
) -> np.ndarray:
    """Full pipeline for ONE track: sort by time -> de-flip -> smooth -> restore input order."""
    order = np.argsort(np.asarray(frames))
    a = np.asarray(angles, dtype=np.float64)[order]
    c = None if conf is None else np.asarray(conf, dtype=np.float64)[order]

    a = correct_flips(a, c, max_jump_deg=max_jump_deg)
    a = circular_smooth(a, window=window)

    out = np.empty_like(a)
    out[order] = a
    return out
