"""The held-out split. Defined ONCE, so every method is scored on the same videos.

WHY THIS IS ITS OWN MODULE
--------------------------
`parts.py` (training-free) and `train_head.py` (supervised) exist to be COMPARED. If each rolls
its own split, the two accuracies are measured on different videos and the comparison is
meaningless -- and the drift is silent, because both still print a plausible number.

SPLIT BY VIDEO. NEVER BY TRACK, NEVER BY FRAME.
-----------------------------------------------
Every track in a video shares one flight, altitude, herd and light, so holding out a *track*
leaves its whole scene in training. Measured: a track-split reported 96.7% while the model
visibly broke on unseen footage.

STRATIFY BY SPECIES.
--------------------
A plain random split over 53 videos put ALL FOUR giraffe videos in train, so giraffe silently
vanished from the test set and the reported "overall" number covered only 3 species. Holding
out a fraction of each species' videos separately makes that impossible.
"""
from __future__ import annotations

import numpy as np

__all__ = ["video_split"]


def video_split(species: np.ndarray, video: np.ndarray, *, frac: float = 1 / 3, seed: int = 0):
    """Boolean test mask + the held-out video names. ~`frac` of EACH species' videos."""
    rng = np.random.default_rng(seed)
    test: set[str] = set()
    for s in sorted(set(species.tolist())):
        vids = np.unique(video[species == s])
        rng.shuffle(vids)
        k = max(1, int(round(frac * len(vids))))       # >=1 held-out video per species, always
        test.update(vids[:k].tolist())
    te = np.array([str(v) in test for v in video])
    return te, test
