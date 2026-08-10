"""The held-out split. Defined ONCE, and RECORDED in every checkpoint that is trained under it.

SPLIT BY VIDEO. NEVER BY TRACK, NEVER BY FRAME.
-----------------------------------------------
Every track in a video shares one flight, altitude, herd and light, so holding out a *track*
leaves its whole scene in training. Measured: a track-split reported 96.7% while the model
visibly broke on unseen footage.

STRATIFY BY SPECIES.
--------------------
A plain random split over 53 videos put ALL FOUR giraffe videos in train, so giraffe silently
vanished from the test set and the "overall" number quietly covered three species.

THE HUMAN-LOCKED VIDEOS ARE HELD OUT FOREVER.
---------------------------------------------
Two zebra videos carry 11,084 human face-locks and are our ONLY independent check on the motion
labels -- and, being grazing animals, our only test of whether the cue transfers from walkers to
standers. A model that trained on them cannot be scored on them. They never enter training,
under any seed.

WHY THE SPLIT IS RECORDED IN THE CHECKPOINT  (this bug already bit us)
----------------------------------------------------------------------
A head was trained under one split, then evaluated under a different one after the split code
changed. Videos the evaluator "held out" had been in the model's training set, and it reported
**94.3%** where the truth was **79.8%**. Nothing could catch it, because the checkpoint carried
no record of what it had seen.

So: `split_fingerprint()` goes into every checkpoint, and `assert_matches()` is called by every
evaluator before it reports a number. A mismatch is a hard failure, never a warning.
"""
from __future__ import annotations

import hashlib

import numpy as np

__all__ = [
    "SPLIT_VERSION",
    "HUMAN_LOCKED_VIDEOS",
    "video_split",
    "template_val_split",
    "split_fingerprint",
    "assert_matches",
]

# Bump whenever the SEMANTICS of the split change (stratification, exclusions, frac).
# Checkpoints record it; evaluators refuse to score across a bump.
SPLIT_VERSION = 2

# The 2 videos carrying the 11,084 human face-locks. Permanently excluded from training.
HUMAN_LOCKED_VIDEOS = (
    "DJI_20250802085130_0007_V",
    "DJI_20250802085520_0008_V",
)


def video_split(species: np.ndarray, video: np.ndarray, *, frac: float = 1 / 3, seed: int = 0):
    """Boolean TEST mask + the sorted list of held-out videos.

    ~`frac` of each species' videos are held out, PLUS the human-locked videos, always.
    """
    rng = np.random.default_rng(seed)
    test: set[str] = set()
    for s in sorted(set(species.tolist())):
        vids = np.unique(video[species == s])
        # The locked videos are already held out; don't let them also consume the species' quota.
        pool = np.array([v for v in vids if str(v) not in HUMAN_LOCKED_VIDEOS])
        rng.shuffle(pool)
        k = max(1, int(round(frac * len(vids))))      # >=1 held-out video per species, always
        test.update(pool[:k].tolist())
    test.update(v for v in HUMAN_LOCKED_VIDEOS if v in set(map(str, video)))

    te = np.array([str(v) in test for v in video])
    return te, sorted(test)


def template_val_split(species: np.ndarray, video: np.ndarray, *, n_val: int = 10, seed: int = 0):
    """Carve a VALIDATION set out of the TEMPLATE videos, for selecting hyper-parameters.

    WHY THIS EXISTS
    ---------------
    The descriptor configuration (layer / facet / resolution) must not be chosen using the
    evaluation videos -- otherwise the reported accuracy is partly a selection effect. `sweep.py`
    did exactly that: it fitted on the template videos and *scored on the held-out evaluation
    videos*. This function supplies the honest alternative: a second, video-level split INSIDE the
    template videos, so the configuration is chosen on data the final evaluation never sees.

    THE RULE, FIXED IN ADVANCE (do not tune it after seeing results)
    ----------------------------------------------------------------
    Largest-remainder apportionment of `n_val` validation videos across species, proportional to
    each species' template-video count, subject to:
      * a species with >= 2 videos gets >= 1 validation video (so it is represented in both), and
      * every species keeps >= 1 fit video (so its template can be built at all).

    Callers pass ONLY the template rows (i.e. `species[~te], video[~te]`); passing the whole
    dataset would leak the evaluation videos straight back in, which is the bug this replaces.

    Returns `(val_mask, fit_videos, val_videos)` -- `val_mask` is over the rows passed in.
    """
    spec = sorted(set(map(str, species.tolist())))
    n_by = {s: len(np.unique(video[species == s])) for s in spec}
    total = sum(n_by.values())
    if total == 0:
        raise ValueError("no template videos to split")

    quota = {s: n_val * n_by[s] / total for s in spec}
    k = {s: int(np.floor(quota[s])) for s in spec}
    for s in spec:                                   # represented in validation where possible,
        if n_by[s] >= 2:                             # and never at the cost of its own template
            k[s] = max(k[s], 1)
        k[s] = min(k[s], max(n_by[s] - 1, 0))
    while sum(k.values()) < n_val:                   # hand out the remainder, largest first
        c = max((s for s in spec if k[s] < n_by[s] - 1),
                key=lambda s: quota[s] - k[s], default=None)
        if c is None:
            break
        k[c] += 1
    while sum(k.values()) > n_val:
        c = max((s for s in spec if k[s] > 1), key=lambda s: k[s] - quota[s], default=None)
        if c is None:
            break
        c and k.__setitem__(c, k[c] - 1)

    rng = np.random.default_rng(seed)
    val: set[str] = set()
    for s in spec:
        pool = np.array(sorted(np.unique(video[species == s]).tolist()))
        rng.shuffle(pool)
        val.update(map(str, pool[: k[s]].tolist()))

    val_mask = np.array([str(v) in val for v in video])
    return val_mask, sorted(set(map(str, video[~val_mask]))), sorted(val)


def split_fingerprint(held_out, *, seed: int) -> dict:
    """What a checkpoint must record so an evaluator can prove it is scoring honestly."""
    vids = sorted(map(str, held_out))
    return {
        "split_version": SPLIT_VERSION,
        "split_seed": int(seed),
        "held_out": vids,
        "held_out_hash": hashlib.sha1("\n".join(vids).encode()).hexdigest()[:12],
    }


def assert_matches(ckpt: dict, held_out, *, seed: int, what: str = "checkpoint") -> None:
    """Refuse to report a number from a model trained under a different split.

    This is a hard failure by design. A warning would be ignored, and the failure mode it guards
    against is a *believably good* score (94.3% vs the true 79.8%) -- the kind nobody questions.
    """
    want = split_fingerprint(held_out, seed=seed)
    got = ckpt.get("split")
    if got is None:
        raise RuntimeError(
            f"{what} records no split. It predates the leak fix and cannot be scored honestly "
            f"-- retrain with train_head.py."
        )
    if got.get("split_version") != want["split_version"]:
        raise RuntimeError(
            f"{what} was trained under split_version={got.get('split_version')}, this code is "
            f"v{want['split_version']}. The held-out sets are not the same. Retrain."
        )
    if got.get("held_out_hash") != want["held_out_hash"]:
        extra = sorted(set(want["held_out"]) - set(got.get("held_out", [])))
        raise RuntimeError(
            f"{what} held out {len(got.get('held_out', []))} videos; this evaluation holds out "
            f"{len(want['held_out'])}. Videos being scored as 'held out' were in TRAINING: "
            f"{extra[:5]}{' ...' if len(extra) > 5 else ''}. Retrain, or pass the matching seed."
        )
