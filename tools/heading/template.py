"""THE METHOD: an anatomical profile along the body axis, and the head/tail decision it yields.

    tmpl = AxisTemplate.fit(ex, crops, idx_train, cfg)      # from WALKING crops -- nothing trained
    s    = tmpl.score_faces(g, fg, face_uv, face_ids, species)   # (4,) score per candidate front face

Imported by `sweep.py` (X1), `tracklet.py` (X2) and `evaluate.py` (X3), so all three score the
SAME descriptors. If each rolled its own copy they would drift, and the drift would be invisible
-- every one of them would still print a plausible number.

THE IDEA
--------
Geometry gives the animal's body AXIS but never its SIGN (the box's PCA/SVD axis signs are
arbitrary). Locomotion gives the sign, but only while the animal walks -- 16% of frames. So the
one thing we need from appearance is: WHICH END IS THE HEAD?

DINOv3 answers exactly that and nothing more. It matches head-to-head and rump-to-rump, and it is
mirror-blind (left and right flanks are near-mirror images) -- which is fine, because left/right
is geometry (`left = up x forward`), never appearance.

HOW A HYPOTHESIS IS SCORED
--------------------------
Each of the box's 4 horizontal faces is a hypothesis "the head is at THIS end". For a hypothesis,
project the animal's foreground patches onto the axis (opposite face -> this face) and bin them:
that is a profile, rump -> head. Score it against the species' anatomical template. The
hypothesis whose profile best explains the head->rump gradient wins.

This resolves the AXIS and the SIGN in a single score -- which matters, because the "longest
horizontal axis" shortcut picks the wrong axis 13% of the time (zebra: 26%), so a sign-only cue
is capped at 87% no matter how good it is.

Note the profile of a hypothesis is the exact REVERSE of the profile of its opposite face, so 4
hypotheses cost only 2 profile computations.

NOTHING IS TRAINED. The template is the mean profile of walking crops, where motion already told
us which end is the head. Fitting it is an average.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tools.heading.conventions import OPPOSITE_FACE
from tools.heading.descriptors import Config, DenseExtractor, axis_profile, foreground

if TYPE_CHECKING:
    from tools.heading.cropset import CropSet

__all__ = ["AxisTemplate", "Accumulator", "choose", "opposite_slot"]


def opposite_slot(face_ids: np.ndarray, j: int) -> int:
    """Slot (0..3) of the face diametrically opposite slot `j`.

    Uses the real face IDs, NOT the distance between projected centres -- those degenerate when
    the animal is viewed end-on and the two centres nearly coincide, which is exactly when we most
    need the answer to be right.
    """
    want = OPPOSITE_FACE[int(face_ids[j])]
    hit = np.where(np.asarray(face_ids) == want)[0]
    return int(hit[0]) if len(hit) else int(j)


@dataclass(frozen=True)
class AxisTemplate:
    """Per-species mean anatomical profile (rump -> head) along the body axis."""

    cfg: Config
    templates: dict[str, np.ndarray]        # species -> (bins, D), L2-normalised rows
    n_fitted: dict[str, int]

    # ---- fitting: an average over walking crops. No gradient descent anywhere. ----
    @classmethod
    def fit(cls, ex: DenseExtractor, crops: "CropSet", idx, cfg: Config,
            batch: int = 32) -> "AxisTemplate":
        acc = Accumulator(cfg)
        for b in range(0, len(idx), batch):
            items = crops.batch(idx[b: b + batch])
            G = ex.grid(np.stack([it.image for it in items]), cfg)
            for g, it in zip(G, items):
                acc.add(g, it)
        return acc.build()

    # ---- scoring ----
    def score_faces(self, g: np.ndarray, fg: np.ndarray, face_uv: np.ndarray,
                    face_ids: np.ndarray, species: str) -> np.ndarray:
        """(4,) score for "the head is at face slot j". Higher is better; NaN if unusable."""
        T = self.templates.get(str(species))
        if T is None:
            return np.full(4, np.nan)
        Tc = _centre(T)                                   # score the GRADIENT, not the DC
        Tc = Tc / max(float(np.linalg.norm(Tc)), 1e-9)    # unit template => scores comparable
        #                                                   across species (the margin feeds the
        #                                                   temporal decoder as a confidence)

        out = np.full(4, np.nan)
        done: set[int] = set()
        for j in range(4):
            if j in done:
                continue
            t = opposite_slot(face_ids, j)
            prof, cnt = axis_profile(g, fg, face_uv[t], face_uv[j], self.cfg.bins)
            if not cnt.sum():
                continue
            # The opposite hypothesis is the SAME profile read backwards -- one computation answers
            # both ends of this axis. (Unit-tested: reversing the hypothesis reverses the profile.)
            out[j] = _match(prof, cnt, Tc)
            out[t] = _match(prof[::-1], cnt[::-1], Tc)
            done.update({j, t})
        return out

    def predict_face(self, g, fg, face_uv, face_ids, species, **kw) -> tuple[int, float]:
        """Convenience: score, then choose. Callers that need several choices from the SAME scores
        (e.g. the sweep, which compares 4-way vs geometry-restricted vs prior) must call
        `score_faces` once and `choose` N times -- otherwise they pay for the descriptors N times
        over, which is exactly what made the sweep crawl."""
        return choose(self.score_faces(g, fg, face_uv, face_ids, species), face_ids, **kw)


def choose(scores: np.ndarray, face_ids, *, axis: int | None = None,
           axis_prior: float = 0.0) -> tuple[int, float]:
    """Pick the front face from already-computed scores. Pure -- no model, no descriptors.

    Kept separate from `score_faces` because the expensive part (the descriptors) is shared across
    every decision rule, while the rules themselves are trivial. The sweep compares three of them
    on the same crop; fusing them into one call meant recomputing the descriptors three times.

    `axis`       -- a slot on the geometric body axis. GEOMETRY PROPOSES, APPEARANCE DISPOSES: only
                    that axis' two ends are considered. Picking the axis is geometry's job (the
                    box's longest horizontal axis is right 87% of the time) and appearance is
                    measurably bad at it -- letting appearance override geometry is what dropped the
                    4-way to 39% while the head/tail cue itself was 83.7%.
    `axis_prior` -- the soft version: a bonus on the geometric axis' ends rather than an outright
                    ban on the others. The principled middle ground, since geometry is right 87% of
                    the time, not 100%.

    Returns (slot, margin). The MARGIN is what the temporal decoder consumes as confidence: a
    head-on animal yields a near-zero margin, and that is honest -- the cue genuinely is not visible
    in that frame.
    """
    s = np.asarray(scores, dtype=np.float64)
    if np.isnan(s).all():
        return -1, 0.0
    s = np.where(np.isnan(s), -np.inf, s)

    if axis is not None:
        ends = {int(axis), opposite_slot(face_ids, int(axis))}
        if axis_prior > 0:
            s = s + np.array([axis_prior if j in ends else 0.0 for j in range(len(s))])
        else:                                              # hard restriction to the geometric axis
            s = np.where([j in ends for j in range(len(s))], s, -np.inf)

    order = np.argsort(s)[::-1]
    best = int(order[0])
    runner = s[order[1]] if np.isfinite(s[order[1]]) else s[best]
    return best, float(s[best] - runner)


class Accumulator:
    """Builds one species-keyed AxisTemplate incrementally, from crops whose head end is known.

    Exposed (rather than hidden inside `fit`) so the sweep can fit templates for EVERY LAYER from a
    single forward pass -- `output_hidden_states` already returns all of them, and re-running the
    ViT once per layer was making the sweep 24x slower than it needed to be.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.acc: dict[str, np.ndarray] = {}
        self.wts: dict[str, np.ndarray] = {}
        self.n: dict[str, int] = {}

    def add(self, g: np.ndarray, it) -> None:
        fg = foreground(g, it.instance)                    # the SAM mask when we have one
        t = opposite_slot(it.face_ids, it.y_face)          # y_face = TRUE head end, from motion
        prof, cnt = axis_profile(g, fg, it.face_uv[t], it.face_uv[it.y_face], self.cfg.bins)
        if not cnt.sum():
            return
        sp = it.species
        if sp not in self.acc:
            self.acc[sp] = np.zeros_like(prof, dtype=np.float64)
            self.wts[sp] = np.zeros(self.cfg.bins, dtype=np.float64)
            self.n[sp] = 0
        self.acc[sp] += prof * cnt[:, None]                # weight by the evidence in each bin
        self.wts[sp] += cnt
        self.n[sp] += 1

    def build(self) -> "AxisTemplate":
        templates = {}
        for sp, a in self.acc.items():
            m = a / np.maximum(self.wts[sp], 1.0)[:, None]
            nrm = np.linalg.norm(m, axis=1, keepdims=True)
            templates[sp] = (m / np.maximum(nrm, 1e-9)).astype(np.float32)
        return AxisTemplate(cfg=self.cfg, templates=templates, n_fitted=dict(self.n))


def _centre(P: np.ndarray) -> np.ndarray:
    """Remove the across-bin mean: keep the ANATOMICAL GRADIENT, drop the shared "animal-ness".

    THIS IS THE 4-WAY FIX. Every patch on an animal looks like "animal", so the cosine between any
    profile bin and any template bin sits around 0.9 -- FOR ANY AXIS, including the wrong one. A
    profile taken across the animal's WIDTH is nearly symmetric and flat, yet still scored high on
    that shared DC component, so it could beat the true axis. The discriminative signal is the
    VARIATION along the axis, not the absolute descriptors.

    After centring, a flat (wrong-axis) profile collapses to ~0 while the true axis keeps its
    head->rump structure.
    """
    return P - P.mean(axis=0, keepdims=True)


def _match(prof: np.ndarray, cnt: np.ndarray, template_c: np.ndarray) -> float:
    """Evidence-weighted correlation between a crop's CENTRED profile and the centred template.

    Both are centred, so this correlates GRADIENTS -- head->rump structure against head->rump
    structure -- and the shared "animal-ness" that used to dominate is gone.

    The profile is deliberately NOT normalised to unit length. Its magnitude is its CONTRAST along
    the axis, and contrast is itself evidence: the true body axis has a strong head->rump gradient,
    while a profile taken across the animal's width is nearly flat. Normalising would rescale that
    flat profile up into pure noise, which can then beat the truth by luck. Since the four
    hypotheses are compared WITHIN ONE CROP, keeping the raw magnitude lets the true axis win on
    alignment AND contrast together.

    Empty bins contribute nothing rather than noise -- an occluded or off-frame slice of the animal
    should not vote.
    """
    w = cnt.astype(np.float64)
    tot = w.sum()
    if tot < 1:
        return float("nan")

    # Centre using ONLY the bins that carry evidence. An empty bin is a ZERO ROW, and including it
    # in the mean drags the centre toward the origin -- so an animal with an occluded rump would be
    # under-centred and the DC would creep back in through the gap. We can only centre what we
    # actually see.
    mu = (prof * w[:, None]).sum(0) / tot
    sim = np.einsum("bd,bd->b", prof - mu, template_c)
    return float((sim * w).sum() / tot)      # empty bins have w=0 => they contribute exactly 0
