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

import io
from dataclasses import dataclass

import numpy as np

from tools.heading.conventions import OPPOSITE_FACE
from tools.heading.descriptors import Config, DenseExtractor, axis_profile, foreground

__all__ = ["AxisTemplate", "opposite_slot"]


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
    def fit(cls, ex: DenseExtractor, crops, idx, cfg: Config, batch: int = 32) -> "AxisTemplate":
        from PIL import Image

        jpeg, face_uv, y_face = crops["jpeg"], crops["face_uv"], crops["y_face"]
        face_ids, species = crops["face_ids"], crops["species"]

        acc: dict[str, np.ndarray] = {}
        wts: dict[str, np.ndarray] = {}
        n: dict[str, int] = {}

        for b in range(0, len(idx), batch):
            ii = [int(k) for k in idx[b: b + batch]]
            imgs = np.stack([np.asarray(Image.open(io.BytesIO(jpeg[i])).convert("RGB")) for i in ii])
            G = ex.grid(imgs, cfg)

            for g, i in zip(G, ii):
                fg = foreground(g)
                h = int(y_face[i])                        # the TRUE head end, from motion
                t = opposite_slot(face_ids[i], h)
                prof, cnt = axis_profile(g, fg, face_uv[i][t], face_uv[i][h], cfg.bins)
                if not cnt.sum():
                    continue
                sp = str(species[i])
                if sp not in acc:
                    acc[sp] = np.zeros_like(prof, dtype=np.float64)
                    wts[sp] = np.zeros(cfg.bins, dtype=np.float64)
                    n[sp] = 0
                acc[sp] += prof * cnt[:, None]            # weight by the evidence in each bin
                wts[sp] += cnt
                n[sp] += 1

        templates = {}
        for sp, a in acc.items():
            w = np.maximum(wts[sp], 1.0)[:, None]
            m = a / w
            nrm = np.linalg.norm(m, axis=1, keepdims=True)
            templates[sp] = (m / np.maximum(nrm, 1e-9)).astype(np.float32)
        return cls(cfg=cfg, templates=templates, n_fitted=n)

    # ---- scoring ----
    def score_faces(self, g: np.ndarray, fg: np.ndarray, face_uv: np.ndarray,
                    face_ids: np.ndarray, species: str) -> np.ndarray:
        """(4,) score for "the head is at face slot j". Higher is better; NaN if unusable."""
        T = self.templates.get(str(species))
        if T is None:
            return np.full(4, np.nan)

        out = np.full(4, np.nan)
        done: set[int] = set()
        for j in range(4):
            if j in done:
                continue
            t = opposite_slot(face_ids, j)
            prof, cnt = axis_profile(g, fg, face_uv[t], face_uv[j], self.cfg.bins)
            if not cnt.sum():
                continue
            # The opposite hypothesis is the SAME profile, read backwards -- so one computation
            # answers both ends of this axis.
            out[j] = _match(prof, cnt, T)
            out[t] = _match(prof[::-1], cnt[::-1], T)
            done.update({j, t})
        return out

    def predict_face(self, *args, **kw) -> tuple[int, float]:
        """Best front-face slot, and the margin over the runner-up (a natural confidence).

        The margin is what a temporal decoder consumes: a frame where the animal is head-on gives
        a near-zero margin, and that is honest -- the cue genuinely is not visible there.
        """
        s = self.score_faces(*args, **kw)
        if np.isnan(s).all():
            return -1, 0.0
        order = np.argsort(np.where(np.isnan(s), -np.inf, s))[::-1]
        best = int(order[0])
        second = s[order[1]] if len(order) > 1 and np.isfinite(s[order[1]]) else s[best]
        return best, float(s[best] - second)


def _match(prof: np.ndarray, cnt: np.ndarray, template: np.ndarray) -> float:
    """Evidence-weighted cosine between a crop's profile and the species template.

    Empty bins contribute nothing rather than contributing noise -- a slice of the animal that is
    occluded or off-frame should not vote.
    """
    w = cnt.astype(np.float64)
    tot = w.sum()
    if tot < 1:
        return float("nan")
    sim = np.einsum("bd,bd->b", prof, template)
    return float((sim * w).sum() / tot)
