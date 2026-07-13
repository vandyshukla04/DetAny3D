"""The heading head — defined ONCE, imported everywhere.

Training and inference used to build this MLP separately, by hand. The moment the two drift
(e.g. adding a layer in training), `load_state_dict` breaks -- or worse, silently loads a
mismatched net. So the architecture lives here and nowhere else.

Input : frozen DINOv3 features  [CLS ‖ mean-pooled patch tokens]  (2048-d for ViT-L/16)
Output: (cos θ, sin θ) — the animal's heading as a point on the unit circle, IN IMAGE SPACE.

Why the unit circle and not a raw angle: an angle is discontinuous at ±π (359° and 1° are
neighbours but numerically far apart), so regressing it directly puts a cliff in the middle of
the target space. (cos, sin) is smooth everywhere, and a cosine loss on it is exactly angular
error.
"""
from __future__ import annotations

__all__ = ["build_head", "HEAD_VERSION"]

HEAD_VERSION = 2        # bump when the architecture changes; checkpoints record it


def build_head(in_dim: int, hidden: int = 512):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.2),
        nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.2),
        nn.Linear(hidden, 2),
    )
