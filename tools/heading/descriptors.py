"""Dense DINOv3 features. FEATURES ONLY -- no experiment logic, no prototypes, no scoring.

    from tools.heading.descriptors import Config, DenseExtractor, foreground, axis_profile

The method that consumes these lives in `template.py`; the experiments live in `sweep.py`,
`tracklet.py` and `evaluate.py`. Keeping the feature layer free of experiment code is what lets
all three score the *same* descriptors without copy-paste drifting between them.

WHAT WE GET FROM DINOv3, AND WHAT WE USED TO IGNORE
---------------------------------------------------
DINOv3's headline contribution (Gram anchoring) exists to make DENSE features good. Its dense
correspondence matches head-to-head and rump-to-rump across instances -- and it is mirror-blind
(a left flank and a right flank are near-mirror images), which is fine: we only ever ask
appearance for FRONT vs BACK. Left/right comes from geometry (`left = up x forward`).

The first attempt used the CLS token and a mean-pool of the last layer -- close to the least
dense-aware way to use it. This module exposes the axes that actually matter:

  layer       part semantics usually peak MID-network; the last block is global
  facet       token output vs KEY. The standard result (Amir et al., "Deep ViT Features as Dense
              Visual Descriptors") is that the `key` facet wins for dense correspondence.
  resolution  DINOv3 uses RoPE, so it takes any size natively. 448 -> a 28x28 grid, 4x the
              spatial granularity of the 224 we started with -- and localising a head is exactly
              a granularity problem.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"

__all__ = ["Config", "DenseExtractor", "foreground", "axis_profile", "DEFAULT_MODEL"]


@dataclass(frozen=True)
class Config:
    layer: int = -1         # 1-indexed transformer block; -1 = last
    facet: str = "key"      # "token" | "key"
    size: int = 448         # input resolution fed to DINOv3 (RoPE => anything)
    bins: int = 5           # slices along the body axis

    def __str__(self) -> str:
        return f"L{self.layer:>2}/{self.facet:<5}/{self.size}/b{self.bins}"


class DenseExtractor:
    """Frozen DINOv3 -> an L2-normalised [gh, gw, D] descriptor grid, at any layer, token or key.

    Mirrors `get_intermediate_layers(..., reshape=True, norm=True)`: take the hidden state after
    the requested block, apply the model's final LayerNorm, drop CLS + register tokens, reshape.

    Registers: DINOv3 lays out [CLS, registers..., patches] and the register count varies by
    checkpoint, so we take the LAST gh*gw tokens rather than assuming an offset. Guessing it
    shifts the grid by a row and silently ruins every 2D lookup.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cuda"):
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.device = device
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.patch = int(self.model.config.patch_size)
        self.n_layers = int(self.model.config.num_hidden_layers)
        self.dim = int(self.model.config.hidden_size)

        # We bypass AutoImageProcessor so the resolution is ours to choose.
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        # `norm=True` in get_intermediate_layers applies the model's FINAL LayerNorm. The
        # attribute name moves between HF versions, so resolve it once and fail loudly --
        # silently skipping it would change every descriptor while still producing a plausible
        # sweep table.
        self._final_norm = next(
            (getattr(self.model, a) for a in ("layernorm", "norm", "final_layernorm")
             if hasattr(self.model, a)), None)
        if self._final_norm is None:
            raise RuntimeError("no final LayerNorm on the DINOv3 model (tried layernorm/norm/"
                               "final_layernorm). Wire it up rather than skipping it.")

        self._keys: dict[int, object] = {}
        self._hooks: list = []
        self._hook_keys()

    def _hook_keys(self) -> None:
        """Capture the KEY projection of every attention block.

        Located by MODULE NAME rather than a guessed attribute path: the nesting differs across HF
        versions, and a wrong guess would silently fall back to token features while the sweep
        table still printed "key" -- i.e. it would fake the very comparison being run.
        """
        import re

        pat = re.compile(r"(?:^|\.)layers?\.(\d+)\..*\b(?:key|k_proj)$")

        def mk(i):
            def hook(_m, _inp, out):
                self._keys[i] = out.detach()
            return hook

        for name, mod in self.model.named_modules():
            m = pat.search(name)
            if m and hasattr(mod, "weight"):
                self._hooks.append(mod.register_forward_hook(mk(int(m.group(1)))))

        if not self._hooks:
            seen = [n for n, _ in self.model.named_modules() if "attention" in n][:12]
            raise RuntimeError(
                "could not hook any key projection. Fix the pattern rather than falling back to "
                f"token features -- that would fake the token-vs-key comparison. Saw: {seen}")

    def grid(self, imgs: np.ndarray, cfg: Config) -> np.ndarray:
        """(B, H, W, 3) uint8 -> (B, gh, gw, D) float32, L2-normalised."""
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(imgs)).to(self.device)
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        x = torch.nn.functional.interpolate(x, size=(cfg.size, cfg.size),
                                            mode="bicubic", align_corners=False)
        x = (x - self.mean) / self.std

        self._keys.clear()
        with torch.no_grad():
            out = self.model(pixel_values=x, output_hidden_states=True)

        li = self.n_layers if cfg.layer < 0 else cfg.layer      # 1-indexed block
        gh = gw = cfg.size // self.patch
        n = gh * gw

        if cfg.facet == "key":
            h = self._keys[li - 1]                              # hook index is 0-based
            if h.dim() == 4:                                    # (B, heads, T, dh) -> (B, T, D)
                h = h.permute(0, 2, 1, 3).reshape(h.shape[0], h.shape[2], -1)
        else:
            h = self._final_norm(out.hidden_states[li])         # [0] is the embedding, not a block

        h = torch.nn.functional.normalize(h[:, -n:].float(), dim=-1)
        return h.reshape(-1, gh, gw, h.shape[-1]).cpu().numpy()

    def close(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def foreground(g: np.ndarray, *, core: float = 0.22, rim: float = 0.15) -> np.ndarray:
    """(gh, gw, D) -> boolean foreground mask, from CENTRE-vs-CORNER prototypes.

    Two things are true by construction, for free:
      * the crop was cut from the ANIMAL'S OWN 2D box, so its CENTRE is animal;
      * after square-padding, the CORNERS are grass (or black padding).

    So: mean descriptor of the central patches = "animal", mean of the corners = "background", and
    each patch goes to whichever it is closer to. Two prototypes, no model, nothing to tune.

    This replaces a per-crop PC1 threshold, which was mediocre for a structural reason: PC1's SIGN
    is arbitrary and its THRESHOLD is arbitrary, so the mask could invert or drift from crop to
    crop. Here both are pinned by facts we already know instead of estimated from the data.
    """
    gh, gw, D = g.shape
    r0, r1 = int(gh * (0.5 - core / 2)), int(np.ceil(gh * (0.5 + core / 2)))
    c0, c1 = int(gw * (0.5 - core / 2)), int(np.ceil(gw * (0.5 + core / 2)))
    kr, kc = max(1, int(gh * rim)), max(1, int(gw * rim))

    fg_p = g[r0:r1, c0:c1].reshape(-1, D).mean(0)
    bg_p = np.concatenate([g[:kr, :kc].reshape(-1, D), g[:kr, -kc:].reshape(-1, D),
                           g[-kr:, :kc].reshape(-1, D), g[-kr:, -kc:].reshape(-1, D)]).mean(0)

    nf, nb = np.linalg.norm(fg_p), np.linalg.norm(bg_p)
    if nf < 1e-9 or nb < 1e-9:
        return np.ones((gh, gw), dtype=bool)

    m = (g @ (fg_p / nf)) > (g @ (bg_p / nb))            # g is L2-normalised => this is cosine
    return m if m.sum() >= 4 else np.ones((gh, gw), dtype=bool)


def axis_profile(g: np.ndarray, fg: np.ndarray, tail_uv, head_uv, bins: int):
    """Mean descriptor in each of `bins` slices ALONG THE BODY AXIS -- the anatomical template.

    Every foreground patch is projected onto the image-space axis (tail_uv -> head_uv) to get a
    coordinate s in [0,1], then binned. The (bins, D) result reads rump -> torso -> neck -> head.

    WHY A PROFILE AND NOT A POINT
    -----------------------------
    The first version sampled the descriptor AT the projected front-face centre. But that is the
    centre of the box's front FACE -- mid-height on the cuboid's front plane -- and an animal's
    head is not there. It is higher, and often outside that plane entirely (a giraffe's head is
    metres above the box centreline; an elephant's trunk juts forward). So the prototype averaged
    descriptors taken from the chest, the neck, or thin air, and came out as mush -- which is
    exactly why the head-similarity map was uninformative.

    Pooling a whole slice is immune to that misplacement, and it captures the head->rump GRADIENT
    rather than two endpoints.

    Returns (bins, D) L2-normalised rows and a (bins,) patch count. An empty bin is a zero row and
    a zero count, so callers weight by evidence instead of averaging noise.

    NOTE the profile for the OPPOSITE hypothesis is exactly this one reversed -- callers exploit
    that rather than recomputing (see template.AxisTemplate.score_faces).
    """
    gh, gw, D = g.shape
    ys, xs = np.nonzero(fg)
    empty = (np.zeros((bins, D), dtype=np.float32), np.zeros(bins, dtype=np.int32))
    if len(ys) < bins:
        return empty

    a = np.asarray(head_uv, dtype=np.float64) - np.asarray(tail_uv, dtype=np.float64)
    L = float(np.linalg.norm(a))
    if L < 1e-6:                    # animal exactly end-on: the axis has no extent in the image
        return empty

    P = np.stack([(xs + 0.5) / gw, (ys + 0.5) / gh], axis=1)     # same [0,1]^2 coords as face_uv
    s = (P - np.asarray(tail_uv, dtype=np.float64)) @ (a / L) / L   # 0 at tail, 1 at head
    idx = (np.clip(s, 0.0, 1.0 - 1e-9) * bins).astype(int)

    F = g[ys, xs]
    out = np.zeros((bins, D), dtype=np.float32)
    cnt = np.zeros(bins, dtype=np.int32)
    for b in range(bins):
        m = idx == b
        cnt[b] = int(m.sum())
        if cnt[b]:
            v = F[m].mean(0)
            n = np.linalg.norm(v)
            if n > 1e-9:
                out[b] = v / n
    return out, cnt
