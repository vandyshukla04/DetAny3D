"""Dense DINOv3 descriptors, and the sweep that decides how to use them.  [GPU / cluster]

    python -m tools.heading.descriptors --crops data/heading/crops.npz \
        --out data/heading/desc --device cuda

THE QUESTION
------------
Geometry gives the animal's body AXIS (from the 3D box) but never its SIGN -- the box's PCA/SVD
axis signs are arbitrary. Locomotion gives the sign, but only while the animal walks (16% of
frames). So the one thing we need from appearance is:

    of the box's 4 horizontal faces, which one is the HEAD end?

DINOv3 is claimed to match head-to-head and rump-to-rump across instances -- dense semantic
correspondence is its headline property (Gram anchoring exists to make exactly this work). It is
mirror-blind (a left flank and a right flank are near-mirror images), but we never wanted
left/right from appearance: that is pure geometry (`left = up x forward`).

WHY WE DO NOT SEARCH THE HEATMAP
--------------------------------
The reference recipe argmaxes over the whole patch grid to find a correspondence. We do not need
to. The 3D box already tells us where the 4 candidate ends ARE: `crops.npz` stores `face_uv`, the
4 horizontal face centres projected into crop coordinates. So the task is a scored 4-way
comparison at geometrically-fixed points -- far better conditioned than an open search, with no
clustering and no foreground argmax.

Scoring is free: on WALKING crops, motion already told us which face is the head (`y_face`).

WHY THE EARLIER PROBE (parts.py, 16.6%) DOES NOT SETTLE THIS
-------------------------------------------------------------
It tested a strawman: global k-means over every species at once, a 14x14 grid, the LAST layer
only, and cluster centroids instead of correspondence. Every one of those is wrong for dense part
discovery. This module fixes all four and sweeps them:

  layer       1..N   -- part semantics usually peak MID-network; the last block is global
  facet       token output vs KEY -- the standard result (Amir et al., "Deep ViT Features as
              Dense Visual Descriptors") is that the `key` facet wins for correspondence
  resolution  224 (14x14) vs 448 (28x28) -- DINOv3 uses RoPE, so it takes any size natively.
              Upsampling the existing crops is free: we gain SPATIAL GRANULARITY, which is what
              localising a head actually needs.
  foreground  all patches vs PC1-thresholded foreground (the standard DINO trick, no extra model)
  pooling     the single patch at a face centre vs a small neighbourhood (robustness)
"""
from __future__ import annotations

import argparse
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"

__all__ = ["DenseExtractor", "foreground_pc1", "sample_at", "Config"]


@dataclass(frozen=True)
class Config:
    layer: int          # 1-indexed transformer block; -1 = last
    facet: str          # "token" | "key"
    size: int           # input resolution fed to DINOv3 (RoPE => anything)
    foreground: bool    # restrict to PC1-thresholded foreground
    radius: int         # 0 = the single patch at the face centre; 1 = its 3x3 neighbourhood

    def __str__(self) -> str:
        return (f"L{self.layer:>2}/{self.facet:<5}/{self.size}"
                f"/{'fg ' if self.foreground else 'all'}/r{self.radius}")


class DenseExtractor:
    """Frozen DINOv3 -> a [D, gh, gw] descriptor grid, at any layer, token or key facet.

    Mirrors `get_intermediate_layers(..., reshape=True, norm=True)`: we take the hidden state
    AFTER the requested block, apply the model's final LayerNorm, drop CLS+register tokens, and
    reshape to a spatial grid.

    Register tokens: DINOv3 lays out [CLS, registers..., patches] and the register count varies
    by checkpoint. We take the LAST gh*gw tokens rather than assuming an offset -- guessing it
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

        # ImageNet stats -- we bypass AutoImageProcessor so we control the resolution exactly.
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        # `get_intermediate_layers(..., norm=True)` applies the model's FINAL LayerNorm to the
        # intermediate hidden state. The attribute name differs across HF versions, so resolve it
        # once and fail loudly -- silently skipping the norm would change every descriptor while
        # still producing a plausible sweep table.
        self._final_norm = next(
            (getattr(self.model, a) for a in ("layernorm", "norm", "final_layernorm")
             if hasattr(self.model, a)), None)
        if self._final_norm is None:
            raise RuntimeError(
                "no final LayerNorm found on the DINOv3 model (tried layernorm/norm/"
                "final_layernorm). Find it and wire it up rather than skipping it."
            )

        self._keys: dict[int, "torch.Tensor"] = {}
        self._hooks = []
        self._install_key_hooks()

    def _install_key_hooks(self):
        """Capture the KEY projection of every attention block.

        Amir et al. showed the `key` facet is the strongest dense descriptor in a DINO ViT --
        better than the block's output tokens, which is all we used before. HF exposes no direct
        access, so we hook the key Linear inside each attention module.

        We locate them by MODULE NAME (`...layer.<i>....key|k_proj`) rather than by walking a
        guessed attribute path, because the exact nesting differs across HF versions and a wrong
        guess would silently fall back to token features while still printing "key" in the sweep
        table -- i.e. it would fake the very comparison we are running.
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
            names = [n for n, _ in self.model.named_modules() if "attention" in n][:12]
            raise RuntimeError(
                "could not hook any key projection. The HF DINOv3 attention layout is not what "
                "this code expects -- fix the pattern rather than silently falling back to token "
                f"features (that would fake the token-vs-key comparison). Saw: {names}"
            )

    def grid(self, imgs: np.ndarray, cfg: Config):
        """(B, H, W, 3) uint8 -> (B, gh, gw, D) float32 descriptors, L2-normalised."""
        torch = self.torch
        x = torch.from_numpy(imgs).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        x = torch.nn.functional.interpolate(x, size=(cfg.size, cfg.size),
                                            mode="bicubic", align_corners=False)
        x = (x - self.mean) / self.std

        self._keys.clear()
        with torch.no_grad():
            out = self.model(pixel_values=x, output_hidden_states=True)

        li = self.n_layers if cfg.layer < 0 else cfg.layer     # 1-indexed block
        gh = gw = cfg.size // self.patch
        n = gh * gw

        if cfg.facet == "key":
            h = self._keys[li - 1]                             # hook index is 0-based
            if h.dim() == 4:                                   # (B, heads, T, dh) -> (B, T, D)
                h = h.permute(0, 2, 1, 3).reshape(h.shape[0], h.shape[2], -1)
        else:
            h = out.hidden_states[li]                          # hidden_states[0] is the embedding
            h = self._final_norm(h)                            # == get_intermediate_layers(norm=True)

        h = h[:, -n:]                                          # the LAST n tokens are the grid
        h = torch.nn.functional.normalize(h.float(), dim=-1)
        return h.reshape(-1, gh, gw, h.shape[-1]).cpu().numpy()

    def close(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def foreground_pc1(g: np.ndarray) -> np.ndarray:
    """(gh, gw, D) -> boolean foreground mask via the first principal component.

    The standard DINO trick: PC1 of the patch tokens separates object from background almost
    perfectly. We orient it so the CENTRE of the crop is foreground -- the animal is centred by
    construction (the crop came from its own 2D box), so the centre is the one thing we know.
    Without that orientation step the sign of PC1 is arbitrary and the mask inverts at random.
    """
    gh, gw, _ = g.shape
    X = g.reshape(-1, g.shape[-1])
    X = X - X.mean(0, keepdims=True)
    # PC1 by power iteration -- cheaper than a full SVD and we only need one component.
    v = X[np.argmax(np.linalg.norm(X, axis=1))]
    for _ in range(12):
        v = X.T @ (X @ v)
        nv = np.linalg.norm(v)
        if nv < 1e-9:
            return np.ones((gh, gw), dtype=bool)
        v /= nv
    p = (X @ v).reshape(gh, gw)

    cy, cx = gh // 2, gw // 2
    core = p[max(cy - 1, 0): cy + 2, max(cx - 1, 0): cx + 2].mean()
    if core < p.mean():                                  # make the crop centre the positive side
        p = -p
    return p > p.mean()


def sample_at(g: np.ndarray, uv: np.ndarray, radius: int, fg: np.ndarray | None) -> np.ndarray:
    """Descriptor at a crop-space point (u, v) in [0,1]^2, averaged over a (2r+1)^2 neighbourhood.

    Foreground-masked patches are dropped from the average; if none survive, we fall back to the
    plain neighbourhood rather than returning garbage -- a face centre can legitimately fall just
    off the animal when the box is loose.
    """
    gh, gw, _ = g.shape
    c = int(np.clip(uv[0] * gw, 0, gw - 1))
    r = int(np.clip(uv[1] * gh, 0, gh - 1))
    r0, r1 = max(r - radius, 0), min(r + radius + 1, gh)
    c0, c1 = max(c - radius, 0), min(c + radius + 1, gw)

    patch = g[r0:r1, c0:c1].reshape(-1, g.shape[-1])
    if fg is not None:
        m = fg[r0:r1, c0:c1].reshape(-1)
        if m.any():
            patch = patch[m]
    v = patch.mean(0)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


# ---------------------------------------------------------------------------------------------
def run_config(ex: DenseExtractor, crops, idx_cal, idx_te, cfg: Config, batch: int = 32):
    """Build HEAD and TAIL prototypes on the calibration videos; score the 4-way choice on the
    held-out ones.

    The prototypes are just the mean descriptor at the true head (and true tail) face centre --
    read straight off the motion labels. Nothing is trained.
    """
    from PIL import Image

    from tools.heading.conventions import OPPOSITE_FACE

    jpeg, face_uv, y_face = crops["jpeg"], crops["face_uv"], crops["y_face"]
    face_ids, species = crops["face_ids"], crops["species"]

    def decode(ii):
        return np.stack([np.asarray(Image.open(io.BytesIO(jpeg[i])).convert("RGB")) for i in ii])

    def opposite_slot(i: int, j: int) -> int:
        """Slot (0..3) of the face diametrically opposite slot j. Uses the real face IDs from
        crops.npz -- NOT the distance between projected centres, which degenerates when the
        animal is viewed end-on and the two centres nearly coincide."""
        want = OPPOSITE_FACE[int(face_ids[i][j])]
        hit = np.where(face_ids[i] == want)[0]
        return int(hit[0]) if len(hit) else j

    # ---- pass 1: per-species HEAD and TAIL prototypes, from CALIBRATION videos only ----
    # Nothing is trained. The prototype is literally the mean descriptor at the face centre that
    # motion already told us is the head (and at its opposite, the rump).
    acc: dict[tuple[str, str], list[np.ndarray]] = {}
    for b in range(0, len(idx_cal), batch):
        ii = idx_cal[b: b + batch]
        G = ex.grid(decode(ii), cfg)
        for k, i in enumerate(ii):
            g = G[k]
            fg = foreground_pc1(g) if cfg.foreground else None
            h = int(y_face[i])
            t = opposite_slot(i, h)
            sp = str(species[i])
            acc.setdefault((sp, "head"), []).append(sample_at(g, face_uv[i][h], cfg.radius, fg))
            acc.setdefault((sp, "tail"), []).append(sample_at(g, face_uv[i][t], cfg.radius, fg))

    proto = {}
    for k, v in acc.items():
        m = np.mean(v, axis=0)
        proto[k] = m / (np.linalg.norm(m) + 1e-9)

    # ---- pass 2: score the 4-way choice on HELD-OUT videos ----
    ok, tot = {}, {}
    for b in range(0, len(idx_te), batch):
        ii = idx_te[b: b + batch]
        G = ex.grid(decode(ii), cfg)
        for k, i in enumerate(ii):
            sp = str(species[i])
            if (sp, "head") not in proto:
                continue
            g = G[k]
            fg = foreground_pc1(g) if cfg.foreground else None
            d = np.stack([sample_at(g, face_uv[i][j], cfg.radius, fg) for j in range(4)])
            # Score candidate j as "j looks like a HEAD *and* its opposite looks like a RUMP".
            # Using both prototypes is the point: DINOv3 matches rumps as reliably as heads, so
            # the joint score is strictly more evidence than head-similarity alone.
            opp = [opposite_slot(i, j) for j in range(4)]
            s = d @ proto[(sp, "head")] + d[opp] @ proto[(sp, "tail")]
            ok[sp] = ok.get(sp, 0) + int(int(np.argmax(s)) == int(y_face[i]))
            tot[sp] = tot.get(sp, 0) + 1
    return ok, tot


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--subset", type=int, default=2000, help="crops per split for the sweep")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from tools.heading.split import video_split

    crops = np.load(args.crops, allow_pickle=True)
    sp, vid = crops["species"], crops["video"]
    te, test_v = video_split(sp, vid, seed=args.seed)
    print(f"{len(sp)} crops | held-out {len(test_v)} videos ({te.sum()} crops)")

    rng = np.random.default_rng(args.seed)
    idx_cal = np.sort(rng.permutation(np.where(~te)[0])[: args.subset])
    idx_te = np.sort(rng.permutation(np.where(te)[0])[: args.subset])
    print(f"sweep on {len(idx_cal)} calibration / {len(idx_te)} held-out crops\n")

    ex = DenseExtractor(args.model, args.device)
    print(f"{ex.n_layers} layers, {ex.dim}-d, patch {ex.patch}\n")

    # Sweep the axes the old probe never touched. Layers are coarse-sampled first; the winner
    # gets refined. Everything is scored on the SAME 4-way metric as the 79.8% MLP baseline.
    layers = sorted({ex.n_layers // 2, (3 * ex.n_layers) // 4, ex.n_layers - 4, ex.n_layers})
    configs = [Config(l, f, s, fg, r)
               for l in layers
               for f in ("token", "key")
               for s in (224, 448)
               for fg in (True,)
               for r in (1,)]
    configs += [Config(layers[1], "key", 448, False, 1),      # foreground ablation
                Config(layers[1], "key", 448, True, 0)]       # neighbourhood ablation

    rows = []
    print(f"{'config':<28s} {'ALL':>7s}  " +
          "  ".join(f"{s[:5]:>6s}" for s in sorted(set(sp.tolist()))))
    for cfg in configs:
        ok, tot = run_config(ex, crops, idx_cal, idx_te, cfg, args.batch)
        n = sum(tot.values())
        if not n:
            continue
        overall = sum(ok.values()) / n
        rows.append((str(cfg), overall, {k: ok[k] / tot[k] for k in tot}))
        per = "  ".join(f"{100*ok[s]/tot[s]:5.1f}%" if s in tot else "     -"
                        for s in sorted(set(sp.tolist())))
        print(f"{str(cfg):<28s} {100*overall:6.1f}%  {per}", flush=True)
    ex.close()

    if not rows:
        print("nothing scored")
        return 1

    rows.sort(key=lambda r: -r[1])
    best = rows[0]
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "sweep.npz",
             config=np.array([r[0] for r in rows]),
             overall=np.array([r[1] for r in rows]))

    print(f"\n  BEST: {best[0]}  ->  {100*best[1]:.1f}%   (chance 25%)")
    print(f"  MLP baseline (pooled features, supervised): 79.8%")
    print(f"  naive k-means probe (retracted strawman)  : 16.6%")
    if best[1] > 0.60:
        print("\n  => DINOv3 dense correspondence DOES carry head/tail. Take it to X2 (standing animals).")
    elif best[1] > 0.35:
        print("\n  => a real but weak signal. Worth combining with the supervised head, not replacing it.")
    else:
        print("\n  => correspondence does not carry head/tail HERE. Say so; the MLP remains the method.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
