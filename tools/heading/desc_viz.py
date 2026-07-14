"""SEE the dense features: foreground PCA-RGB + the head/tail similarity maps.  [GPU / cluster]

    python -m tools.heading.desc_viz --crops data/heading/crops.npz \
        --out data/heading/desc --layer 18 --facet key --size 448 --device cuda

WHY THIS EXISTS
---------------
The last part probe returned a number (16.6%) and I declared DINOv3 parts dead. The number was
right and the conclusion was wrong: the probe tested a strawman. What would have caught that
instantly is a PICTURE -- the user looked at `parts_viz.jpg`, saw the "head" cluster covering the
whole animal, and said so in one line.

So before any sweep table is believed, look at the pixels.

WHAT IS DRAWN, per crop (3 panels)
----------------------------------
  1. the crop, with the 4 candidate face centres marked; the TRUE head (from motion) in GREEN
  2. FOREGROUND PCA-RGB -- the classic DINO part visualisation: PCA of the *foreground-only*
     patch descriptors, top-3 components -> RGB. If body parts are separable at all, distinct
     regions (head / torso / legs) take distinct colours HERE. Doing the PCA on foreground only
     is the step the old probe skipped, and it is the step that matters: on all patches, the
     dominant variance is object-vs-grass and parts never surface.
  3. HEAD-similarity map -- cosine between every patch and the species HEAD prototype. If DINOv3
     carries head/tail, this lights up on the head and nowhere else.

If panel 2 is a single flat colour over the animal, there are no parts to find and no metric will
rescue it. If panel 3 lights up the rump as brightly as the head, correspondence is mirror-blind
along the body axis and the whole approach fails -- and we would want to know that from a picture
rather than from a confusing number.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np

from tools.heading.descriptors import Config, DenseExtractor, foreground_pc1, sample_at

DEFAULT_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def pca_rgb(g: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """(gh, gw, D) + foreground mask -> (gh, gw, 3) RGB, PCA fitted on FOREGROUND ONLY."""
    gh, gw, D = g.shape
    X = g.reshape(-1, D)
    m = fg.reshape(-1)
    if m.sum() < 4:
        return np.zeros((gh, gw, 3), dtype=np.float32)

    Xf = X[m]
    mu = Xf.mean(0, keepdims=True)
    # top-3 right singular vectors of the FOREGROUND patches
    _, _, Vt = np.linalg.svd(Xf - mu, full_matrices=False)
    P = (X - mu) @ Vt[:3].T
    P = P / (P.std(0, keepdims=True) + 1e-9)            # whiten, as in the reference recipe
    rgb = 1.0 / (1.0 + np.exp(-2.0 * P))                # sigmoid(2x) -> vibrant colours
    rgb = rgb.reshape(gh, gw, 3)
    rgb *= fg[..., None]                                # black out the background
    return rgb.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--facet", default="key", choices=["token", "key"])
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--radius", type=int, default=1)
    ap.add_argument("--per-species", type=int, default=5)
    ap.add_argument("--proto-n", type=int, default=300, help="crops used to build the prototypes")
    ap.add_argument("--cell", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from PIL import Image, ImageDraw

    from tools.heading.conventions import OPPOSITE_FACE
    from tools.heading.split import video_split

    d = np.load(args.crops, allow_pickle=True)
    jpeg, face_uv, y_face = d["jpeg"], d["face_uv"], d["y_face"]
    face_ids, sp, vid = d["face_ids"], d["species"], d["video"]

    te, _ = video_split(sp, vid, seed=args.seed)
    cfg = Config(args.layer, args.facet, args.size, True, args.radius)
    ex = DenseExtractor(args.model, args.device)
    print(f"{cfg}  |  {ex.n_layers} layers, patch {ex.patch}")

    def decode(i):
        return np.asarray(Image.open(io.BytesIO(jpeg[i])).convert("RGB"))

    def opp(i, j):
        hit = np.where(face_ids[i] == OPPOSITE_FACE[int(face_ids[i][j])])[0]
        return int(hit[0]) if len(hit) else j

    # ---- head/tail prototypes from CALIBRATION videos (never the ones we visualise) ----
    rng = np.random.default_rng(args.seed)
    proto: dict[tuple[str, str], np.ndarray] = {}
    acc: dict[tuple[str, str], list] = {}
    cal = np.where(~te)[0]
    for i in rng.permutation(cal)[: args.proto_n]:
        i = int(i)
        g = ex.grid(decode(i)[None], cfg)[0]
        fg = foreground_pc1(g)
        h = int(y_face[i])
        acc.setdefault((str(sp[i]), "head"), []).append(sample_at(g, face_uv[i][h], cfg.radius, fg))
        acc.setdefault((str(sp[i]), "tail"), []).append(
            sample_at(g, face_uv[i][opp(i, h)], cfg.radius, fg))
    for k, v in acc.items():
        m = np.mean(v, axis=0)
        proto[k] = m / (np.linalg.norm(m) + 1e-9)
    print(f"prototypes from {args.proto_n} calibration crops: {sorted({k[0] for k in proto})}")

    # ---- render HELD-OUT crops ----
    C = args.cell
    species = sorted(set(sp.tolist()))
    picks: list[int] = []
    for s in species:
        pool = np.where(te & (sp == s))[0]
        if len(pool):
            picks.extend(rng.permutation(pool)[: args.per_species].tolist())

    sheet = Image.new("RGB", (3 * C, len(picks) * C), (14, 14, 16))
    draw = ImageDraw.Draw(sheet)

    for row, i in enumerate(picks):
        i = int(i)
        img = decode(i)
        g = ex.grid(img[None], cfg)[0]
        fg = foreground_pc1(g)
        y0 = row * C

        # panel 1 -- the crop + the 4 candidates, true head in GREEN
        base = Image.fromarray(img).resize((C, C))
        sheet.paste(base, (0, y0))
        for j in range(4):
            u, v = face_uv[i][j] * C
            col = (40, 255, 90) if j == int(y_face[i]) else (150, 150, 150)
            r = 6 if j == int(y_face[i]) else 4
            draw.ellipse([u - r, y0 + v - r, u + r, y0 + v + r], fill=col)
        draw.text((4, y0 + 4), f"{sp[i]}", fill=(230, 230, 230))

        # panel 2 -- FOREGROUND PCA-RGB (the classic part visualisation)
        rgb = (255 * pca_rgb(g, fg)).astype(np.uint8)
        sheet.paste(Image.fromarray(rgb).resize((C, C), Image.NEAREST), (C, y0))
        draw.text((C + 4, y0 + 4), "fg PCA-RGB", fill=(230, 230, 230))

        # panel 3 -- similarity to the species HEAD prototype
        key = (str(sp[i]), "head")
        if key in proto:
            s = (g @ proto[key]) * fg                    # cosine; background zeroed
            s = (s - s.min()) / (s.max() - s.min() + 1e-9)
            hm = np.stack([s, s * 0.35, 1.0 - s], -1)    # blue -> red
            hm = (255 * hm * fg[..., None]).astype(np.uint8)
            sheet.paste(Image.fromarray(hm).resize((C, C), Image.NEAREST), (2 * C, y0))
            draw.text((2 * C + 4, y0 + 4), "HEAD similarity", fill=(230, 230, 230))

    ex.close()
    args.out.mkdir(parents=True, exist_ok=True)
    p = args.out / f"desc_L{args.layer}_{args.facet}_{args.size}.jpg"
    sheet.save(p, quality=92)
    print(f"\nwrote {p}")
    print("  panel 1: crop + 4 candidate face centres (GREEN = true head, from motion)")
    print("  panel 2: foreground PCA-RGB -- if parts exist, head/torso/legs take DIFFERENT colours")
    print("  panel 3: cosine to the species HEAD prototype -- should light the head, NOT the rump")
    print("\n  If panel 2 is one flat colour, there are no parts and no metric will rescue it.")
    print("  If panel 3 lights the rump as brightly as the head, correspondence is blind along")
    print("  the body axis -- and the approach fails. Better to see that than to argue with a number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
