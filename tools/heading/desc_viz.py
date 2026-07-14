"""SEE the method: foreground, the axis profile, and the head/tail decision.  [GPU / cluster]

    python -m tools.heading.desc_viz --crops data/heading/crops.npz \
        --out data/heading/desc --layer 18 --facet key --size 448 --device cuda

WHY THIS EXISTS
---------------
Twice now a picture has caught what a number could not. The k-means probe returned 16.6% and I
called DINOv3 dead -- but the picture showed the "head" cluster covering the WHOLE ANIMAL, which
is what actually explained it. Then the head-similarity map "helped understand nothing", and that
told us the prototype (a point sample at the box's front-face centre, which is not where an
animal's head is) was the broken part, not the features.

So: look at the pixels before believing the table.

FOUR PANELS PER CROP
--------------------
  1. the crop + the 4 candidate face centres. GREEN = the true head end (from motion).
     The PREDICTED head is ringed. A red border means we got it wrong.
  2. FOREGROUND -- the centre-vs-corner mask. If this is ragged, everything downstream suffers.
  3. foreground PCA-RGB -- the classic part visualisation. This is the panel that already told us
     parts ARE present ("you can clearly see it discerning the animal features").
  4. the AXIS PROFILE -- foreground patches binned rump->head, coloured by the same PCA. If the
     anatomy has a gradient along the body, the bins differ; if it does not, they are one colour
     and no head/tail cue exists to find.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np

from tools.heading.descriptors import DEFAULT_MODEL, Config, DenseExtractor, axis_profile, foreground
from tools.heading.split import video_split
from tools.heading.template import AxisTemplate, opposite_slot


def pca_rgb(g: np.ndarray, fg: np.ndarray):
    """(gh,gw,D) + mask -> (gh,gw,3) RGB, PCA fitted on FOREGROUND ONLY, plus the projector.

    Fitting on the foreground is the step the old probe skipped. Over *all* patches the dominant
    variance is animal-vs-grass, so parts never surface; restricted to the animal, they do.
    """
    gh, gw, D = g.shape
    X, m = g.reshape(-1, D), fg.reshape(-1)
    if m.sum() < 4:
        return np.zeros((gh, gw, 3), np.float32), None
    mu = X[m].mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(X[m] - mu, full_matrices=False)

    def project(F):
        P = (F - mu) @ Vt[:3].T
        P /= (P.std(0, keepdims=True) + 1e-9)                  # whiten
        return 1.0 / (1.0 + np.exp(-2.0 * P))                  # sigmoid(2x) -> vivid

    rgb = project(X).reshape(gh, gw, 3) * fg[..., None]
    return rgb.astype(np.float32), project


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
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--per-species", type=int, default=5)
    ap.add_argument("--fit-n", type=int, default=400)
    ap.add_argument("--cell", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from PIL import Image, ImageDraw

    d = np.load(args.crops, allow_pickle=True)
    jpeg, face_uv, y_face = d["jpeg"], d["face_uv"], d["y_face"]
    face_ids, sp, vid = d["face_ids"], d["species"], d["video"]

    te, _ = video_split(sp, vid, seed=args.seed)
    cfg = Config(args.layer, args.facet, args.size, args.bins)
    rng = np.random.default_rng(args.seed)

    with DenseExtractor(args.model, args.device) as ex:
        print(f"{cfg} | {ex.n_layers} layers, patch {ex.patch}")

        # Template from CALIBRATION videos only -- never the ones we render.
        idx_fit = np.sort(rng.permutation(np.where(~te)[0])[: args.fit_n])
        tmpl = AxisTemplate.fit(ex, d, idx_fit, cfg)
        print(f"template from {len(idx_fit)} calibration crops: {tmpl.n_fitted}")

        picks: list[int] = []
        for s in sorted(set(sp.tolist())):
            pool = np.where(te & (sp == s))[0]
            picks.extend(int(i) for i in rng.permutation(pool)[: args.per_species])

        C = args.cell
        sheet = Image.new("RGB", (4 * C, len(picks) * C), (14, 14, 16))
        draw = ImageDraw.Draw(sheet)
        n_ok = 0

        for row, i in enumerate(picks):
            img = np.asarray(Image.open(io.BytesIO(jpeg[i])).convert("RGB"))
            g = ex.grid(img[None], cfg)[0]
            fg = foreground(g)
            y0 = row * C

            h = int(y_face[i])
            t = opposite_slot(face_ids[i], h)
            pred, margin = tmpl.predict_face(g, fg, face_uv[i], face_ids[i], str(sp[i]))
            ok = pred == h
            n_ok += int(ok)

            # 1 -- crop, candidates, truth (green) and prediction (ringed)
            sheet.paste(Image.fromarray(img).resize((C, C)), (0, y0))
            for j in range(4):
                u, v = face_uv[i][j] * C
                col = (40, 255, 90) if j == h else (150, 150, 150)
                draw.ellipse([u - 5, y0 + v - 5, u + 5, y0 + v + 5], fill=col)
                if j == pred:
                    draw.ellipse([u - 9, y0 + v - 9, u + 9, y0 + v + 9],
                                 outline=(255, 60, 60), width=3)
            if not ok:
                draw.rectangle([0, y0, C - 1, y0 + C - 1], outline=(255, 40, 40), width=4)
            draw.text((4, y0 + 4), f"{sp[i]}  margin {margin:+.3f}", fill=(235, 235, 235))

            # 2 -- the foreground mask
            fgi = (255 * fg.astype(np.uint8))[..., None].repeat(3, -1)
            sheet.paste(Image.fromarray(fgi).resize((C, C), Image.NEAREST), (C, y0))
            draw.text((C + 4, y0 + 4), "foreground", fill=(235, 235, 235))

            # 3 -- foreground PCA-RGB (the panel that showed parts ARE there)
            rgb, project = pca_rgb(g, fg)
            sheet.paste(Image.fromarray((255 * rgb).astype(np.uint8)).resize((C, C), Image.NEAREST),
                        (2 * C, y0))
            draw.text((2 * C + 4, y0 + 4), "fg PCA-RGB", fill=(235, 235, 235))

            # 4 -- the AXIS PROFILE: bins from rump (left) to head (right), in the same PCA colours
            prof, cnt = axis_profile(g, fg, face_uv[i][t], face_uv[i][h], cfg.bins)
            if project is not None and cnt.sum():
                cols = (255 * project(prof)).astype(np.uint8)
                bw = C // cfg.bins
                for b in range(cfg.bins):
                    x0 = 3 * C + b * bw
                    fill = tuple(int(v) for v in cols[b]) if cnt[b] else (30, 30, 34)
                    draw.rectangle([x0, y0, x0 + bw - 2, y0 + C - 20], fill=fill)
                    draw.text((x0 + 3, y0 + C - 18), f"{cnt[b]}", fill=(200, 200, 200))
            draw.text((3 * C + 4, y0 + 4), "profile: rump -> head", fill=(235, 235, 235))

        ex.close()

    args.out.mkdir(parents=True, exist_ok=True)
    p = args.out / f"method_L{args.layer}_{args.facet}_{args.size}.jpg"
    sheet.save(p, quality=92)
    print(f"\nwrote {p}   ({n_ok}/{len(picks)} correct on these samples)")
    print("  1 crop      GREEN = true head; RINGED = predicted; RED BORDER = wrong")
    print("  2 fg mask   centre-vs-corner. Ragged here => everything downstream suffers.")
    print("  3 PCA-RGB   the panel that already showed parts ARE present")
    print("  4 profile   rump -> head bins. Different colours => an anatomical gradient exists;")
    print("              one flat colour => there is no head/tail cue to find, and we stop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
