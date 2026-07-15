"""STAGE 1 (CLUSTER, GPU): export per-frame predictions + DINOv3 features + SAM masks for N tracks.

    python -m tools.heading.track_export --crops data/heading/crops.npz \
        --out data/heading/track_export.npz --n 25 --frames 12 --device cuda

Picks N random tracks from the HELD-OUT videos, balanced across species, and for each frame records
everything the local renderer (stage 2) cannot compute itself:

  * head_face_id / alpha / flank / end / margin  -- the resolved heading + viewpoint tag (needs DINOv3)
  * pca                                           -- the DINOv3 PCA-RGB patch image, ONE BASIS PER
                                                     TRACK (so the same colour = the same body part
                                                     in every frame of the track)
  * masked                                        -- the SAM-masked crop (needs the archive masks)
  * coverage                                      -- per-aspect duty cycle + the exemplar frame of
                                                     each captured aspect (LEFT / RIGHT / FACE / REAR)

Everything geometric (full frames, all-track 3D boxes, cameras) is left to the LOCAL renderer, where
papersubdata is full-resolution and verified. This file exports only the DINOv3- and SAM-dependent
parts, in ONE .npz, so stage 2 needs a single scp.

Frames here are the WALKING crops in crops.npz (that is what the pipeline built). The coverage card is
therefore over the track's walking frames; the renderer labels it as such.
"""
from __future__ import annotations

import argparse
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.heading.cropset import CropSet
from tools.heading.descriptors import DEFAULT_MODEL, Config, DenseExtractor, foreground
from tools.heading.paper_fig import track_pca
from tools.heading.split import video_split
from tools.heading.template import Accumulator, choose
from tools.heading.viewpoint import EDGE_ON, viewpoint_of

ASPECTS = ("LEFT", "RIGHT", "FACE", "REAR")


def dominant_aspect(alpha: float) -> str:
    """The single aspect a frame most shows: the larger of the flank / end components, and its sign."""
    s, c = np.sin(alpha), np.cos(alpha)
    if abs(s) >= abs(c):
        return "LEFT" if s > 0 else "RIGHT"
    return "FACE" if c < 0 else "REAR"


def aspect_strength(alpha: float, aspect: str) -> float:
    s, c = np.sin(alpha), np.cos(alpha)
    return {"LEFT": max(s, 0), "RIGHT": max(-s, 0),
            "FACE": max(-c, 0), "REAR": max(c, 0)}[aspect]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--facet", default="token", choices=["token", "key"])
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--fit", type=int, default=1500)
    ap.add_argument("--n", type=int, default=25, help="number of tracks")
    ap.add_argument("--frames", type=int, default=12, help="max frames per track (evenly in time)")
    ap.add_argument("--pca-size", type=int, default=112)
    ap.add_argument("--mask-size", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from PIL import Image

    crops = CropSet(args.crops)
    d = crops.d
    te, _ = video_split(crops.species, crops.video, seed=args.seed)
    cfg = Config(args.layer, args.facet, args.size, args.bins)

    # ---- pick N held-out tracks, balanced across species ----
    by_track = defaultdict(list)
    for i in np.where(te)[0]:
        by_track[str(d["track"][i])].append(int(i))
    per_sp = defaultdict(list)
    for tid, ii in by_track.items():
        if len(ii) >= 4:                                  # need a few frames to be a "track"
            per_sp[str(d["species"][ii[0]])].append(tid)
    rng = np.random.default_rng(args.seed)
    for s in per_sp:
        rng.shuffle(per_sp[s])
    order = sorted(per_sp)                                # round-robin across species
    picks, qi = [], 0
    while len(picks) < args.n and any(per_sp.values()):
        s = order[qi % len(order)]
        if per_sp[s]:
            picks.append(per_sp[s].pop())
        qi += 1
    print(f"{len(picks)} tracks: " + ", ".join(f"{p.split('::')[-1]}" for p in picks[:8]) + " ...")

    # ---- template from TRAINING crops (nothing trained; a mean) ----
    idx_fit = np.sort(rng.permutation(np.where(~te)[0])[: args.fit])
    crops.prefetch(idx_fit)
    with DenseExtractor(args.model, args.device, args.dtype) as ex:
        acc = Accumulator(cfg)
        for b in range(0, len(idx_fit), args.batch):
            items = crops.batch(idx_fit[b: b + args.batch])
            for g, it in zip(ex.grid(np.stack([i.image for i in items]), cfg), items):
                acc.add(g, it)
        tmpl = acc.build()
        print(f"template: {tmpl.n_fitted}")

        rec = []                                          # flat per-frame records
        pcas, maskeds = [], []
        cover = {}                                        # track -> coverage dict

        for ti, tid in enumerate(picks, 1):
            ii = sorted(by_track[tid], key=lambda k: int(d["frame"][k]))
            sel = np.linspace(0, len(ii) - 1, min(args.frames, len(ii))).astype(int)
            ii = [ii[k] for k in sel]
            crops.prefetch(np.array(ii))
            items = crops.batch(ii)
            grids = [ex.grid(it.image[None], cfg)[0] for it in items]
            fgs = [foreground(g, it.instance) for g, it in zip(grids, items)]
            project = track_pca(grids, fgs)               # ONE PCA basis for the whole track

            alphas, tframe = [], []
            for it, g, fg in zip(items, grids, fgs):
                if it.species in tmpl.templates:
                    s = tmpl.score_faces(g, fg, it.face_uv, it.face_ids, it.species)
                    slot, margin = choose(s, it.face_ids, axis=it.geo_axis)
                else:
                    slot, margin = -1, 0.0                # no template -> abstain to motion label
                if slot < 0:
                    slot = it.y_face                      # abstain -> fall back to motion label
                head_id = int(it.face_ids[slot])
                alpha = float(d["face_alpha"][it.i, slot])
                vp = viewpoint_of(alpha)
                alphas.append(alpha)
                tframe.append(it.frame)
                rec.append({
                    "track": tid, "video": it.video, "seg": str(d["seg_name"][it.i]),
                    "frame": int(it.frame), "image_name": str(d["image_name"][it.i]),
                    "head_face_id": head_id, "alpha": alpha, "margin": float(margin),
                    "flank": vp["flank"], "flank_w": float(vp["flank_strength"]),
                    "end": vp["end"], "end_w": float(vp["end_strength"]),
                    "az": float(d["az"][it.i]),           # motion heading azimuth (walking ref)
                })
                # DINOv3 PCA image (per-track basis), and the SAM-masked crop
                rgb = (project(g) * fg[..., None]) if project is not None else np.zeros((*g.shape[:2], 3))
                pcas.append(np.asarray(Image.fromarray((255 * rgb).astype(np.uint8))
                                       .resize((args.pca_size, args.pca_size), Image.NEAREST)))
                m = it.instance
                cimg = it.image.copy()
                if m is not None:
                    mm = np.asarray(Image.fromarray(m.astype(np.uint8) * 255)
                                    .resize((cimg.shape[1], cimg.shape[0]), Image.NEAREST)) > 127
                    cimg = np.where(mm[..., None], cimg, 255)     # white background
                buf = io.BytesIO()
                Image.fromarray(cimg.astype(np.uint8)).resize(
                    (args.mask_size, args.mask_size), Image.BICUBIC).save(buf, "JPEG", quality=92)
                maskeds.append(buf.getvalue())

            # ---- coverage over this track ----
            alphas = np.array(alphas)
            dom = [dominant_aspect(a) for a in alphas]
            duty = {asp: float(np.mean([x == asp for x in dom])) for asp in ASPECTS}
            exemplar = {}                                 # aspect -> the frame that best shows it
            base = len(rec) - len(items)                  # global index of this track's first frame
            for asp in ASPECTS:
                strengths = [aspect_strength(a, asp) for a in alphas]
                if max(strengths) >= EDGE_ON:
                    exemplar[asp] = base + int(np.argmax(strengths))
                else:
                    exemplar[asp] = -1                    # MISSING (never shown broadside enough)
            cover[tid] = {"duty": duty, "exemplar": exemplar,
                          "n_frames": len(items),
                          "species": items[0].species, "video": items[0].video}
            print(f"  [{ti}/{len(picks)}] {tid.split('::')[-1]} ({items[0].species}): "
                  f"{len(items)} frames | " +
                  " ".join(f"{a}:{100*duty[a]:.0f}%" for a in ASPECTS), flush=True)

    # ---- write ONE npz ----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    keys = ["track", "video", "seg", "frame", "image_name", "head_face_id",
            "alpha", "margin", "flank", "flank_w", "end", "end_w", "az"]
    out = {k: np.array([r[k] for r in rec]) for k in keys}
    out["pca"] = np.stack(pcas).astype(np.uint8)
    out["masked"] = np.array(maskeds, dtype=object)
    out["coverage"] = np.array([json.dumps(cover)], dtype=object)
    out["tracks"] = np.array(picks)
    np.savez_compressed(args.out, **out)
    print(f"\nwrote {len(rec)} frames across {len(picks)} tracks -> {args.out} "
          f"({args.out.stat().st_size/1e6:.0f} MB)")
    print("scp this ONE file to the local machine, then run tools.heading.track_render")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
