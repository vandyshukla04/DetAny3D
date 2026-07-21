"""Cut crops + 4-candidate geometry for EVERY detection of chosen videos.  [LOCAL / CPU]

    python -m tools.heading.predict_crops \
        --root /mnt/d/3DBOX/papersubdata --group zebr3 \
        --videos DJI_20250802085130_0007_V,DJI_20250802085520_0008_V \
        --out data/heading/crops_reid.npz

WHY A SEPARATE EXTRACTOR
------------------------
`extract_crops.py` cuts crops only where a heading LABEL exists (walking, or human-locked). The re-ID
study needs a PREDICTED heading on *every* detection of a video -- including standing, unlabelled tracks
-- so this walks `tracking_summary.json` directly and cuts a crop for each (track, frame). It emits the
SAME per-crop geometry (`face_uv`, `face_alpha`, `face_az`, `geo_axis`, `face_ids`, `crop_box`, `box2d`,
`image_name`) so `cropset.CropSet` and the template scorer read it with zero changes. The only fields it
CANNOT fill are the label-derived ones (`y_face`, `az`, `alpha`, `front_face`); `y_face` is written as
`-1` (unknown) and the rest are omitted -- prediction never reads them.

The mask, DINOv3 forward and template scoring happen later on the cluster (see the re-ID plan); this step
is pure geometry and runs where papersubdata lives.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np

from tools.heading.conventions import FACE_AXIS, face_centers_world
from tools.heading.extract_crops import crop_transform, square_crop
from tools.heading.papersub import SPECIES_OF_GROUP, load_segment


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("/mnt/d/3DBOX/papersubdata"))
    ap.add_argument("--group", default="zebr3", help="papersubdata group dir (species prefix)")
    ap.add_argument("--videos", required=True, help="comma-separated video dir names")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--pad", type=float, default=0.15)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth frame of each track")
    ap.add_argument("--min-body-px", type=float, default=24.0)
    args = ap.parse_args()

    from PIL import Image

    species = SPECIES_OF_GROUP[args.group[:4]]
    videos = [v.strip() for v in args.videos.split(",") if v.strip()]

    jpegs: list[bytes] = []
    rec: list[dict] = []
    n_small = n_bad = 0

    seg_dirs = sorted(p for v in videos
                      for p in (args.root / args.group / v).glob("seg*") if p.is_dir())
    print(f"{species}: {len(videos)} videos, {len(seg_dirs)} segments")

    for sd in seg_dirs:
        seg = load_segment(sd)
        _ = seg.scale                                  # measure + assert the 518 ratio, once
        seg_key = f"{args.group}/{sd.parent.name}/{sd.name}"

        # decode each frame once: iterate frames, then the tracks present in that frame
        frames = sorted(seg.cameras)
        for fidx in frames:
            img = None
            for tid, tr in seg.tracks.items():
                try:
                    i = tr.index_of_frame(fidx)
                except KeyError:
                    continue
                if (i % args.stride) != 0:
                    continue

                box = seg.crop_box(tr, i)
                body_px = float(max(box[2] - box[0], box[3] - box[1]))
                if body_px < args.min_body_px:
                    n_small += 1
                    continue

                faces = seg.horizontal_face_dirs(tr, i)
                fids = sorted(faces)
                if len(fids) != 4:
                    n_bad += 1
                    continue
                centres = face_centers_world(tr.centers[i], tr.dims[i], tr.rotations[i])
                uv = seg.cameras[fidx].project(np.stack([centres[f] for f in fids]))
                if not np.isfinite(uv).all():
                    n_bad += 1
                    continue

                if img is None:                        # first usable track in this frame
                    fp = seg.frame_path(fidx)
                    if not fp.is_file():
                        n_bad += 1
                        break
                    img = np.asarray(Image.open(fp).convert("RGB"))
                crop = square_crop(img, box, args.pad)
                if crop is None:
                    n_bad += 1
                    continue

                ox, oy, side = crop_transform(box, args.pad)
                uv = (uv - np.array([ox, oy])) / side          # -> [0,1] within the square crop

                up = seg.up_at(tr, i)
                ext = {}
                for k, f in enumerate(fids):
                    c = FACE_AXIS[f]
                    a = tr.rotations[i][:, c] - np.dot(tr.rotations[i][:, c], up) * up
                    ext[k] = float(tr.dims[i][c] * np.linalg.norm(a))
                geo_axis = int(max(ext, key=ext.get))

                to_cam = seg.cameras[fidx].center - tr.centers[i]
                h_cam = to_cam - np.dot(to_cam, up) * up
                cam_az = seg.azimuth_of(h_cam) if np.linalg.norm(h_cam) > 1e-9 else 0.0
                cam_elev = float(np.arctan2(np.dot(to_cam, up), np.linalg.norm(h_cam) + 1e-12))

                buf = io.BytesIO()
                Image.fromarray(crop).resize((args.size, args.size), Image.BICUBIC).save(
                    buf, format="JPEG", quality=args.quality)
                jpegs.append(buf.getvalue())
                rec.append({
                    "seg": seg_key, "video": sd.parent.name, "species": species,
                    "track": f"{seg_key}::{tr.tid}", "frame": int(fidx),
                    "face_ids": np.array(fids, dtype=np.int8),
                    "face_uv": uv.astype(np.float32),
                    "face_alpha": np.array([seg.alpha_of(tr, i, faces[f]) for f in fids],
                                           dtype=np.float32),
                    "face_az": np.array([seg.azimuth_of(faces[f]) for f in fids], dtype=np.float32),
                    "cam_az": float(cam_az), "cam_elev": float(cam_elev),
                    "body_px": body_px, "geo_axis": geo_axis,
                    "crop_box": np.array([ox, oy, side], dtype=np.float32),
                    "box2d": box.astype(np.float32),
                    "image_name": seg.cameras[fidx].image_name,
                })
        print(f"  {seg_key}: {len(rec)} crops so far", flush=True)

    if not rec:
        print("no crops produced")
        return 1

    out = dict(
        jpeg=np.array(jpegs, dtype=object),
        y_face=np.full(len(rec), -1, dtype=np.int8),          # UNKNOWN: prediction never reads it
        face_uv=np.stack([r["face_uv"] for r in rec]),
        face_alpha=np.stack([r["face_alpha"] for r in rec]),
        face_az=np.stack([r["face_az"] for r in rec]),
        cam_az=np.array([r["cam_az"] for r in rec], dtype=np.float32),
        cam_elev=np.array([r["cam_elev"] for r in rec], dtype=np.float32),
        face_ids=np.stack([r["face_ids"] for r in rec]),
        geo_axis=np.array([r["geo_axis"] for r in rec], dtype=np.int8),
        species=np.array([r["species"] for r in rec]),
        video=np.array([r["video"] for r in rec]),
        seg=np.array([r["seg"] for r in rec]),
        track=np.array([r["track"] for r in rec]),
        frame=np.array([r["frame"] for r in rec], dtype=np.int32),
        body_px=np.array([r["body_px"] for r in rec], dtype=np.float32),
        crop_box=np.stack([r["crop_box"] for r in rec]),
        box2d=np.stack([r["box2d"] for r in rec]),
        image_name=np.array([r["image_name"] for r in rec]),
        seg_name=np.array([r["seg"].split("/")[-1] for r in rec]),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    mb = args.out.stat().st_size / 1e6
    print(f"\nwrote {len(rec)} crops -> {args.out}  ({mb:.0f} MB)")
    print(f"  dropped: {n_small} too small (<{args.min_body_px:.0f}px), {n_bad} unusable")
    for v in videos:
        n = int((out["video"] == v).sum())
        ntr = len({t for t, vv in zip(out["track"], out["video"]) if vv == v})
        print(f"    {v}: {n} crops, {ntr} tracks")
    print("\nscp this to the cluster; score it with the template + emit DINOv3 features.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
