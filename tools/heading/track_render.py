"""STAGE 2 (LOCAL, CPU): render per-track figure ELEMENTS into numbered folders.  [/mnt/d papersubdata]

    python -m tools.heading.track_render --export data/heading/track_export.npz \
        --root /mnt/d/3DBOX/papersubdata --out /mnt/d/detany3d/track_figs

Reads the cluster bundle (predictions + DINOv3 PCA + SAM masked crops) and the full-resolution
papersubdata (frames, all-track 3D boxes, cameras), and writes, PER TRACK, every element as a
SEPARATE file so they can be arranged freely:

    track_NN_<species>_<video>_<trackid>/
      01_frame_boxaxis/ frame_XXXXXX.jpg   full frame; ALL animals' 3D boxes (grey), the target's
                                           box bold + body axis + red heading arrow + tag.  1/frame.
      02_dino_pca/      frame_XXXXXX.png    DINOv3 PCA-RGB, one basis for the track.        1/frame.
      03_heading/       frame_XXXXXX.jpg    the target crop + heading arrow + motion reference (grey)
                                           + viewpoint weights.                            1/frame.
      04_coverage/      wheel.png  bars.png                    the two coverage cards
                        view_LEFT.jpg view_RIGHT.jpg view_FACE.jpg view_REAR.jpg   masked exemplars
                                                                    (or a MISSING placeholder)
      info.json                            species, video, track, per-frame tags, coverage
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.heading.conventions import corners_of, face_centers_world
from tools.heading.papersub import load_segment


def _install_numpy2_compat() -> None:
    """Let a numpy<2 reader open object arrays pickled by numpy>=2.0.

    numpy 2.0 renamed its private `numpy.core` package to `numpy._core`, so the object-dtype
    members of the bundle (`coverage`, `masked`) name a module that numpy 1.x does not have --
    the error is `ModuleNotFoundError: No module named 'numpy._core'` at ACCESS time, not load
    time. The cluster (numpy>=2.0) writes the npz; this local box (numpy 1.x) reads it.

    We ALIAS the already-loaded `numpy.core.*` module objects into the `numpy._core.*` names the
    pickle asks for. This points the two names at the SAME loaded objects -- crucially it never
    re-imports numpy's C extensions (re-initialising `_multiarray_umath` under a second name
    segfaults). A no-op on numpy>=2.0 (the real `numpy._core` is already present) and on 1.x when
    no such pickle is ever touched.
    """
    if "numpy._core" in sys.modules:
        return                                            # numpy>=2.0, or already aliased
    if not np.__version__.startswith("1."):
        return                                            # only 1.x lacks numpy._core
    # make sure the submodules an object-array pickle names (_reconstruct lives in multiarray) are
    # loaded, then mirror the whole numpy.core subtree onto numpy._core with the same objects.
    import numpy.core.multiarray  # noqa: F401
    import numpy.core.numeric     # noqa: F401
    for name, mod in list(sys.modules.items()):
        if name == "numpy.core" or name.startswith("numpy.core."):
            sys.modules.setdefault("numpy._core" + name[len("numpy.core"):], mod)

# 12 edges of the box, in CORNERS_LOCAL_UNIT order (bottom loop, top loop, 4 verticals)
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
ASPECTS = ("LEFT", "RIGHT", "FACE", "REAR")


def _arrow(draw, x0, y0, x1, y1, col, w, halo=(0, 0, 0)):
    def stroke(c, ww):
        draw.line([(x0, y0), (x1, y1)], fill=c, width=ww)
        a = math.atan2(y1 - y0, x1 - x0)
        L = 0.28 * math.hypot(x1 - x0, y1 - y0)
        for s in (+1, -1):
            b = a + s * math.radians(150)
            draw.line([(x1, y1), (x1 + L * math.cos(b), y1 + L * math.sin(b))], fill=c, width=ww)
    if halo is not None:
        stroke(halo, w + 3)               # dark underlay so the arrow reads on any background
    stroke(col, w)


def _hline(dr, p, q, col, w, halo=(0, 0, 0)):
    """A line with a dark halo -- a white wireframe stays visible on grass or on the animal."""
    if halo is not None:
        dr.line([tuple(p), tuple(q)], fill=halo, width=w + 4)
    dr.line([tuple(p), tuple(q)], fill=col, width=w)


def _project_dir(cam, p0_world, dir_world, length):
    """(box centre, world direction) -> two full-res pixels along `dir` of `length` (world units)."""
    a = cam.project(p0_world[None])[0]
    b = cam.project((p0_world + length * dir_world)[None])[0]
    return a, b


def render_track(tid, frames, cov, seg_dir, out, masked_of):
    """frames: list of per-frame dicts (already sorted). cov: coverage dict for this track."""
    from PIL import Image, ImageDraw

    sp = frames[0]["species"]
    S = load_segment(seg_dir)
    tr = S.tracks[str(frames[0]["track_id"])]

    (out / "01_frame_boxaxis").mkdir(parents=True, exist_ok=True)
    (out / "02_dino_pca").mkdir(parents=True, exist_ok=True)
    (out / "03_heading").mkdir(parents=True, exist_ok=True)
    (out / "04_coverage").mkdir(parents=True, exist_ok=True)

    for fr in frames:
        fidx = fr["frame"]
        cam = S.cameras[fidx]
        i = tr.index_of_frame(fidx)
        fp = S.frame_path(fidx)
        if not fp.is_file():
            continue
        img = Image.open(fp).convert("RGB")

        # shared per-frame geometry (the resolved heading is the RESULT -> drawn only in row 03)
        up = S.up_at(tr, i)
        cen = tr.centers[i]
        fcen = face_centers_world(cen, tr.dims[i], tr.rotations[i])
        L = 0.7 * tr.body_length
        hd = _face_dir(tr, i, fr["head_face_id"], up)
        a0, a1 = _project_dir(cam, cen, hd, L)

        # ---------- 01 GEOMETRY: box wireframe + UNSIGNED body axis + 4 candidate faces ----------
        # The box gives an AXIS, not a direction. This row shows the box, its proposed body axis
        # (BOLD, both ends, no arrowhead), and the 4 candidate face centres -- and deliberately NO
        # heading arrow. Appearance + locomotion resolve the signed heading; that is row 03.
        canvas = img.copy()
        dr = ImageDraw.Draw(canvas)
        for otid, otr in S.tracks.items():                       # herd context, dim grey
            try:
                oi = otr.index_of_frame(fidx)
            except KeyError:
                continue
            cn = cam.project(corners_of(otr.centers[oi], otr.dims[oi], otr.rotations[oi]))
            if np.isfinite(cn).all():
                for a, b in EDGES:
                    dr.line([tuple(cn[a]), tuple(cn[b])], fill=(150, 150, 150), width=2)
        # the TARGET box: bright MAGENTA wireframe -- shines through the savanna greens/browns
        cn = cam.project(corners_of(tr.centers[i], tr.dims[i], tr.rotations[i]))
        if np.isfinite(cn).all():
            for a, b in EDGES:
                _hline(dr, cn[a], cn[b], (255, 40, 190), 3, halo=(25, 0, 20))
        # the 4 CANDIDATE face centres (the box's 4 horizontal faces): white dots
        for f in _hfaces(tr, i, up):
            pf = cam.project(fcen[f][None])[0]
            if np.isfinite(pf).all():
                dr.ellipse([pf[0] - 6, pf[1] - 6, pf[0] + 6, pf[1] + 6],
                           fill=(255, 255, 255), outline=(25, 0, 20), width=2)
        # the PROPOSED body axis: BOLD, UNSIGNED (both ends, no arrowhead), bright yellow
        ga, gb = _axis_ends(tr, i, up)
        pa, pb = cam.project(fcen[ga][None])[0], cam.project(fcen[gb][None])[0]
        if np.isfinite([pa, pb]).all():
            _hline(dr, pa, pb, (255, 235, 0), 5, halo=(30, 25, 0))
        dr.text((8, 8), f"{sp}  t={fidx}", fill=(255, 255, 255))
        canvas.save(out / "01_frame_boxaxis" / f"frame_{fidx:06d}.jpg", quality=92)

        # ---------- 03: the target crop + heading + motion reference ----------
        box = S.crop_box(tr, i)
        ox, oy, side = _crop_transform(box, 0.15)
        crop = _square_crop(np.asarray(img), box, 0.15)
        if crop is not None:
            cc = Image.fromarray(crop)
            dc = ImageDraw.Draw(cc)
            def to_crop(px):
                return ((px[0] - ox) / side * cc.width, (px[1] - oy) / side * cc.height)
            # motion reference (grey) — the walking direction
            e1, e2, _ = S.ground_basis
            md = math.cos(fr["az"]) * e1 + math.sin(fr["az"]) * e2
            m0, m1 = _project_dir(cam, cen, md, L)
            if np.isfinite([m0, m1]).all():
                _arrow(dc, *to_crop(m0), *to_crop(m1), (170, 170, 170), 4)
            # resolved heading (red)
            if np.isfinite([a0, a1]).all():
                _arrow(dc, *to_crop(a0), *to_crop(a1), (255, 55, 55), 4)
            dc.text((5, cc.height - 16),
                    f"{fr['flank']} {fr['flank_w']:.2f}  {fr['end']} {fr['end_w']:.2f}",
                    fill=(90, 235, 255) if fr["flank_w"] >= 0.35 else (225, 175, 70))
            cc.save(out / "03_heading" / f"frame_{fidx:06d}.jpg", quality=92)

        # ---------- 02: DINOv3 PCA (from the cluster bundle) ----------
        pca = fr["pca"]
        Image.fromarray(pca).resize((side_px := 220, side_px), Image.NEAREST).save(
            out / "02_dino_pca" / f"frame_{fidx:06d}.png")

    # ---------- 04: coverage cards + masked exemplars ----------
    _coverage(out / "04_coverage", cov, [fr["alpha"] for fr in frames], masked_of)
    (out / "info.json").write_text(json.dumps({
        "species": sp, "video": frames[0]["video"], "seg": frames[0]["seg"],
        "track": frames[0]["track_id"],
        "n_frames": len(frames), "coverage": cov,
        "frames": [{k: fr[k] for k in ("frame", "flank", "flank_w", "end", "end_w",
                                       "head_face_id", "margin")} for fr in frames],
    }, indent=1))


def _coverage(cdir, cov, alphas, masked_of):
    from PIL import Image, ImageDraw

    duty, exemplar = cov["duty"], cov["exemplar"]

    # --- the RADAR: a top-down animal (drawn HEAD-UP) with one dot per frame at the angle the
    #     CAMERA observed it from. alpha = the animal's heading vs the viewing ray, so alpha=pi is
    #     face-on (camera in front -> dot at top), alpha=0 is tail-on (dot at bottom), +pi/2 is the
    #     LEFT flank (dot at left). Reads directly as "the drone saw this animal mostly from ...". ---
    W, CAP = 320, 26
    radar = Image.new("RGB", (W, W + CAP), (18, 18, 22))
    dw = ImageDraw.Draw(radar)
    cx, cy = W // 2, W // 2
    R = 0.40 * W
    r_ring = 0.80 * R
    dw.ellipse([cx - R, cy - R, cx + R, cy + R], outline=(70, 70, 82), width=2)
    # head-on / tail-on DEAD ZONES (|sin alpha| < 0.35): no usable flank exists there
    th = math.degrees(math.asin(0.35))
    for centre in (90, 270):                              # 90 = tail-on (bottom), 270 = face-on (top)
        dw.arc([cx - r_ring, cy - r_ring, cx + r_ring, cy + r_ring],
               centre - th, centre + th, fill=(74, 50, 50), width=16)
    for asp, (lx, ly) in {"FACE": (cx - 13, cy - R + 2), "REAR": (cx - 13, cy + R - 12),
                          "LEFT": (cx - R + 3, cy - 6), "RIGHT": (cx + R - 33, cy - 6)}.items():
        dw.text((lx, ly), asp, fill=(205, 205, 215))
    # the animal, top-down, HEAD UP -> the direction it FACES points to the FACE label
    bw, bt, bb = 0.055 * W, cy - 0.11 * W, cy + 0.13 * W
    dw.ellipse([cx - bw, bt, cx + bw, bb], fill=(120, 120, 132))                   # body
    dw.polygon([(cx, cy - 0.17 * W), (cx - 0.05 * W, bt + 4), (cx + 0.05 * W, bt + 4)],
               fill=(150, 150, 162))                                              # head (points up)
    # one dot per frame: WHERE THE CAMERA WAS relative to the animal
    for k, a in enumerate(alphas):
        sr = a + math.pi / 2
        rr = r_ring - (k % 3) * 7                         # de-overlap repeated viewpoints
        x, y = cx + rr * math.cos(sr), cy + rr * math.sin(sr)
        usable = abs(math.sin(a)) >= 0.35                 # broadside enough for a flank tag
        col = (90, 220, 255) if usable else (225, 175, 70)
        dw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=col, outline=(15, 15, 18))
    # caption: the re-ID-relevant conclusion (which flanks this track actually shows)
    seenL = any(math.sin(a) >= 0.35 for a in alphas)
    seenR = any(math.sin(a) <= -0.35 for a in alphas)
    cap = ("both flanks seen" if seenL and seenR else
           "LEFT flank only  (RIGHT missing)" if seenL else
           "RIGHT flank only  (LEFT missing)" if seenR else
           "no broadside flank (head/tail-on)")
    dw.text((8, W + 6), cap, fill=(200, 220, 235))
    radar.save(cdir / "wheel.png")

    # --- the BARS: one per aspect, width proportional to duty ---
    BW, BH = 320, 150
    bars = Image.new("RGB", (BW, BH), (18, 18, 22))
    db = ImageDraw.Draw(bars)
    for k, asp in enumerate(ASPECTS):
        y = 10 + k * 34
        du = duty[asp]
        db.rectangle([90, y, 90 + int((BW - 110) * du), y + 22],
                     fill=(90, 200, 235) if du > 0 else (60, 60, 66))
        db.text((6, y + 4), asp, fill=(230, 230, 235))
        db.text((BW - 40, y + 4), f"{100*du:.0f}%",
                fill=(230, 230, 235) if du > 0 else (120, 120, 130))
    bars.save(cdir / "bars.png")

    # --- masked exemplar per aspect (or MISSING placeholder) ---
    for asp in ASPECTS:
        gi = exemplar[asp]
        if gi is not None and gi >= 0 and masked_of(gi) is not None:
            Image.open(io.BytesIO(masked_of(gi))).save(cdir / f"view_{asp}.jpg", quality=92)
        else:
            ph = Image.new("RGB", (160, 160), (30, 30, 36))
            ImageDraw.Draw(ph).text((30, 72), f"{asp}\nMISSING", fill=(200, 120, 120))
            ph.save(cdir / f"view_{asp}.jpg", quality=92)


# ---- small geometry helpers (all full-res, from papersubdata) ----
def _axis_ends(tr, i, up):
    from tools.heading.conventions import FACE_AXIS
    ext = {}
    for f in _hfaces(tr, i, up):
        c = FACE_AXIS[f]
        a = tr.rotations[i][:, c] - np.dot(tr.rotations[i][:, c], up) * up
        ext[f] = float(tr.dims[i][c] * np.linalg.norm(a))
    from tools.heading.conventions import OPPOSITE_FACE
    g = max(ext, key=ext.get)
    return g, OPPOSITE_FACE[g]


def _hfaces(tr, i, up):
    from tools.heading.frame import horizontal_faces
    return horizontal_faces(tr.rotations[i], up)


def _face_dir(tr, i, face_id, up):
    from tools.heading.conventions import face_normals_world
    d = face_normals_world(tr.rotations[i])[int(face_id)]
    d = d - np.dot(d, up) * up
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else d


def _crop_transform(box, pad):
    from tools.heading.extract_crops import crop_transform
    return crop_transform(box, pad)


def _square_crop(img, box, pad):
    from tools.heading.extract_crops import square_crop
    return square_crop(img, box, pad)


def _group(species):
    # the local render only needs the group PREFIX; the segment path is resolved by globbing
    return {"elephant": "elep", "rhino": "rhin", "zebra": "zebr", "giraffe": "gira"}[species]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", type=Path, required=True, help="track_export.npz from the cluster")
    ap.add_argument("--root", type=Path, default=Path("/mnt/d/3DBOX/papersubdata"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _install_numpy2_compat()          # bundle may be written by a newer numpy than the local one
    E = np.load(args.export, allow_pickle=True)
    cover = json.loads(str(E["coverage"][0]))
    masked = E["masked"]
    def masked_of(gi):
        b = masked[gi]
        return b.tobytes() if isinstance(b, np.ndarray) else b

    # group flat records by track, preserving order (already time-sorted in the export)
    by_track = defaultdict(list)
    for gi in range(len(E["track"])):
        by_track[str(E["track"][gi])].append(gi)

    # resolve seg directories by GLOB (species-group prefix + video + seg)
    for k in cover:
        cover[k]["exemplar"] = {a: int(v) for a, v in cover[k]["exemplar"].items()}

    args.out.mkdir(parents=True, exist_ok=True)
    for ti, (tid, gis) in enumerate(sorted(by_track.items()), 1):
        frames = []
        for gi in gis:
            frames.append({
                "track_id": tid.split("::")[-1],
                "species": str(E["species"][gi]) if "species" in E else cover[tid]["species"],
                "video": str(E["video"][gi]), "seg": str(E["seg"][gi]),
                "frame": int(E["frame"][gi]), "image_name": str(E["image_name"][gi]),
                "head_face_id": int(E["head_face_id"][gi]), "alpha": float(E["alpha"][gi]),
                "flank": str(E["flank"][gi]), "flank_w": float(E["flank_w"][gi]),
                "end": str(E["end"][gi]), "end_w": float(E["end_w"][gi]),
                "margin": float(E["margin"][gi]), "az": float(E["az"][gi]),
                "pca": E["pca"][gi],
            })
        sp, vid = cover[tid]["species"], cover[tid]["video"]
        folder = args.out / f"track_{ti:02d}_{sp}_{vid}_{tid.split('::')[-1]}"
        # resolve the seg dir robustly (group prefix may map to elep1/elep2/...)
        seg = frames[0]["seg"]
        matches = sorted(args.root.glob(f"{_group(sp)}*/{vid}/{seg}"))
        if not matches:
            print(f"  [{ti}] SKIP {tid}: no papersubdata dir for {_group(sp)}*/{vid}/{seg}")
            continue
        try:
            render_track(tid, frames, cover[tid], matches[0], folder, masked_of)
            print(f"  [{ti}] {folder.name}: {len(frames)} frames", flush=True)
        except Exception as e:                            # keep going; one bad track is not fatal
            print(f"  [{ti}] ERROR {tid}: {e}")

    print(f"\nwrote {len(by_track)} track folders -> {args.out}/")
    print("  each element is a separate file; arrange them however you like.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
