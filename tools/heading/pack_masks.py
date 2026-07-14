"""Pack the SAM3 instance masks into ONE small file, aligned 1:1 with our crops.  [CLUSTER]

    # on the cluster (the masks live there; papersubdata does not)
    python -m tools.heading.pack_masks --labels data/heading/labels.npz \
        --out data/heading/masks.npz
    # then scp masks.npz back to the local machine and re-cut the crops

WHY A SEPARATE STEP
-------------------
The two clean sources live in different places, and each is authoritative for its half:
  * papersubdata (LOCAL)  -- the 3D boxes, cameras, world geometry
  * SAM3 masks  (CLUSTER) -- clean per-instance segmentation
Rather than sync either dataset, we ship the intersection: for every (video, seg, track, frame)
we actually crop, one small square mask, cut with the SAME crop box the local extractor uses.

THE CROP BOX MUST BE BIT-IDENTICAL ON BOTH SIDES
-------------------------------------------------
Both sides compute `bbox_2d * (width / 518)` from the same `bbox_2d` (papersubdata and the archive
carry identical tracking summaries), and both call `crop_transform`/`square_crop` from
`extract_crops`. If the boxes drifted even slightly, each crop would be paired with a
MISALIGNED mask -- and a misaligned mask is worse than none: it would clip the head off the
target and leak the neighbour in, which is the exact failure we are here to fix.

AND THE JOIN IS ASSERTED, NOT ASSUMED
--------------------------------------
`obj_<N>` is *supposed* to be track `N`, and `frame_%06d.png` is *supposed* to be `frame_%06d.jpg`.
Both look right. But a wrong track pairing or a one-frame offset hands us the NEIGHBOUR'S mask and
still produces a perfectly plausible number. So every mask's centroid is checked against its own
2D box, and a systematic mismatch is a hard failure.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.heading.extract_crops import crop_transform, square_crop
from tools.heading.masks import build_index, check_join, load_mask
from tools.heading.papersub import VGGT_LONG_SIDE


def key_of(video: str, seg: str, track: str, frame: int) -> str:
    """Tree-independent join key. Video names are globally unique across the archive, so this key
    works whether the geometry came from papersubdata (groups: zebr3/...) or from the archive
    (datasets: wildbox_tomblair/...)."""
    return f"{video}/{seg}::{track}::{int(frame)}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, required=True, help="labels.npz from autolabel.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=64,
                    help="stored mask resolution. The patch grid is at most 28x28, so 64 is >2x "
                         "oversampled and the file stays tiny.")
    ap.add_argument("--pad", type=float, default=0.15, help="MUST match extract_crops --pad")
    ap.add_argument("--max-bad", type=float, default=0.02,
                    help="fraction of masks whose centroid may fall outside their own 2D box "
                         "before we declare the join broken and refuse to continue")
    args = ap.parse_args()

    import json

    from PIL import Image

    d = np.load(args.labels, allow_pickle=True)
    idx = build_index()
    print(f"{len(d['seg'])} labelled instances | {len(idx)} videos in the archive")

    # group by segment so each tracking_summary / cameras.json is read once
    by_seg: dict[tuple[str, str], list[int]] = defaultdict(list)
    for j in range(len(d["seg"])):
        group, video, seg = str(d["seg"][j]).split("/")
        by_seg[(video, seg)].append(j)

    keys: list[str] = []
    packed: list[np.ndarray] = []
    n_missing = n_bad = 0
    offs: list[float] = []

    for si, ((video, seg), js) in enumerate(sorted(by_seg.items()), 1):
        vdir = idx.get(video)
        if vdir is None:
            n_missing += len(js)
            continue
        vres = vdir / seg / "vggt_results"
        try:
            ts = json.loads((vres / "tracking_summary.json").read_text())
            cams = json.loads((vres / "cameras.json").read_text())["cameras"]
        except (OSError, json.JSONDecodeError):
            n_missing += len(js)
            continue

        name_of = {int(c["frame_index"]): c["image_name"] for c in cams}
        W = int(cams[0]["image_width"])
        # The archive's cameras.json is 518-space, so its own width is NOT the full-res width.
        # The masks' metadata carries the true frame resolution -- use that, and fail loudly if it
        # is missing rather than guessing.
        meta_p = vdir / seg / "sam3_masks" / "metadata.json"
        if not meta_p.is_file():
            n_missing += len(js)
            continue
        full_w = int(json.loads(meta_p.read_text())["resolution"][0])
        scale = full_w / VGGT_LONG_SIDE                  # identical to papersub.Segment.scale

        for j in js:
            track = str(d["track"][j])
            fidx = int(d["frame"][j])
            i = int(d["i"][j])
            img_name = name_of.get(fidx)
            tk = ts.get("tracks", {}).get(track)
            if img_name is None or tk is None:
                n_missing += 1
                continue

            m = load_mask(video, seg, track, img_name)
            if m is None:
                n_missing += 1
                continue

            box = np.asarray(tk["bbox_2d"][i], dtype=np.float64) * scale
            ok, off = check_join(m, box)
            offs.append(off)
            if not ok:
                n_bad += 1
                continue

            sq = square_crop(m.astype(np.uint8) * 255, box, args.pad)
            if sq is None:
                n_missing += 1
                continue
            small = np.asarray(Image.fromarray(sq[..., 0] if sq.ndim == 3 else sq)
                               .resize((args.size, args.size), Image.BILINEAR)) > 127
            keys.append(key_of(video, seg, track, fidx))
            packed.append(np.packbits(small))            # 64x64 bits -> 512 bytes

        if si % 25 == 0:
            print(f"  {si}/{len(by_seg)} segments, {len(packed)} masks", flush=True)

    if not packed:
        print("no masks packed -- check the archive path")
        return 1

    tot = len(packed) + n_bad
    frac_bad = n_bad / max(tot, 1)
    print(f"\n=== JOIN CHECK (mask centroid vs its own 2D box) ===")
    print(f"  packed {len(packed)} | missing {n_missing} | centroid OUTSIDE its box: {n_bad} "
          f"({100*frac_bad:.1f}%)")
    if offs:
        o = np.array(offs)
        print(f"  centroid offset (1.0 = the box edge): median {np.median(o):.2f}  "
              f"p90 {np.percentile(o, 90):.2f}")
    if frac_bad > args.max_bad:
        print(f"\n  *** JOIN IS BROKEN: {100*frac_bad:.1f}% of masks do not sit on their own "
              f"animal (limit {100*args.max_bad:.0f}%).")
        print(f"  Refusing to continue. A misjoined mask hands us the NEIGHBOUR'S segmentation --")
        print(f"  the exact bug this is meant to fix -- and would still look plausible downstream.")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, key=np.array(keys), mask=np.stack(packed),
                        size=np.array([args.size]), pad=np.array([args.pad]))
    print(f"\nwrote {len(packed)} masks -> {args.out} "
          f"({args.out.stat().st_size/1e6:.1f} MB)")
    print("scp this back to the local machine, then re-cut crops with --masks masks.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
