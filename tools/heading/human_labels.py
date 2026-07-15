"""Human face-locks -> heading labels, joined BY WORLD DIRECTION.  [LOCAL / CPU]

    python -m tools.heading.human_labels --out data/heading/human_labels.npz

11,084 human-annotated instances on 2 videos of GRAZING zebras. These are the committed standers --
animals that never walk in the segment -- which `bridges.py` structurally cannot reach (it only
harvests brief stops). They are the only independent, human check we have, and the two videos are
excluded from training permanently (`split.HUMAN_LOCKED_VIDEOS`).

>>> THE LANDMINE. READ THIS BEFORE CHANGING ANYTHING BELOW. <<<
------------------------------------------------------------------
There are TWO tracking summaries per segment, and they do not agree:

    <seg>/vggt_results/tracking_summary.json               RAW      (per-frame PCA/SVD signs)
    <seg>/vggt_results/annotations/tracking_summary.json   CANONICAL (sign-aligned by the annotator)

**The human locks index into the CANONICAL rotations. papersubdata carries the RAW ones.**

On 1,874 of the 11,084 instances (16.9%) the two differ by a 180-degree rotation about the box's
local Y (up) axis. That maps face 2 <-> face 3 -- and `front` is ALWAYS on the X axis (across all 66
locked tracks, front is only ever 2 or 3). So a naive face-INDEX copy would invert head and tail on
ONE INSTANCE IN SIX, and report a believable ~83% while being systematically wrong.

So the join goes through the WORLD DIRECTION, never the index:

    h_world = R_canonical @ FACE_NORMAL_LOCAL[lock["front"]]        (the annotator's actual heading)
    front   = argmax_f  h_world . (R_papersub @ FACE_NORMAL_LOCAL[f])

and we ASSERT the match exceeds 0.999 on every instance. Centers, dimensions and extrinsics are
bit-identical between the two trees, so this is an exact relabel, not a fuzzy nearest-neighbour --
and if it ever stops being exact, that assert is the only thing standing between us and a plausible,
wrong number.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.heading.conventions import FACE_NORMAL_LOCAL
from tools.heading.papersub import load_segment
from tools.heading.split import HUMAN_LOCKED_VIDEOS

WILDBOX = Path("/mnt/d/3DBOX/Data/WildBox/data/2025_07_Zebras_BlTo/"
               "WildBox_sam3-vggtv1_processed/WildBox")
PAPERSUB = Path("/mnt/d/3DBOX/papersubdata")
GROUP = "zebr3"                    # where those two videos live in papersubdata


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wildbox", type=Path, default=WILDBOX)
    ap.add_argument("--papersub", type=Path, default=PAPERSUB)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-match", type=float, default=0.999,
                    help="the world-direction match must exceed this on EVERY instance")
    args = ap.parse_args()

    NL = {f: np.asarray(n, dtype=np.float64) for f, n in FACE_NORMAL_LOCAL.items()}
    rows = []
    matches = []
    n_flipped = 0
    per_seg = defaultdict(int)

    for video in HUMAN_LOCKED_VIDEOS:
        vdir = args.wildbox / video
        if not vdir.is_dir():
            print(f"  MISSING {vdir}")
            continue
        for sdir in sorted(vdir.glob("seg*")):
            ann = sdir / "vggt_results" / "annotations"
            lock_p = ann / "semantic_faces" / "track_face_locks.json"
            canon_p = ann / "tracking_summary.json"          # ** CANONICAL, not vggt_results/ **
            if not lock_p.is_file() or not canon_p.is_file():
                continue

            locks = json.loads(lock_p.read_text())
            canon = json.loads(canon_p.read_text())["tracks"]
            man_p = ann / "semantic_faces" / "manual_labels.json"
            manual = json.loads(man_p.read_text()) if man_p.is_file() else {}

            seg_key = f"{GROUP}/{video}/{sdir.name}"
            try:
                ps = load_segment(args.papersub / seg_key)
            except (FileNotFoundError, ValueError) as e:
                print(f"  skip {seg_key}: {e}")
                continue

            for tid, lock in locks.items():
                if tid not in canon or tid not in ps.tracks:
                    continue
                tr = ps.tracks[tid]
                Rc = np.asarray(canon[tid]["rotation_matrices"], dtype=np.float64)
                cframes = np.asarray(canon[tid]["frames"], dtype=int)
                c_of = {int(f): k for k, f in enumerate(cframes)}
                per_frame = manual.get(tid, {})

                for i in range(len(tr)):
                    fidx = int(tr.frames[i])
                    k = c_of.get(fidx)
                    if k is None or k >= len(Rc):
                        continue

                    # the annotator's front face -- per-frame if they re-assigned it, else the lock
                    fl = per_frame.get(str(fidx), per_frame.get(fidx))
                    front_canon = int((fl or lock)["front"])

                    # THE JOIN: the annotator's heading as a WORLD DIRECTION
                    h_world = Rc[k] @ NL[front_canon]

                    # ...matched against papersubdata's OWN face normals (raw rotations)
                    Rp = tr.rotations[i]
                    cos = {f: float(h_world @ (Rp @ n)) for f, n in NL.items()}
                    f_ps = max(cos, key=cos.get)
                    m = cos[f_ps]
                    matches.append(m)
                    if m < args.min_match:
                        continue                              # counted below; do not emit a guess
                    n_flipped += int(f_ps != front_canon)

                    up = ps.up_at(tr, i)
                    h = h_world - np.dot(h_world, up) * up
                    nrm = float(np.linalg.norm(h))
                    if nrm < 1e-6:
                        continue                              # the front face is vertical: unusable
                    rows.append({
                        "seg": seg_key, "video": video, "species": "zebra", "track": tid,
                        "i": i, "frame": fidx, "heading": h / nrm, "front_face": int(f_ps),
                    })
                    per_seg[seg_key] += 1

    if not matches:
        print("no human locks found -- check --wildbox")
        return 1

    M = np.asarray(matches)
    bad = int((M < args.min_match).sum())
    print(f"\n=== THE JOIN (world direction, never the face index) ===")
    print(f"  instances            : {len(M)}")
    print(f"  match  median {np.median(M):.5f}   min {M.min():.5f}")
    print(f"  BELOW {args.min_match}        : {bad}")
    print(f"  face INDEX differs between the two trees: {n_flipped} "
          f"({100*n_flipped/max(len(rows),1):.1f}%)")
    print(f"      ^ this is the 180-degree rotation between the RAW and CANONICAL boxes. A naive")
    print(f"        index copy would have inverted head and tail on every one of them.")

    if bad:
        print(f"\n  *** {bad} instance(s) do not match any papersubdata face to {args.min_match}.")
        print(f"  The two trees have diverged and the join can no longer be trusted. REFUSING to")
        print(f"  emit labels -- a wrong join here produces a plausible number, which is worse than")
        print(f"  no number at all.")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        seg=np.array([r["seg"] for r in rows]),
        video=np.array([r["video"] for r in rows]),
        species=np.array([r["species"] for r in rows]),
        track=np.array([r["track"] for r in rows]),
        i=np.array([r["i"] for r in rows], dtype=np.int32),
        frame=np.array([r["frame"] for r in rows], dtype=np.int32),
        heading=np.stack([r["heading"] for r in rows]).astype(np.float32),
        front_face=np.array([r["front_face"] for r in rows], dtype=np.int8),
    )
    print(f"\n  {len(rows)} human-labelled instances -> {args.out}")
    for k in sorted(per_seg):
        print(f"    {k:<44s} {per_seg[k]:5d}")
    print(f"\n  next:  extract_crops.py --labels {args.out} --out crops_human.npz --stride 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
