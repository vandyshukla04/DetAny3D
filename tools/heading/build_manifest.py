"""Build the heading-head training manifest from every human face annotation.

    python -m tools.heading.build_manifest --root /mnt/d/3DBOX --out data/heading/manifest.json

Reads BOTH annotation formats:
  * track_face_locks.json  -- one {front, top, left} per track  (applies to all its frames)
  * manual_labels.json     -- per-frame {front, back, top, bottom, left, right}   [richer]

and converts each to an image-space heading angle (see dataset.py for why a face id is
not a trainable target).

THE SANITY CHECK THAT MATTERS
-----------------------------
The human's *face id* flips between adjacent frames whenever the box's PCA signs flip --
that is the annotators correctly re-labelling a box whose axes were renamed under them.
The animal, of course, did not spin. So if our conversion is right, the derived
**angle must be temporally smooth** even while the face id jumps. We report both:

    face-id flips between adjacent frames   -- expected to be LARGE (the raw chaos)
    heading-angle jumps > 90 deg            -- expected to be ~0  (chaos undone)

If the angle jumps as much as the face id does, the conversion is wrong and nothing
downstream can be trusted.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.heading.dataset import HeadingSample, build_samples_for_segment, write_manifest
from tools.heading.io import find_face_locks, load_face_locks, load_segment, segment_dir_for_locks


def _gt_from_locks(seg_dir: Path, lock_path: Path) -> dict[tuple[str, int], int]:
    """Track-level lock -> one front face, applied to every frame of that track."""
    seg = load_segment(seg_dir, rotations="canonical")
    out: dict[tuple[str, int], int] = {}
    for tid, lock in load_face_locks(lock_path).items():
        tr = seg.tracks.get(tid)
        if tr is None:
            continue
        for f in tr.frames:
            out[(tid, int(f))] = int(lock["front"])
    return out


def _gt_from_manual(manual_path: Path) -> dict[tuple[str, int], int]:
    """Per-frame manual labels -> front face per (track, frame). Preferred: richer."""
    raw = json.loads(manual_path.read_text())
    out: dict[tuple[str, int], int] = {}
    for tid, per_frame in raw.items():
        if not isinstance(per_frame, dict):
            continue
        for fk, faces in per_frame.items():
            if isinstance(faces, dict) and "front" in faces:
                try:
                    out[(str(tid), int(fk))] = int(faces["front"])
                except (TypeError, ValueError):
                    continue
    return out


def _temporal_report(samples: list[HeadingSample]) -> tuple[float, float, int]:
    """(% face-id flips, % angle jumps >90deg, n adjacent pairs) across all tracks."""
    by_track: dict[tuple[str, str], list[HeadingSample]] = defaultdict(list)
    for s in samples:
        by_track[(s.segment, s.track_id)].append(s)

    face_flips = angle_jumps = pairs = 0
    for seq in by_track.values():
        seq.sort(key=lambda s: s.frame_index)
        for a, b in zip(seq, seq[1:]):
            pairs += 1
            face_flips += int(a.front_face != b.front_face)
            d = abs((b.heading_deg - a.heading_deg + 180.0) % 360.0 - 180.0)
            angle_jumps += int(d > 90.0)
    if not pairs:
        return 0.0, 0.0, 0
    return 100 * face_flips / pairs, 100 * angle_jumps / pairs, pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/mnt/d/3DBOX", type=Path)
    ap.add_argument("--out", default="data/heading/manifest.json", type=Path)
    args = ap.parse_args()

    samples: list[HeadingSample] = []
    skipped: list[str] = []

    # Prefer per-frame manual labels; fall back to the track-level locks.
    manual = [p for p in args.root.rglob("semantic_faces/manual_labels.json")
              if "RECYCLE" not in str(p)]
    seen_segments: set[Path] = set()

    for mp in manual:
        # .../<seg>/vggt_results/annotations/semantic_faces/manual_labels.json
        #            parents[2]   parents[1]     parents[0]
        seg_dir = mp.parents[3] if mp.parents[1].name == "annotations" else None
        if seg_dir is None or not (seg_dir / "vggt_results").is_dir():
            skipped.append(f"{mp}  (non-standard layout)")
            continue
        try:
            got = build_samples_for_segment(seg_dir, _gt_from_manual(mp))
        except (FileNotFoundError, KeyError) as e:
            skipped.append(f"{seg_dir}  ({e})")
            continue
        samples += got
        seen_segments.add(seg_dir)

    for lp in find_face_locks(args.root):
        seg_dir = segment_dir_for_locks(lp)
        if seg_dir in seen_segments:
            continue                                  # per-frame labels already used
        try:
            samples += build_samples_for_segment(seg_dir, _gt_from_locks(seg_dir, lp))
        except (FileNotFoundError, KeyError) as e:
            skipped.append(f"{seg_dir}  ({e})")

    if not samples:
        print("no usable samples found")
        return 1

    write_manifest(samples, args.out)

    by_sp: dict[str, list[HeadingSample]] = defaultdict(list)
    for s in samples:
        by_sp[s.species].append(s)

    print(f"wrote {len(samples)} samples -> {args.out}\n")
    print(f"{'species':10s} {'samples':>8s} {'tracks':>7s} {'median px':>10s}")
    for sp, v in sorted(by_sp.items(), key=lambda x: -len(x[1])):
        tracks = len({(s.segment, s.track_id) for s in v})
        print(f"{sp:10s} {len(v):8d} {tracks:7d} {np.median([s.body_px for s in v]):10.0f}")

    flips, jumps, pairs = _temporal_report(samples)
    print(f"\nTEMPORAL SANITY ({pairs} adjacent frame-pairs):")
    print(f"  human FACE-ID flips        : {flips:5.1f}%   <- the raw PCA-sign chaos")
    print(f"  our heading-angle jumps>90 : {jumps:5.1f}%   <- must be ~0 if the conversion is right")
    if jumps > 5.0:
        print("  *** WARNING: the angle is as unstable as the face id -- conversion is WRONG ***")

    with_mask = sum(s.mask_path is not None for s in samples)
    print(f"\nSAM3 mask available for {with_mask}/{len(samples)} samples")
    if skipped:
        print(f"\nskipped {len(skipped)} annotation files (non-standard layout / missing geometry):")
        for s in skipped[:5]:
            print(f"  {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
