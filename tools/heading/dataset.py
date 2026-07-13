"""Turn the human face annotations into heading-head training data.

WHAT THIS SOLVES
----------------
The human annotators labelled which BOX FACE is the animal's front. We cannot train on
a face id directly: face ids are tied to the box's rotation, whose PCA/SVD signs flip
from frame to frame (measured: 12.4% of consecutive frames flip an axis). A model asked
to predict "face 2" would be chasing a target that randomly renames itself.

So we convert each label into something rotation-sign-free and directly learnable:

    the animal's HEADING DIRECTION, as an angle in IMAGE SPACE

i.e. the direction, in the crop, from the animal's centre toward its head. That is a
property of the picture, not of the box's bookkeeping, so it is stable, and a network
looking at the crop can actually see it.

    GT angle  =  atan2( project(front_face_centre) - project(box_centre) )

At inference we go back the other way: predict the angle, then pick whichever of the 4
horizontal candidate faces projects closest to it, and hand that to `frame.py`, which
turns it into front/back/left/right/top (verified exact) and hence the re-ID flank tag.

The angle is emitted in CROP-RELATIVE coordinates (the crop is what the network sees),
alongside everything needed to render or re-derive the geometry.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from tools.heading.conventions import face_centers_world
from tools.heading.io import Segment, load_segment

__all__ = ["HeadingSample", "build_samples_for_segment", "heading_angle_deg"]

SPECIES = ("giraffe", "elephant", "rhino", "zebra", "gazelle")


@dataclass(frozen=True)
class HeadingSample:
    """One training instance: a crop, and the heading angle we want predicted."""
    segment: str            # path to the segment (so features can be re-extracted)
    species: str
    track_id: str
    frame_index: int
    image_name: str
    bbox_2d: list[float]    # animal box in FULL-image pixels (x1, y1, x2, y2)
    mask_path: str | None   # SAM3 mask for this (track, frame), if present
    heading_deg: float      # GT heading, degrees CCW from +x, in image space
    front_face: int         # the human label, kept for auditing/geometry checks
    body_px: float          # animal size on screen (max side) -- for size-stratified eval


def heading_angle_deg(seg: Segment, track_id: str, i: int, front_face: int) -> float | None:
    """Image-space angle from the animal's centre to its front-face centre.

    Returns None when the geometry is degenerate (box behind the camera, or the front
    face projects onto the centroid so the direction is undefined) -- such a sample
    carries no learnable signal and must be dropped rather than encoded as angle 0.
    """
    tr = seg.tracks[track_id]
    cam = seg.cameras.get(int(tr.frames[i]))
    if cam is None:
        return None

    centre_cam = cam.world_to_cam(tr.centers[i])[0]
    face_w = face_centers_world(tr.centers[i], tr.dimensions[i], tr.rotations[i])[int(front_face)]
    face_cam = cam.world_to_cam(face_w)[0]
    if centre_cam[2] <= 1e-6 or face_cam[2] <= 1e-6:
        return None                                     # behind the camera

    d = cam.project(face_cam)[0] - cam.project(centre_cam)[0]
    if float(np.linalg.norm(d)) < 1e-3:
        return None                                     # no direction to learn
    return float(np.degrees(np.arctan2(d[1], d[0])))


def _species_of(path: Path) -> str:
    low = str(path).lower()
    return next((s for s in SPECIES if s in low), "unknown")


def build_samples_for_segment(
    seg_dir: str | Path, gt_front: dict[tuple[str, int], int], *, rotations="canonical"
) -> list[HeadingSample]:
    """`gt_front`: {(track_id, frame_index) -> human front-face id}."""
    seg_dir = Path(seg_dir)
    seg = load_segment(seg_dir, rotations=rotations)
    species = _species_of(seg_dir)

    out: list[HeadingSample] = []
    for (tid, fidx), front in gt_front.items():
        tr = seg.tracks.get(tid)
        cam = seg.cameras.get(int(fidx))
        if tr is None or cam is None:
            continue
        try:
            i = tr.index_of_frame(int(fidx))
        except KeyError:
            continue

        ang = heading_angle_deg(seg, tid, i, front)
        if ang is None:
            continue

        mask = seg.masks_dir / f"obj_{tid}" / f"frame_{int(fidx):06d}.png"
        x1, y1, x2, y2 = (float(v) for v in tr.bbox_2d[i])
        out.append(
            HeadingSample(
                segment=str(seg_dir),
                species=species,
                track_id=tid,
                frame_index=int(fidx),
                image_name=cam.image_name,
                bbox_2d=[x1, y1, x2, y2],
                mask_path=str(mask) if mask.is_file() else None,
                heading_deg=ang,
                front_face=int(front),
                body_px=float(max(x2 - x1, y2 - y1)),
            )
        )
    return out


def write_manifest(samples: list[HeadingSample], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([asdict(s) for s in samples], indent=1))
    return out_path
