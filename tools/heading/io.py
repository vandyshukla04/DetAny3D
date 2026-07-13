"""Loaders for the WildBox / VGGT segment artefacts.

Everything the heading pipeline reads comes through here, so the on-disk layout is
described in exactly one place.

PROCESSED SEGMENT LAYOUT (the canonical one -- what the cluster has, and what the
human face-locks are keyed to)::

    <seg>/
      frame_NNNNNN.jpg                     frames (at the segment root)
      sam3_masks/masks/                    SAM3 per-object masks  (DINOv3 crops)
      vggt_results/
        cameras.json                       per-frame extrinsic 3x4 / intrinsic 3x3
        tracking_summary.json              RAW rotations  (<- what WildBox was built from)
        depth_maps.npz                     VGGT depth (no UniDepth needed)
        annotations/
          tracking_summary.json            CANONICALISED rotations
          semantic_faces/track_face_locks.json    human {front,top,left}  [validation GT]

TWO ROTATION SETS -- DO NOT MIX THEM
------------------------------------
* `raw`       = vggt_results/tracking_summary.json. Per-frame PCA/SVD signs, arbitrary
                and unstable (12.4% of consecutive frames flip an axis). **WildBox's
                `R_cam` was built from these** (prepare_wildbox_dataset.py:207).
* `canonical` = vggt_results/annotations/tracking_summary.json. Sign-aligned by the
                human annotator. **The face-locks index into THESE.**

Scoring our output against the locks with `raw` rotations would be meaningless, so the
caller must state which it wants; there is no silent default fallback.

A flatter layout also exists (`papersubdata/`: cameras.json + tracking_summary.json at
the segment root, no annotations). It is supported for convenience, but it carries no
face-locks and so cannot be used for validation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

__all__ = [
    "Track", "Camera", "Segment", "Rotations",
    "load_segment", "find_face_locks", "load_face_locks", "segment_dir_for_locks",
]

Rotations = Literal["raw", "canonical"]

# Bounded (non-recursive) glob: the lock path has FIXED depth, and a `**` scan of the
# NTFS mount takes minutes (4s vs minutes). Never use `**` here.
FACE_LOCK_RELGLOB = (
    "Data/WildBox/data/*/WildBox_sam3-vggtv1_processed/WildBox/*/*/"
    "vggt_results/annotations/semantic_faces/track_face_locks.json"
)


def _resolve(seg: Path, *candidates: str) -> Path:
    """First existing candidate, else fail loudly naming everything we tried."""
    for rel in candidates:
        p = seg / rel
        if p.exists():
            return p
    raise FileNotFoundError(
        f"none of {list(candidates)} found under {seg}. "
        "Is this a VGGT-processed segment?"
    )


@dataclass(frozen=True)
class Track:
    """One animal across a segment. All 3D quantities are in VGGT WORLD coords."""
    track_id: str
    class_name: str
    frames: np.ndarray            # (T,)      frame indices
    centers: np.ndarray           # (T, 3)
    dimensions: np.ndarray        # (T, 3)    extents along the box's local X, Y, Z
    rotations: np.ndarray         # (T, 3, 3) COLUMNS are the box's local axes
    bbox_2d: np.ndarray           # (T, 4)    image-space box (GDino / DINOv3 crops)

    def __len__(self) -> int:
        return len(self.frames)

    def index_of_frame(self, frame_index: int) -> int:
        hit = np.flatnonzero(self.frames == frame_index)
        if not len(hit):
            raise KeyError(f"track {self.track_id} has no frame {frame_index}")
        return int(hit[0])


@dataclass(frozen=True)
class Camera:
    frame_index: int
    extrinsic: np.ndarray         # (3, 4)  world -> camera
    intrinsic: np.ndarray         # (3, 3)
    image_name: str
    width: int
    height: int

    def world_to_cam(self, pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
        return pts @ self.extrinsic[:, :3].T + self.extrinsic[:, 3]

    def rotate_world_to_cam(self, vecs: np.ndarray) -> np.ndarray:
        """Directions (not points): rotation only, no translation."""
        return np.atleast_2d(np.asarray(vecs, dtype=np.float64)) @ self.extrinsic[:, :3].T

    def project(self, pts_cam: np.ndarray) -> np.ndarray:
        """Camera-space points -> pixels. Callers must handle points behind the camera."""
        pts_cam = np.atleast_2d(np.asarray(pts_cam, dtype=np.float64))
        z = np.where(np.abs(pts_cam[:, 2]) < 1e-9, 1e-9, pts_cam[:, 2])
        uv = pts_cam[:, :2] / z[:, None]
        return uv @ self.intrinsic[:2, :2].T + self.intrinsic[:2, 2]


@dataclass(frozen=True)
class Segment:
    path: Path
    rotations: Rotations          # which rotation set these tracks came from
    tracks: dict[str, Track]
    cameras: dict[int, Camera]    # keyed by frame_index

    @property
    def name(self) -> str:
        return f"{self.path.parent.name}/{self.path.name}"

    def frame_path(self, frame_index: int) -> Path:
        return self.path / self.cameras[frame_index].image_name

    @property
    def masks_dir(self) -> Path:
        return self.path / "sam3_masks" / "masks"

    @property
    def depth_npz(self) -> Path:
        return self.path / "vggt_results" / "depth_maps.npz"


def load_segment(seg_dir: str | Path, rotations: Rotations = "canonical") -> Segment:
    """Load a segment.

    `rotations="canonical"` (default) reads the annotator's sign-aligned rotations --
    the ones the face-locks index into, so this is what validation must use.
    `rotations="raw"` reads the per-frame PCA rotations that WildBox was built from.
    """
    seg = Path(seg_dir)
    if rotations == "canonical":
        summary_path = _resolve(seg, "vggt_results/annotations/tracking_summary.json")
    elif rotations == "raw":
        summary_path = _resolve(seg, "vggt_results/tracking_summary.json", "tracking_summary.json")
    else:  # pragma: no cover - guarded by the Literal
        raise ValueError(f"rotations must be 'raw' or 'canonical', got {rotations!r}")

    cams_path = _resolve(seg, "vggt_results/cameras.json", "cameras.json")

    summary = json.loads(summary_path.read_text())
    cams_raw = json.loads(cams_path.read_text())["cameras"]

    tracks = {
        tid: Track(
            track_id=tid,
            class_name=t.get("class_name", "object"),
            frames=np.asarray(t["frames"], dtype=int),
            centers=np.asarray(t["centers"], dtype=np.float64),
            dimensions=np.asarray(t["dimensions"], dtype=np.float64),
            rotations=np.asarray(t["rotation_matrices"], dtype=np.float64),
            bbox_2d=np.asarray(t["bbox_2d"], dtype=np.float64),
        )
        for tid, t in summary["tracks"].items()
    }
    cameras = {
        int(c["frame_index"]): Camera(
            frame_index=int(c["frame_index"]),
            extrinsic=np.asarray(c["extrinsic"], dtype=np.float64),
            intrinsic=np.asarray(c["intrinsic"], dtype=np.float64),
            image_name=c["image_name"],
            width=int(c["image_width"]),
            height=int(c["image_height"]),
        )
        for c in cams_raw
    }
    return Segment(path=seg, rotations=rotations, tracks=tracks, cameras=cameras)


def find_face_locks(root: str | Path) -> list[Path]:
    """Human face-lock files under `root` (e.g. /mnt/d/3DBOX). Bounded glob -- fast."""
    return sorted(p for p in Path(root).glob(FACE_LOCK_RELGLOB) if "RECYCLE" not in str(p))


def load_face_locks(path: str | Path) -> dict[str, dict[str, int]]:
    """{track_id: {front, top, left}} -- only tracks a human actually locked."""
    raw = json.loads(Path(path).read_text())
    return {
        tid: {k: int(v) for k, v in lock.items()}
        for tid, lock in raw.items()
        if {"front", "top", "left"} <= set(lock)
    }


def segment_dir_for_locks(lock_path: str | Path) -> Path:
    """<seg>/vggt_results/annotations/semantic_faces/track_face_locks.json -> <seg>"""
    return Path(lock_path).parents[3]
