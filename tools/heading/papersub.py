"""Loader for the paper-submission dataset (`papersubdata`) -- the dataset of record.

    /mnt/d/3DBOX/papersubdata/<group>/<video>/<seg>/
        frame_*.jpg           1920x1080
        cameras.json          per-frame world->cam extrinsic + FULL-RES intrinsics
        tracking_summary.json per-frame WORLD centers / dimensions / rotation_matrices
        kitti_labels/         per-frame 2D+3D boxes (full-res) -- used only to cross-check

Groups: elep{1,2,3} rhin{1,2} zebr{1,2,3} gira{1,2}.  `gaze1` is EXCLUDED by default.

THE SCALE TRAP -- READ THIS BEFORE TOUCHING 2D COORDINATES
----------------------------------------------------------
`cameras.json` intrinsics are FULL-RES (fx=1266, cx=960, cy=540 for 1920x1080), but
`tracking_summary.json`'s `bbox_2d` is in VGGT's **518-space** (x3.707 to reach full-res).
Mixing them cut every crop from the background last time and produced a confident,
meaningless result on animals the model had never seen.

So this loader **never reads `bbox_2d` for anything real**. The crop box is obtained by
projecting the 3D box's 8 corners with the (verified) full-res K -- exact, self-consistent,
and with no magic constant anywhere. `bbox_2d` survives only inside `check_scale()`, which
*measures* the ratio and fails loudly if it is not what we think.

COORDINATES (verified against kitti_labels: projected centre (870.8, 577.7) vs (870.7, 580.6))
    centers, rotation_matrices : WORLD
    extrinsic                  : 3x4, world -> camera
    scale                      : VGGT-arbitrary  => every threshold here is in BODY-LENGTHS,
                                 never metres.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from tools.heading.conventions import face_normals_world
from tools.heading.frame import horizontal_faces, sign_up_toward_camera, world_up_from_boxes

__all__ = [
    "Camera", "Track", "Segment",
    "SPECIES_OF_GROUP", "EXCLUDED_GROUPS",
    "load_segment", "iter_segments", "check_scale",
]

SPECIES_OF_GROUP = {"elep": "elephant", "rhin": "rhino", "zebr": "zebra", "gira": "giraffe"}
EXCLUDED_GROUPS = ("gaze",)          # gazelles -- excluded by the user
VGGT_LONG_SIDE = 518                 # VGGT runs at long-side 518; bbox_2d lives in that space


# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Camera:
    frame_index: int
    extrinsic: np.ndarray            # (3, 4) world -> camera
    K: np.ndarray                    # (3, 3) FULL-RES intrinsics
    width: int
    height: int
    image_name: str

    @cached_property
    def center(self) -> np.ndarray:
        """Camera centre, in WORLD coordinates."""
        R, t = self.extrinsic[:, :3], self.extrinsic[:, 3]
        return -R.T @ t

    def world_to_cam(self, pts: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pts, dtype=np.float64))
        return p @ self.extrinsic[:, :3].T + self.extrinsic[:, 3]

    def rotate_world_to_cam(self, v: np.ndarray) -> np.ndarray:
        """Rotate a DIRECTION (no translation)."""
        return np.atleast_2d(np.asarray(v, dtype=np.float64)) @ self.extrinsic[:, :3].T

    def project(self, pts_world: np.ndarray) -> np.ndarray:
        """WORLD points -> full-res pixels. Points behind the camera come back as NaN."""
        pc = self.world_to_cam(pts_world)
        z = pc[:, 2]
        uv = pc @ self.K.T
        with np.errstate(invalid="ignore", divide="ignore"):
            uv = uv[:, :2] / uv[:, 2:3]
        uv[z <= 1e-9] = np.nan
        return uv


@dataclass
class Track:
    tid: str
    frames: np.ndarray               # (T,)   frame indices into cameras
    centers: np.ndarray              # (T, 3) WORLD
    dims: np.ndarray                 # (T, 3)
    rotations: np.ndarray            # (T, 3, 3) WORLD
    bbox_2d: np.ndarray              # (T, 4) 518-SPACE -- do not use; cross-check only

    @cached_property
    def body_length(self) -> float:
        """Longest box extent, median over the track. The unit for every motion threshold."""
        return float(np.median(self.dims.max(axis=1)))

    def __len__(self) -> int:
        return len(self.frames)

    @cached_property
    def _frame_to_i(self) -> dict[int, int]:
        return {int(f): i for i, f in enumerate(self.frames)}

    def index_of_frame(self, fidx: int) -> int:
        return self._frame_to_i[int(fidx)]


@dataclass
class Segment:
    root: Path
    group: str                       # "zebr3"
    video: str                       # "DJI_20250802085130_0007_V"
    name: str                        # "seg3"
    tracks: dict[str, Track]
    cameras: dict[int, Camera]

    @property
    def species(self) -> str:
        return SPECIES_OF_GROUP[self.group[:4]]

    @property
    def key(self) -> str:
        """Unambiguous id. `seg1` alone is NOT unique across videos -- an earlier bug silently
        showed a different track than the one asked for."""
        return f"{self.group}/{self.video}/{self.name}"

    @cached_property
    def up_unsigned(self) -> np.ndarray:
        """Consensus ground normal from ALL box axes in the segment (axes only -- the signs
        are PCA junk). See frame.world_up_from_boxes."""
        return world_up_from_boxes(np.concatenate([t.rotations for t in self.tracks.values()]))

    def up_at(self, track: Track, i: int) -> np.ndarray:
        """The SIGNED world up for one instance: the drone is always above the animal."""
        cam = self.cameras[int(track.frames[i])]
        return sign_up_toward_camera(self.up_unsigned, track.centers[i], cam.center)

    def frame_path(self, fidx: int) -> Path:
        return self.root / self.cameras[int(fidx)].image_name

    # ---- the crop box -----------------------------------------------------------------
    @cached_property
    def scale(self) -> float:
        """bbox_2d (518-space) -> full-res pixels.

        The VALUE is exact and derived (`width / 518`); `check_scale()` independently MEASURES the
        ratio by projecting the 3D centres with the verified full-res K, and raises if the two
        disagree. So a segment that breaks the convention fails loudly instead of silently handing
        back a crop of the background.

        Value-derived rather than value-measured on purpose: the SAM-mask packer computes the same
        crop box on the cluster from the archive tree, and the two must agree EXACTLY. A measured
        median would differ in the last decimals between the two code paths and shift the crop --
        which, for a mask, means pairing a crop with a slightly misaligned instance.
        """
        check_scale(self)                                # assert the convention holds
        return next(iter(self.cameras.values())).width / VGGT_LONG_SIDE

    def crop_box(self, track: Track, i: int) -> np.ndarray:
        """Full-res 2D box (x1, y1, x2, y2). The detector's own `bbox_2d` -- tighter than a
        projected 3D hull, which inherits the box fit's slop -- rescaled by `self.scale`."""
        return np.asarray(track.bbox_2d[i], dtype=np.float64) * self.scale

    # ---- the WORLD ground frame: azimuths comparable across every frame of the segment ----
    @cached_property
    def ground_basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A deterministic (e1, e2, up) orthonormal frame for the ground plane of this segment.

        An azimuth measured in this basis means the same thing in EVERY frame of the segment --
        which is precisely what makes world-space temporal smoothing possible. The allocentric
        angle cannot do this: it is relative to the viewing ray, so a *stationary* animal's alpha
        changes as the drone moves. Smoothing alpha would be smoothing the drone's motion, which is
        exactly why the earlier image-space tracker made things WORSE (96.8% -> 79.0%).

        `up_unsigned` is signed canonically here (largest component positive) rather than toward the
        camera: a per-instance sign would flip `e2 = up x e1` and therefore flip the azimuth's sense
        halfway through a track.
        """
        up = np.asarray(self.up_unsigned, dtype=np.float64).copy()
        if up[int(np.argmax(np.abs(up)))] < 0:
            up = -up
        a = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = a - np.dot(a, up) * up
        e1 /= np.linalg.norm(e1)
        return e1, np.cross(up, e1), up

    def azimuth_of(self, direction: np.ndarray) -> float:
        """A WORLD direction -> its azimuth in the segment's ground basis. Comparable across frames."""
        e1, e2, up = self.ground_basis
        d = np.asarray(direction, dtype=np.float64)
        d = d - np.dot(d, up) * up
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            raise ValueError("direction is vertical; it has no azimuth")
        d = d / n
        return float(np.arctan2(np.dot(d, e2), np.dot(d, e1)))

    # ---- the allocentric (view-relative) frame ----------------------------------------
    def allocentric_basis(self, track: Track, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(r, s, up) in WORLD coords, where `r` is the horizontalised camera->animal ray
        and `s = up x r`.

        A crop determines the animal's orientation RELATIVE TO THE VIEWING RAY, not relative
        to the image: the same animal at the left and right edges of a frame looks identical
        but projects to different image angles. So the appearance-learnable quantity is the
        angle in this basis -- which is exactly DetAny3D's `alpha` (the allocentric /
        observation angle), the head that currently exists and is fed a hardcoded 0.0.
        """
        cam = self.cameras[int(track.frames[i])]
        up = self.up_at(track, i)
        r = track.centers[i] - cam.center
        r = r - np.dot(r, up) * up
        n = float(np.linalg.norm(r))
        if n < 1e-9:
            raise ValueError("animal is directly below the camera; the ray has no azimuth")
        r = r / n
        return r, np.cross(up, r), up

    def alpha_of(self, track: Track, i: int, direction: np.ndarray) -> float:
        """A WORLD direction -> its allocentric angle at this instance. Inverse: `dir_of_alpha`."""
        r, s, up = self.allocentric_basis(track, i)
        d = np.asarray(direction, dtype=np.float64)
        d = d - np.dot(d, up) * up
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            raise ValueError("direction is vertical; it has no allocentric angle")
        d = d / n
        return float(np.arctan2(np.dot(d, s), np.dot(d, r)))

    def dir_of_alpha(self, track: Track, i: int, alpha: float) -> np.ndarray:
        """An allocentric angle -> the WORLD direction it denotes. Inverse of `alpha_of`."""
        r, s, _ = self.allocentric_basis(track, i)
        return np.cos(alpha) * r + np.sin(alpha) * s

    def horizontal_face_dirs(self, track: Track, i: int) -> dict[int, np.ndarray]:
        """The 4 candidate FRONT faces -> their outward normals, flattened into the ground
        plane and unit-normalised. The animal's head is on exactly one of them (verified on
        11,084 human-locked instances: 100%)."""
        R, up = track.rotations[i], self.up_at(track, i)
        normals = face_normals_world(R)
        out = {}
        for f in horizontal_faces(R, up):
            d = normals[f] - np.dot(normals[f], up) * up
            n = float(np.linalg.norm(d))
            if n > 1e-9:
                out[int(f)] = d / n
        return out


# --------------------------------------------------------------------------------------
def load_segment(seg_dir: str | Path) -> Segment:
    seg_dir = Path(seg_dir)
    ts = json.loads((seg_dir / "tracking_summary.json").read_text())
    cm = json.loads((seg_dir / "cameras.json").read_text())

    cameras = {}
    for c in cm["cameras"]:
        E = np.asarray(c["extrinsic"], dtype=np.float64)
        if E.shape == (4, 4):
            E = E[:3]
        if E.shape != (3, 4):
            raise ValueError(f"{seg_dir}: extrinsic has shape {E.shape}, expected (3,4)")
        cameras[int(c["frame_index"])] = Camera(
            frame_index=int(c["frame_index"]),
            extrinsic=E,
            K=np.asarray(c["intrinsic"], dtype=np.float64),
            width=int(c["image_width"]),
            height=int(c["image_height"]),
            image_name=c["image_name"],
        )

    tracks = {}
    for tid, t in ts.get("tracks", {}).items():
        R = np.asarray(t["rotation_matrices"], dtype=np.float64)
        if R.ndim != 3 or R.shape[1:] != (3, 3):
            continue
        frames = np.asarray(t["frames"], dtype=int)
        keep = np.array([int(f) in cameras for f in frames])
        if keep.sum() < 2:
            continue
        tracks[str(tid)] = Track(
            tid=str(tid),
            frames=frames[keep],
            centers=np.asarray(t["centers"], dtype=np.float64)[keep],
            dims=np.asarray(t["dimensions"], dtype=np.float64)[keep],
            rotations=R[keep],
            bbox_2d=np.asarray(t["bbox_2d"], dtype=np.float64)[keep],
        )
    if not tracks:
        raise ValueError(f"{seg_dir}: no usable tracks")

    return Segment(
        root=seg_dir,
        group=seg_dir.parent.parent.name,
        video=seg_dir.parent.name,
        name=seg_dir.name,
        tracks=tracks,
        cameras=cameras,
    )


def iter_segments(root: str | Path, *, exclude_groups: tuple[str, ...] = EXCLUDED_GROUPS):
    """Yield every usable Segment under `root`, skipping excluded groups (gazelles)."""
    root = Path(root)
    for group in sorted(p for p in root.iterdir() if p.is_dir()):
        if group.name[:4] not in SPECIES_OF_GROUP or group.name[:4] in exclude_groups:
            continue
        for ts in sorted(group.glob("*/seg*/tracking_summary.json")):
            try:
                yield load_segment(ts.parent)
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                print(f"  skip {ts.parent}: {e}")


# --------------------------------------------------------------------------------------
def check_scale(seg: Segment, *, tol: float = 0.06) -> float:
    """MEASURE the bbox_2d -> full-res ratio by projecting the 3D centres, and verify it.

    This is the guard-rail for the bug that void'd the first experiment. We do not *use*
    the ratio anywhere (crops come from the projected 3D corners) -- we assert it, so that
    a future dataset with different conventions fails loudly here instead of silently
    cropping grass.

    Returns the measured ratio; raises if it disagrees with width/518.
    """
    ratios = []
    for tr in seg.tracks.values():
        for i in range(0, len(tr), max(1, len(tr) // 10)):
            cam = seg.cameras[int(tr.frames[i])]
            uv = cam.project(tr.centers[i][None])[0]
            b = tr.bbox_2d[i]
            bc = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
            if not np.isfinite(uv).all() or np.abs(bc).min() < 1e-6:
                continue
            ratios.append(uv / bc)
    if not ratios:
        raise ValueError(f"{seg.key}: could not measure the bbox_2d scale")

    got = float(np.median(np.asarray(ratios)))
    cam = next(iter(seg.cameras.values()))
    want = cam.width / VGGT_LONG_SIDE
    if abs(got - want) / want > tol:
        raise ValueError(
            f"{seg.key}: bbox_2d scale is {got:.3f} but width/{VGGT_LONG_SIDE} = {want:.3f}. "
            f"The 2D conventions of this segment are not what this loader assumes -- refusing "
            f"to continue rather than silently crop the wrong pixels."
        )
    return got
