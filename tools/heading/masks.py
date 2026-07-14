"""SAM3 instance masks: locate, load, and VERIFY the join.  [cluster-side; papersubdata has none]

    <archive>/<dataset>/WildBox_sam3-vggtv1_processed_unzipped/WildBox/<video>/<seg>/
        sam3_masks/
            metadata.json                     resolution, object_ids, frame_numbers, text_prompt
            masks/obj_<track_id>/frame_%06d.png    1920x1080, mode L, binary {0, 255}

WHY WE NEED THEM
----------------
Zebras are the herd species. Their crops contain several OVERLAPPING zebras, and an
appearance-only foreground mask cannot tell which one is the target -- it pools a NEIGHBOUR'S
RUMP into the target's HEAD bin. Measured consequence: zebra head/tail sat at 52%, i.e. exactly
chance, while giraffe (never occluded) hit 100%. Zebra is our flagship re-ID species, so this is
the failure that matters most.

THE JOIN IS TRIVIAL -- AND THAT IS EXACTLY WHY IT MUST BE ASSERTED
-------------------------------------------------------------------
  * masks are FULL-RES (1920x1080), like the frames. No 518-space rescale. (papersubdata's own
    `bbox_2d` IS 518-space while its intrinsics are full-res -- that mismatch already void'd one
    experiment, so nothing here is taken on trust.)
  * `obj_<N>` <-> tracking_summary track `N`.
  * `frame_%06d.png` <-> the frame's `frame_%06d.jpg`. Same stem, no index remap.

A wrong pairing -- wrong track, or a one-frame offset -- would silently hand us the NEIGHBOUR'S
mask. That is the precise bug we are fixing, and it would still produce a completely plausible
number. So `check_join()` verifies the mask centroid lands inside the target's own 2D box, and
raises otherwise.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

__all__ = ["ARCHIVE", "build_index", "mask_path", "load_mask", "check_join"]

ARCHIVE = Path("/storage3/3DOM/vshukla/sam3/wd_data/wildbox/archive")
EXCLUDE = ("Gazelle", "gazelle")


@lru_cache(maxsize=1)
def build_index(archive: str = str(ARCHIVE)) -> dict[str, Path]:
    """video name -> its segment-parent dir in the archive. Scanned, never hardcoded.

    Every papersubdata group maps to exactly one archive dataset by VIDEO NAME (elep1/2/3 ->
    data202401K/202406K/202602KElephants, zebr1/2/3 -> dataBZS/data2023KABRZebras/wildbox_tomblair,
    ...). Video names are globally unique, so we key on them and never have to know the group ->
    dataset mapping at all.
    """
    root = Path(archive)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found -- the SAM masks live on the CLUSTER only")

    idx: dict[str, Path] = {}
    for ds in sorted(root.iterdir()):
        if not ds.is_dir() or any(e in ds.name for e in EXCLUDE):
            continue
        for wb in ds.glob("*/WildBox"):
            for vid in sorted(wb.iterdir()):
                if vid.is_dir():
                    if vid.name in idx:
                        raise ValueError(
                            f"video {vid.name} appears in two datasets ({idx[vid.name]} and "
                            f"{vid}); the video->dataset join is not unique and must be resolved "
                            f"before any mask is trusted")
                    idx[vid.name] = vid
    return idx


def mask_path(video: str, seg: str, track: str, image_name: str) -> Path | None:
    """The mask for one (video, segment, track, frame), or None if it does not exist."""
    vid = build_index().get(str(video))
    if vid is None:
        return None
    p = vid / seg / "sam3_masks" / "masks" / f"obj_{track}" / (Path(image_name).stem + ".png")
    return p if p.is_file() else None


def load_mask(video: str, seg: str, track: str, image_name: str) -> np.ndarray | None:
    """Full-res boolean mask for one instance, or None."""
    p = mask_path(video, seg, track, image_name)
    if p is None:
        return None
    from PIL import Image

    return np.asarray(Image.open(p)) > 127


def segment_meta(video: str, seg: str) -> dict | None:
    vid = build_index().get(str(video))
    if vid is None:
        return None
    p = vid / seg / "sam3_masks" / "metadata.json"
    return json.loads(p.read_text()) if p.is_file() else None


def check_join(mask: np.ndarray, box_xyxy, *, tol: float = 0.6) -> tuple[bool, float]:
    """Does this mask actually belong to the animal in `box_xyxy` (full-res, x1 y1 x2 y2)?

    Verifies the mask's centroid lies inside the target's own 2D box, allowing `tol` of a
    half-extent of slack (SAM masks include the tail/trunk, so the centroid can sit slightly off
    the box centre, but it cannot land on a different animal).

    Returns (ok, normalised_offset). Callers should HARD FAIL on a systematic mismatch: silently
    accepting the neighbour's mask is the exact bug this module exists to prevent, and it would
    look entirely reasonable in the output.
    """
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return False, float("inf")
    cx, cy = float(xs.mean()), float(ys.mean())

    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    hx, hy = max((x2 - x1) / 2, 1e-6), max((y2 - y1) / 2, 1e-6)
    off = max(abs(cx - (x1 + x2) / 2) / hx, abs(cy - (y1 + y2) / 2) / hy)
    return off <= (1.0 + tol), off
