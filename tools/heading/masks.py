"""SAM3 instance masks, loaded straight from the archive at feature time.  [CLUSTER]

    <archive>/<dataset>/WildBox_sam3-vggtv1_processed_unzipped/WildBox/<video>/<seg>/
        sam3_masks/masks/obj_<track_id>/frame_%06d.png     1920x1080, mode L, binary {0,255}

WHY WE NEED THEM
----------------
Zebras are the herd species. Their crops contain several OVERLAPPING zebras, and an
appearance-only foreground mask cannot tell which one is the target -- it pools a NEIGHBOUR'S
RUMP into the target's HEAD bin. Measured: zebra head/tail sat at 52%, i.e. exactly chance, while
giraffe (never occluded) hit 100%. Zebra is our flagship re-ID species, so this is the failure
that matters most.

WHY THERE IS NO PACK-AND-SHIP STEP
-----------------------------------
An earlier version packed the masks into their own .npz on the cluster, shipped it back, and
merged it into crops.npz. That was over-built. The masks are ALREADY TRACKED (`obj_<track_id>/` is
per-track), crops.npz already carries video/seg/track/image_name, and the sweep already runs on
the cluster -- where the masks are. So we just open the PNG when we need it.

The one thing that must travel with the crop is its EXACT crop box, so the mask is cut identically
to the JPEG it accompanies. `crops.npz` stores `crop_box = (ox, oy, side)` in full-res pixels;
nothing is recomputed here, so nothing can drift.

THE JOIN IS TRIVIAL -- WHICH IS EXACTLY WHY IT IS ASSERTED
-----------------------------------------------------------
`obj_<N>` is *supposed* to be track `N` and `frame_%06d.png` is *supposed* to be the same frame as
`frame_%06d.jpg`. Both look obviously right. But a wrong track pairing, or a one-frame offset,
hands us the NEIGHBOUR'S mask -- the precise bug this module exists to prevent -- and would still
produce a completely plausible number. So `check_join()` verifies the mask centroid falls inside
the target's own 2D box, and callers fail loudly on a systematic mismatch.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

__all__ = ["ARCHIVE", "build_index", "load_crop_mask", "check_join"]

ARCHIVE = Path("/storage3/3DOM/vshukla/sam3/wd_data/wildbox/archive")
EXCLUDE = ("Gazelle", "gazelle")


@lru_cache(maxsize=1)
def build_index(archive: str = str(ARCHIVE)) -> dict[str, Path]:
    """video name -> its segment-parent dir in the archive. Scanned, never hardcoded.

    KEYED OFF `sam3_masks` ITSELF, NOT OFF A GUESSED DIRECTORY LAYOUT.
    An earlier version assumed every dataset nests as `<dataset>/*/WildBox/<video>/`. The elephant,
    giraffe and zebra trees do -- the rhino trees do not, so 78% of rhino crops silently reported
    "no_video" and fell back to the appearance mask. The data was fine; the glob was too rigid.

    `sam3_masks` is always at `<video>/<seg>/sam3_masks`, whatever sits above it. So we find those
    and walk up two levels. That is a fact about what a segment IS, rather than a guess about how
    someone happened to lay out a directory.
    """
    root = Path(archive)
    if not root.is_dir():
        return {}                                   # not on this machine (masks are cluster-only)

    idx: dict[str, Path] = {}
    dupes: dict[str, list[Path]] = {}
    for ds in sorted(root.iterdir()):
        if not ds.is_dir() or any(e in ds.name for e in EXCLUDE):
            continue
        for sm in ds.rglob("sam3_masks"):
            if not sm.is_dir():
                continue
            vid = sm.parent.parent                  # <video>/<seg>/sam3_masks
            prev = idx.get(vid.name)
            if prev is not None and prev != vid:
                dupes.setdefault(vid.name, [prev]).append(vid)
            idx[vid.name] = vid

    if dupes:
        raise ValueError(
            f"{len(dupes)} video name(s) resolve to more than one directory, e.g. "
            f"{next(iter(dupes.items()))}. The video->segment join is not unique, and a wrong "
            f"resolution would hand us another animal's mask entirely. Resolve it before trusting "
            f"any mask.")
    return idx


def _path(video: str, seg: str, track: str, image_name: str) -> Path | None:
    vdir = build_index().get(str(video))
    if vdir is None:
        return None
    p = vdir / seg / "sam3_masks" / "masks" / f"obj_{track}" / (Path(image_name).stem + ".png")
    return p if p.is_file() else None


def check_join(mask: np.ndarray, box_xyxy, *, tol: float = 0.6) -> tuple[bool, float]:
    """Does this mask actually belong to the animal in `box_xyxy` (full-res x1 y1 x2 y2)?

    Checks the mask centroid against the target's own 2D box, with slack (a SAM mask includes the
    tail/trunk, so the centroid can sit off the box centre -- but it cannot land on a DIFFERENT
    animal). Returns (ok, normalised_offset), where 1.0 is the box edge.
    """
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return False, float("inf")
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    hx, hy = max((x2 - x1) / 2, 1e-6), max((y2 - y1) / 2, 1e-6)
    off = max(abs(float(xs.mean()) - (x1 + x2) / 2) / hx,
              abs(float(ys.mean()) - (y1 + y2) / 2) / hy)
    return off <= (1.0 + tol), off


def load_crop_mask(video: str, seg: str, track: str, image_name: str,
                   crop_box, size: int, *, verify_box=None) -> np.ndarray | None:
    """The instance mask, cut with the SAME square crop as the image, at `size` x `size`.

    `crop_box` is (ox, oy, side) in full-res pixels -- the exact square `extract_crops` used, read
    straight out of crops.npz. Nothing is recomputed, so the mask cannot drift out of alignment
    with the pixels the network sees.

    If `verify_box` (the animal's full-res 2D box) is given, the join is checked and a mask that
    belongs to a different animal is REJECTED rather than silently returned.
    """
    p = _path(video, seg, track, image_name)
    if p is None:
        return None

    from PIL import Image

    m = np.asarray(Image.open(p)) > 127
    if verify_box is not None:
        ok, _ = check_join(m, verify_box)
        if not ok:
            return None                             # a neighbour's mask: worse than no mask at all

    ox, oy, side = (float(v) for v in crop_box)
    H, W = m.shape
    x0, y0 = int(round(ox)), int(round(oy))
    x1, y1 = int(round(ox + side)), int(round(oy + side))

    out = np.zeros((y1 - y0, x1 - x0), dtype=bool)   # zero-pad past the frame edge, like the image
    ix0, iy0 = max(x0, 0), max(y0, 0)
    ix1, iy1 = min(x1, W), min(y1, H)
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    out[iy0 - y0: iy1 - y0, ix0 - x0: ix1 - x0] = m[iy0:iy1, ix0:ix1]

    return np.asarray(Image.fromarray(out).resize((size, size), Image.NEAREST))
