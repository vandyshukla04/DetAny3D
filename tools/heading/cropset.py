"""One accessor for crops.npz. Every consumer reads it through here.

`sweep.py`, `desc_viz.py`, `tracklet.py` and `evaluate.py` all need the same things out of
crops.npz: the image, the instance mask (unpacked), the 4 candidate face centres, the geometric
axis, and the motion-derived truth. Doing that lookup three times in three files is how the
consumers silently drift apart -- and each one would still print a plausible number.

The instance mask in particular is easy to get subtly wrong: it is stored PACKED (np.packbits), an
absent mask is NOT an empty mask (an empty foreground would zero the animal instead of falling
back to the appearance mask), and it must be unpacked to its own square resolution before being
resampled onto the patch grid.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np

__all__ = ["CropSet", "Item"]


@dataclass(frozen=True)
class Item:
    i: int
    image: np.ndarray                 # (H, W, 3) uint8
    instance: np.ndarray | None       # (m, m) bool -- the SAM mask, or None
    face_uv: np.ndarray               # (4, 2) in [0,1] crop coords
    face_ids: np.ndarray              # (4,) real face ids
    y_face: int                       # the TRUE head slot, from motion
    geo_axis: int                     # the slot on the longest horizontal box axis
    species: str
    video: str
    track: str
    frame: int


class CropSet:
    def __init__(self, path):
        self.d = np.load(path, allow_pickle=True)
        self.has_masks = "mask" in self.d
        self.mask_size = int(self.d["mask_size"][0]) if self.has_masks else 0

    def __len__(self) -> int:
        return len(self.d["jpeg"])

    @property
    def species(self) -> np.ndarray:
        return self.d["species"]

    @property
    def video(self) -> np.ndarray:
        return self.d["video"]

    def get(self, i: int) -> Item:
        from PIL import Image

        d = self.d
        inst = None
        if self.has_masks and bool(d["has_mask"][i]):
            m = self.mask_size
            inst = np.unpackbits(d["mask"][i])[: m * m].reshape(m, m).astype(bool)

        return Item(
            i=int(i),
            image=np.asarray(Image.open(io.BytesIO(d["jpeg"][i])).convert("RGB")),
            instance=inst,
            face_uv=d["face_uv"][i],
            face_ids=d["face_ids"][i],
            y_face=int(d["y_face"][i]),
            geo_axis=int(d["geo_axis"][i]) if "geo_axis" in d else -1,
            species=str(d["species"][i]),
            video=str(d["video"][i]),
            track=str(d["track"][i]),
            frame=int(d["frame"][i]),
        )

    def batch(self, idx) -> list[Item]:
        return [self.get(int(i)) for i in idx]
