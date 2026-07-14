"""One accessor for crops.npz. Every consumer reads it through here.

`sweep.py`, `desc_viz.py`, `tracklet.py` and `evaluate.py` all need the same things: the image, the
SAM instance mask, the 4 candidate face centres, the geometric axis, and the motion-derived truth.
Doing that lookup four times in four files is how consumers silently drift apart -- and each one
would still print a plausible number.

THE MASK IS LOADED HERE, ON DEMAND, FROM THE ARCHIVE
-----------------------------------------------------
It is NOT packed into crops.npz. The masks are already tracked (`obj_<track_id>/`), crops.npz
already carries video/seg/track/image_name, and the GPU work runs on the cluster -- where the masks
already are. So we just open the PNG.

The one thing that must travel with each crop is its EXACT crop box (`crop_box = ox, oy, side`), so
the mask is cut identically to the JPEG beside it. Nothing is recomputed at load time, so the mask
cannot drift out of alignment with the pixels the network sees.

Missing mask != empty mask. A crop with no mask falls back to the appearance-based foreground; an
empty mask would zero out the animal entirely and silently produce a blank profile.
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
    instance: np.ndarray | None       # (m, m) bool -- the SAM mask, or None -> appearance fallback
    face_uv: np.ndarray               # (4, 2) in [0,1] crop coords
    face_ids: np.ndarray              # (4,) real face ids
    y_face: int                       # the TRUE head slot, from motion
    geo_axis: int                     # the slot on the longest horizontal box axis
    species: str
    video: str
    track: str
    frame: int


class CropSet:
    def __init__(self, path, *, mask_size: int = 224, verify_masks: bool = True):
        self.d = np.load(path, allow_pickle=True)
        self.mask_size = mask_size
        self.verify = verify_masks
        self._n_mask = 0
        self._n_try = 0

        from tools.heading.masks import build_index

        self.can_mask = ("crop_box" in self.d and "image_name" in self.d
                         and len(build_index()) > 0)

    def __len__(self) -> int:
        return len(self.d["jpeg"])

    @property
    def species(self) -> np.ndarray:
        return self.d["species"]

    @property
    def video(self) -> np.ndarray:
        return self.d["video"]

    @property
    def mask_rate(self) -> float:
        return self._n_mask / max(self._n_try, 1)

    def _mask(self, i: int):
        if not self.can_mask:
            return None
        from tools.heading.masks import load_crop_mask

        d = self.d
        self._n_try += 1
        m = load_crop_mask(
            str(d["video"][i]), str(d["seg_name"][i]), str(d["track"][i]).split("::")[-1],
            str(d["image_name"][i]), d["crop_box"][i], self.mask_size,
            verify_box=d["box2d"][i] if self.verify else None,
        )
        self._n_mask += int(m is not None)
        return m

    def get(self, i: int) -> Item:
        from PIL import Image

        d = self.d
        return Item(
            i=int(i),
            image=np.asarray(Image.open(io.BytesIO(d["jpeg"][i])).convert("RGB")),
            instance=self._mask(i),
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
