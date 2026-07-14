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
    """Crops + their SAM instance masks.

    MASKS ARE CACHED AND PREFETCHED, and both matter. Each mask is a 1920x1080 PNG on network
    storage (~50-200 ms to open), the sweep re-reads the SAME crops on every facet/size pass, and
    the work is pure I/O. Naively that is ~12,000 blocking NFS reads -- tens of minutes before a
    single GPU op, and it looks like a hang. Cached + threaded it is a minute, once.

    Cached masks are stored PACKED (1 bit/pixel): 12k crops at 224x224 is ~80 MB instead of 640 MB.
    """

    def __init__(self, path, *, mask_size: int = 224, verify_masks: bool = True):
        self.d = np.load(path, allow_pickle=True)
        self.mask_size = mask_size
        self.verify = verify_masks
        self._cache: dict[int, np.ndarray | None] = {}

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
        got = [v for v in self._cache.values() if v is not None]
        return len(got) / max(len(self._cache), 1)

    def _load_mask(self, i: int):
        from tools.heading.masks import load_crop_mask

        d = self.d
        m = load_crop_mask(
            str(d["video"][i]), str(d["seg_name"][i]), str(d["track"][i]).split("::")[-1],
            str(d["image_name"][i]), d["crop_box"][i], self.mask_size,
            verify_box=d["box2d"][i] if self.verify else None,
        )
        return None if m is None else np.packbits(m)

    def prefetch(self, idx, workers: int = 16, label: str = "") -> None:
        """Load these masks in parallel, once. Pure I/O, so threads scale well over NFS."""
        if not self.can_mask:
            return
        todo = [int(i) for i in idx if int(i) not in self._cache]
        if not todo:
            return

        from concurrent.futures import ThreadPoolExecutor

        print(f"  prefetching {len(todo)} SAM masks{' ' + label if label else ''} "
              f"({workers} threads) ...", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, m in zip(todo, pool.map(self._load_mask, todo)):
                self._cache[i] = m
        hit = sum(v is not None for v in self._cache.values())
        print(f"  masks ready: {hit}/{len(self._cache)} "
              f"({100*hit/max(len(self._cache),1):.0f}% -- the rest fall back to the "
              f"appearance mask)", flush=True)

    def _mask(self, i: int):
        if not self.can_mask:
            return None
        if i not in self._cache:
            self._cache[i] = self._load_mask(i)
        p = self._cache[i]
        if p is None:
            return None
        m = self.mask_size
        return np.unpackbits(p)[: m * m].reshape(m, m).astype(bool)

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
