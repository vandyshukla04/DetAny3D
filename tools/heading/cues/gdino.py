"""GroundingDINO head cue: zero-shot text prompt "head" picks the front face.

    crop the animal -> ground the word "head" -> project the 4 candidate face centres
    -> the face whose centre lands nearest the detected head is the FRONT.

WHY HF `transformers` AND NOT THE NATIVE `groundingdino`
-------------------------------------------------------
* The repo's `GroundingDINO` symlink is dead -- it points at the *original authors'*
  path (`/cpfs01/user/zhanghanxue/...`), not anything on this cluster.
* The native package needs compiled CUDA ops (MultiScaleDeformableAttention), so it
  cannot run on CPU -- which would kill the whole local dev/validate tier.
* We already depend on HF `transformers` for DINOv3, so this keeps one ecosystem, and
  `IDEA-Research/grounding-dino-tiny` is already in the local HF cache.

The model is loaded lazily and cached per-process, so a cluster shard pays for it once.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from tools.heading.cues.base import CueContext, projected_face_centers

__all__ = ["GDinoHeadCue"]

DEFAULT_MODEL = "IDEA-Research/grounding-dino-tiny"
# GroundingDINO wants a lowercase, period-terminated caption.
DEFAULT_PROMPT = "head."


@lru_cache(maxsize=2)
def _load(model_id: str, device: str):
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    return processor, model, torch


@lru_cache(maxsize=4)
def _box_threshold_kwarg(processor_cls: type) -> str:
    """`transformers` renamed GroundingDINO's `box_threshold` to `threshold`.

    Local is 4.46 (`box_threshold`); the cluster env may differ. Introspect rather than
    pin a version -- a silent TypeError here would look like "the cue never fires".
    """
    import inspect

    params = inspect.signature(
        processor_cls.post_process_grounded_object_detection
    ).parameters
    return "box_threshold" if "box_threshold" in params else "threshold"


def _threshold_kwargs(processor, box_threshold: float, text_threshold: float) -> dict:
    return {
        _box_threshold_kwarg(type(processor)): box_threshold,
        "text_threshold": text_threshold,
    }


@dataclass
class GDinoHeadCue:
    """Picks the front face by grounding the word "head" in the animal crop.

    Two failure modes are designed against, both observed on real data:

    * **Whole-animal grounding.** GroundingDINO's top "head" box is very often the entire
      crop (it grounds the noun onto the salient object). Measured on a zebra: the
      100%-of-crop box scored highest for *every* prompt tried. Any detection covering
      more than `max_area_frac` of the animal is therefore rejected as "that's the body".
    * **Distance-to-face-centre scoring.** Scoring a face by how near the head lands to
      its projected centre structurally favours the LATERAL faces, whose centres project
      close to the animal's 2D centre (they are foreshortened). We instead score by
      *direction from the body centre* -- which is what "where is the head?" actually
      asks -- making the score scale-free and unbiased between end and flank faces.
    """

    name: str = "gdino"
    model_id: str = DEFAULT_MODEL
    prompt: str = DEFAULT_PROMPT
    device: str = "cpu"
    box_threshold: float = 0.15
    text_threshold: float = 0.15
    pad: float = 0.15            # expand the crop; heads sit at the very edge of a box
    max_area_frac: float = 0.45  # reject "heads" bigger than this share of the crop
    min_offset_frac: float = 0.10  # head must sit off-centre by this much of the crop
    temperature: float = 0.25    # softmax temperature on the direction cosine

    # -- internals -------------------------------------------------------------
    def _detect_head(self, ctx: CueContext) -> tuple[np.ndarray, float] | None:
        """Best head detection, as (centre_xy in FULL-image pixels, confidence)."""
        from PIL import Image

        processor, model, torch = _load(self.model_id, self.device)

        x1, y1, x2, y2 = ctx.crop_box
        h, w = ctx.image.shape[:2]
        px, py = self.pad * (x2 - x1), self.pad * (y2 - y1)
        cx1, cy1 = int(max(0, x1 - px)), int(max(0, y1 - py))
        cx2, cy2 = int(min(w, x2 + px)), int(min(h, y2 + py))
        if cx2 - cx1 < 8 or cy2 - cy1 < 8:
            return None

        crop = Image.fromarray(ctx.image[cy1:cy2, cx1:cx2])
        inputs = processor(images=crop, text=self.prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)

        res = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            target_sizes=[crop.size[::-1]],          # (h, w)
            **_threshold_kwargs(processor, self.box_threshold, self.text_threshold),
        )[0]
        if not len(res["scores"]):
            return None

        boxes = res["boxes"].cpu().numpy()
        scores = res["scores"].cpu().numpy()
        cw, ch = crop.size
        crop_area = float(cw * ch)

        # Reject whole-animal grounding: a head is a PART, not the object.
        keep = []
        for b, s in zip(boxes, scores):
            area = max(b[2] - b[0], 0) * max(b[3] - b[1], 0)
            if area / crop_area > self.max_area_frac:
                continue
            # ...and it must sit off-centre, or it carries no directional information.
            off = np.hypot((b[0] + b[2]) / 2 - cw / 2, (b[1] + b[3]) / 2 - ch / 2)
            if off / max(np.hypot(cw, ch), 1.0) < self.min_offset_frac:
                continue
            keep.append((float(s), b))
        if not keep:
            return None

        conf, b = max(keep, key=lambda t: t[0])
        centre = np.array([cx1 + (b[0] + b[2]) / 2.0, cy1 + (b[1] + b[3]) / 2.0])
        return centre, conf

    # -- HeadCue ---------------------------------------------------------------
    def score(self, ctx: CueContext) -> dict[int, float]:
        hit = self._detect_head(ctx)
        if hit is None:
            return {}                                 # abstain: no head grounded
        head_xy, conf = hit

        proj = projected_face_centers(ctx)
        if len(proj) < 2:
            return {}                                 # nothing to discriminate between

        # DIRECTION from the body centre, not distance to a face centre. The lateral
        # faces project near the animal's 2D centre (foreshortened), so a distance metric
        # would hand them every detection that lands mid-body. Asking "in which direction
        # from the body centre does the head lie?" is both what we mean and unbiased.
        body_uv = ctx.camera.project(ctx.camera.world_to_cam(ctx.center))[0]
        head_dir = head_xy - body_uv
        n = float(np.linalg.norm(head_dir))
        if n < 1e-6:
            return {}                                 # head on the centroid: no direction
        head_dir /= n

        faces, cos = [], []
        for f, uv in proj.items():
            v = uv - body_uv
            m = float(np.linalg.norm(v))
            if m < 1e-6:
                continue                              # degenerate (face centre on centroid)
            faces.append(f)
            cos.append(float(np.dot(head_dir, v / m)))
        if len(faces) < 2:
            return {}

        logits = np.array(cos) / max(self.temperature, 1e-6)
        p = np.exp(logits - logits.max())
        p /= p.sum()

        # Weight by the detector's own confidence, so a weak grounding is a weak vote
        # rather than a loud guess.
        return {f: float(conf * p[i]) for i, f in enumerate(faces)}
