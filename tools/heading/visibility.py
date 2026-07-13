"""Which side of the animal is the camera seeing?  -- WITHOUT a 3D box.

    image heading angle  +  camera  +  gravity   ->   LEFT / RIGHT flank

WHY THIS EXISTS (the architectural point)
-----------------------------------------
The flank the camera sees is fully determined by `left = up x forward` and where the camera
is. **The 3D box contributes nothing to that.** In the earlier pipeline the box was only ever
a scaffold: we used its 4 horizontal faces to turn the predicted image-angle into a 3D
direction. But we can get that direction directly:

  * an animal's heading is HORIZONTAL (it walks on the ground), so given gravity there is a
    one-parameter family of candidate headings (azimuth phi);
  * exactly one of them projects to the angle the model predicted;
  * the animal's POSITION comes from intersecting the camera ray through its 2D box with the
    ground plane.

So the whole thing needs only: **a 2D box, the camera, and gravity.** No `R_cam`, no PCA
rotation, no 3D box -- which matters, because DetAny3D's rotation head is untrained and its
GT rotation has arbitrary PCA signs (see HEADING_PROJECT.md). We are no longer standing on
that sand.

THE PROJECTION OF A DIRECTION (why it is not just the raw angle)
---------------------------------------------------------------
A 3D direction `d` at a 3D point `p` does NOT project to a fixed image angle -- it depends on
where in the frame the animal is (perspective). Differentiating u = f*px/pz + cx:

    image_dir  ∝  ( d_x - (p_x/p_z) d_z ,  d_y - (p_y/p_z) d_z )

We invert that: sweep the azimuth, project, and take the one matching the predicted angle.
Ignoring this and treating the image angle as the world azimuth is wrong away from the
principal point -- which is most of the frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Flank", "ground_basis", "forward_from_image_angle", "visible_flank", "face_visibility"]


@dataclass(frozen=True)
class Flank:
    side: str                 # "LEFT" | "RIGHT"
    confidence: float         # |cos| between the flank normal and the view ray: 0 = edge-on
    forward_cam: np.ndarray   # the animal's heading, in CAMERA coords
    left_cam: np.ndarray      # the animal's LEFT direction, in CAMERA coords


def ground_basis(up_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal vectors spanning the ground plane (both perpendicular to `up_cam`)."""
    up = np.asarray(up_cam, dtype=np.float64)
    up = up / np.linalg.norm(up)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(seed, up))) > 0.9:
        seed = np.array([0.0, 0.0, 1.0])
    e1 = seed - np.dot(seed, up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    return e1, e2


def _project_direction(d: np.ndarray, p_cam: np.ndarray) -> np.ndarray:
    """Image-space direction of the 3D direction `d` located at camera-space point `p_cam`.

    Perspective: the same 3D direction points different ways in the image depending on where
    the object sits in the frame. Focal length cancels for a *direction* (up to aspect, which
    we have already normalised away in io.py).
    """
    pz = p_cam[2] if abs(p_cam[2]) > 1e-9 else 1e-9
    return np.array([d[0] - (p_cam[0] / pz) * d[2],
                     d[1] - (p_cam[1] / pz) * d[2]])


def forward_from_image_angle(
    image_angle: float, up_cam: np.ndarray, p_cam: np.ndarray, *, samples: int = 720
) -> np.ndarray:
    """The horizontal 3D heading (camera coords) whose projection matches `image_angle`.

    Sweeps the azimuth over the ground plane and picks the best match. 720 samples = 0.5 deg
    resolution, which is far below the model's ~4 deg error, so the discretisation is free.
    """
    e1, e2 = ground_basis(up_cam)
    phis = np.linspace(-np.pi, np.pi, samples, endpoint=False)
    dirs = np.cos(phis)[:, None] * e1 + np.sin(phis)[:, None] * e2      # (S, 3) horizontal

    proj = np.array([_project_direction(d, p_cam) for d in dirs])       # (S, 2)
    n = np.linalg.norm(proj, axis=1, keepdims=True)
    n[n < 1e-12] = 1e-12
    proj /= n

    target = np.array([np.cos(image_angle), np.sin(image_angle)])
    return dirs[int(np.argmax(proj @ target))]


def visible_flank(image_angle: float, up_cam: np.ndarray, p_cam: np.ndarray) -> Flank:
    """LEFT or RIGHT: which flank of the animal faces the camera.

    The camera sits at the origin in camera coords, so the ray from the animal to the camera
    is simply `-p_cam`. A face is visible when its outward normal points along that ray.
    """
    up = np.asarray(up_cam, dtype=np.float64)
    up = up / np.linalg.norm(up)
    p = np.asarray(p_cam, dtype=np.float64)

    fwd = forward_from_image_angle(image_angle, up, p)
    left = np.cross(up, fwd)                     # same rule verified 66/66 on the human locks
    left /= max(float(np.linalg.norm(left)), 1e-12)

    to_cam = -p / max(float(np.linalg.norm(p)), 1e-12)
    c = float(np.dot(left, to_cam))              # >0 => the LEFT flank faces the camera
    return Flank(
        side="LEFT" if c > 0 else "RIGHT",
        confidence=abs(c),                       # ~0 means edge-on: the flank call is unstable
        forward_cam=fwd,
        left_cam=left,
    )


def face_visibility(image_angle: float, up_cam: np.ndarray, p_cam: np.ndarray) -> dict[str, float]:
    """How much each named face faces the camera. Positive = visible; the magnitude is the
    cosine, i.e. how square-on it is. This is the re-ID viewpoint tag."""
    f = visible_flank(image_angle, up_cam, p_cam)
    up = np.asarray(up_cam, dtype=np.float64)
    up = up / np.linalg.norm(up)
    p = np.asarray(p_cam, dtype=np.float64)
    to_cam = -p / max(float(np.linalg.norm(p)), 1e-12)

    normals = {
        "front": f.forward_cam, "back": -f.forward_cam,
        "left": f.left_cam, "right": -f.left_cam,
        "top": up, "bottom": -up,
    }
    return {k: float(np.dot(v, to_cam)) for k, v in normals.items()}
