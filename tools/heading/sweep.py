"""Run the heading -> flank tagger over MANY unlabelled segments, and validate it WITHOUT labels.

    python -m tools.heading.sweep \
        --root /storage3/3DOM/vshukla/sam3/wd_data/wildbox/archive/data2023KABRZebras \
        --head data/heading/head.pt --out data/heading/sweep_zebra \
        --every 2 --device cuda

Emits, per segment:
  * `tags.json`  -- the re-ID payload: {track, frame} -> flank, confidence, face visibilities
and prints a stability report over every track.

HOW WE VALIDATE WITHOUT GROUND TRUTH  (this is the point)
---------------------------------------------------------
There are no face labels on these segments, so we cannot score accuracy. But physics gives us
a free check:

    the flank tag may ONLY switch when the animal passes through a head-on / tail-on view,
    i.e. exactly when the flank is edge-on to the camera and |cos| -> 0.

So:
  * a switch at LOW confidence  = CORRECT: the animal genuinely turned side-on and we crossed
    the boundary from seeing its left to seeing its right;
  * a switch at HIGH confidence = NONSENSE: the model is claiming the zebra teleported from
    showing one flank to the other while still square-on to the camera.

**Switches-at-high-confidence is therefore a label-free error detector.** A good tagger has
few switches, and the ones it has cluster at low confidence. A broken tagger flickers at high
confidence.

We also report the flank DUTY CYCLE (how much of the track shows each side), which is the
dataset-coverage answer re-ID actually cares about: "do we ever see this animal's right side?"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
EDGE_ON = 0.35          # |cos| below this = the flank is edge-on; a switch here is expected


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True,
                    help="dir to scan for segments (any depth)")
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--every", type=int, default=2, help="use every Nth frame")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-segments", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    from tools.heading.extract_crops import square_crop
    from tools.heading.frame import sign_up_toward_camera, world_up_from_boxes
    from tools.heading.io import load_segment
    from tools.heading.predict import load_head
    from tools.heading.visibility import face_visibility, visible_flank

    segs = sorted({p.parent.parent for p in args.root.rglob("vggt_results/tracking_summary.json")})
    if args.max_segments:
        segs = segs[: args.max_segments]
    if not segs:
        print(f"no segments under {args.root}")
        return 1
    print(f"{len(segs)} segments\n")

    net, mu, sd = load_head(args.head, args.device)
    proc = AutoImageProcessor.from_pretrained(args.model)
    dino = AutoModel.from_pretrained(args.model).to(args.device).eval()
    args.out.mkdir(parents=True, exist_ok=True)

    def embed(crops: list[Image.Image]) -> np.ndarray:
        out = []
        for i in range(0, len(crops), args.batch):
            inp = proc(images=crops[i: i + args.batch], return_tensors="pt").to(args.device)
            with torch.no_grad():
                h = dino(**inp).last_hidden_state
                out.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], -1).float().cpu().numpy())
        return np.concatenate(out)

    rows = []                                   # (segment, track, n, switches, bad_switches, duty)
    for si, sd_path in enumerate(segs, 1):
        try:
            seg = load_segment(sd_path, rotations="raw")   # WildBox's own rotations
        except (FileNotFoundError, KeyError) as e:
            print(f"[{si}/{len(segs)}] skip {sd_path.name}: {e}")
            continue

        # Only the AXES of the rotations are used (for the gravity consensus); the signs --
        # which are the broken part -- are never trusted. Positions come from box centres.
        up_un = world_up_from_boxes(np.concatenate([t.rotations for t in seg.tracks.values()]))

        tags: dict[str, dict[str, dict]] = {}
        for tid, tr in seg.tracks.items():
            idxs = list(range(0, len(tr), args.every))
            crops, meta = [], []
            for i in idxs:
                fidx = int(tr.frames[i])
                cam = seg.cameras.get(fidx)
                if cam is None:
                    continue
                fp = seg.frame_path(fidx)
                if not fp.is_file():
                    continue
                img = np.asarray(Image.open(fp).convert("RGB"))
                c = square_crop(img, list(tr.bbox_2d[i]), 0.15)
                if c is None:
                    continue
                crops.append(Image.fromarray(c).resize((224, 224), Image.BICUBIC))
                meta.append((i, fidx, cam))
            if not crops:
                continue

            X = embed(crops)
            with torch.no_grad():
                v = torch.nn.functional.normalize(
                    net(torch.tensor((X - mu) / sd, device=args.device)), dim=-1).cpu().numpy()
            angles = np.arctan2(v[:, 1], v[:, 0])

            seq = []
            for (i, fidx, cam), ang in zip(meta, angles):
                cam_c = -cam.extrinsic[:, :3].T @ cam.extrinsic[:, 3]
                up_w = sign_up_toward_camera(up_un, tr.centers[i], cam_c)
                up_cam = cam.rotate_world_to_cam(up_w)[0]
                p_cam = cam.world_to_cam(tr.centers[i])[0]        # POSITION only -- no rotation
                fl = visible_flank(float(ang), up_cam, p_cam)
                seq.append((fidx, fl))
                tags.setdefault(tid, {})[str(fidx)] = {
                    "flank": fl.side,
                    "confidence": round(fl.confidence, 3),
                    "heading_deg": round(float(np.degrees(ang)), 1),
                    "faces": {k: round(x, 3) for k, x in
                              face_visibility(float(ang), up_cam, p_cam).items()},
                }

            # --- the label-free validator ---
            sides = [f.side for _, f in seq]
            confs = np.array([f.confidence for _, f in seq])
            sw = [k for k in range(1, len(sides)) if sides[k] != sides[k - 1]]
            bad = [k for k in sw if confs[k] > EDGE_ON and confs[k - 1] > EDGE_ON]
            duty = sides.count("LEFT") / max(len(sides), 1)
            rows.append((f"{sd_path.parent.name}/{sd_path.name}", tid, len(sides),
                         len(sw), len(bad), duty, float(np.median(confs))))

        (args.out / f"{sd_path.parent.name}__{sd_path.name}__tags.json").write_text(
            json.dumps(tags, indent=1))
        print(f"[{si}/{len(segs)}] {sd_path.parent.name}/{sd_path.name}: "
              f"{len(tags)} tracks tagged", flush=True)

    if not rows:
        print("nothing tagged")
        return 1

    # --- report -------------------------------------------------------------------
    print("\n=== FLANK STABILITY (no labels needed) ===")
    print(f"{'segment/track':>42s} {'n':>4s} {'switches':>9s} {'BAD':>4s} {'%LEFT':>6s} {'medconf':>8s}")
    tot_sw = tot_bad = 0
    for seg, tid, n, sw, bad, duty, mc in rows:
        flag = "  <-- flickering at HIGH confidence" if bad else ""
        print(f"{seg+' t'+tid:>42s} {n:4d} {sw:9d} {bad:4d} {100*duty:5.0f}% {mc:8.2f}{flag}")
        tot_sw += sw
        tot_bad += bad

    n_tracks = len(rows)
    print(f"\n  tracks: {n_tracks}")
    print(f"  flank switches total          : {tot_sw}")
    print(f"  ...of which at HIGH confidence: {tot_bad}   <-- these are ERRORS "
          f"({100*tot_bad/max(tot_sw,1):.0f}% of switches)")
    print("\n  A switch at LOW confidence is CORRECT (the animal passed through head-on/tail-on).")
    print("  A switch at HIGH confidence is NONSENSE (it cannot flip sides while square-on).")
    print(f"  => the high-confidence switch rate is a label-free error estimate.")
    print(f"\n  tags written to {args.out}/  (the re-ID viewpoint payload)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
