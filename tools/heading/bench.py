"""Runtime study for the heading pipeline.  [CLUSTER, GPU -- run on an EXCLUSIVE GPU]

    python -m tools.heading.bench --crops data/heading/crops.npz --device cuda --dtype bf16 \
        --sample 300 --iters 100 --sweep --out data/heading/bench.json

Measures the two costs that matter, using CORRECT GPU timing (warmup, torch.cuda.synchronize(),
median + p10/p90 over many iters -- never a mean; GPU jitter is one-sided):

  * ONE-TIME template construction  -- per species, from the walking TRAINING crops. Amortised: it
    happens once and is then reused unchanged on every frame (see the 'once per species' story).
  * PER-FRAME inference, broken down by STAGE                -- so the paper can state that the
    method's marginal cost on top of a single frozen DINOv3 forward is ~1 ms of numpy.
  * OPTIONAL sweep (--sweep): forward-only throughput over resolution x dtype x batch, to back the
    '224 is 4x cheaper than 448' and batching claims.

Run it ALONE on the GPU: a CONTENDED gpu inflated earlier timings 3-4x, and that is not a property
of the method. The full path is timed -- JPEG decode, the numpy head, and the GPU->CPU transfer --
because a profile that omits a stage is a guess with numbers attached. (SAM mask GENERATION is an
upstream preprocessing cost of the detector/tracker stack, not of this method; we time only the
cost to LOAD a precomputed mask, and label it as such.)
"""
from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import numpy as np

from tools.heading.cropset import CropSet
from tools.heading.descriptors import DEFAULT_MODEL, Config, DenseExtractor, foreground
from tools.heading.split import video_split
from tools.heading.template import Accumulator, choose
from tools.heading.viewpoint import viewpoint_of


def _stats(ms: list[float]) -> dict:
    a = np.asarray(ms, dtype=np.float64)
    return {"median": float(np.median(a)), "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90)), "n": len(a)}


def _reps(fn, *, warmup: int, iters: int, sync) -> dict:
    """Time ONE callable over many reps (same input): for the forward and the numpy head."""
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()                                    # drain the async GPU queue before stopping
        out.append((time.perf_counter() - t0) * 1e3)
    return _stats(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="dump the measurements to JSON")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--facet", default="token", choices=["token", "key"])
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--fit", type=int, default=1500, help="template fit-set size (the one-time cost)")
    ap.add_argument("--sample", type=int, default=300, help="held-out crops used for the study")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--sweep", action="store_true", help="resolution x dtype x batch forward sweep")
    ap.add_argument("--allow-shared", action="store_true",
                    help="measure even though another process holds GPU memory. The result is NOT a "
                         "property of the method and must not go in a paper.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from PIL import Image

    # ---- REFUSE to time a contended GPU. -------------------------------------------------------
    # The docstring has always said "run it ALONE", but nothing enforced it, and a run that shared
    # the card with a 9.8 GB job produced 30.6 crops/s where the exclusive figure was ~357 -- a 12x
    # error that looks like a plausible measurement. Check BEFORE allocating anything.
    if args.device == "cuda" and torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        other_gb = (total_b - free_b) / 1e9
        if free_b < 0.90 * total_b:
            msg = (f"GPU IS NOT EXCLUSIVE: {other_gb:.1f} GB of {total_b/1e9:.1f} GB is already in "
                   f"use by another process. A contended GPU inflates these timings several-fold, "
                   f"and that is not a property of the method.")
            if not args.allow_shared:
                print(f"[FATAL] {msg}\n        Wait for the GPU (nvidia-smi), or pass "
                      f"--allow-shared to measure anyway -- but do NOT report the result.")
                return 2
            print(f"  WARNING: {msg}  (--allow-shared given; results are NOT reportable)\n")

    sync = torch.cuda.synchronize if (args.device == "cuda" and torch.cuda.is_available()) else None
    cfg = Config(args.layer, args.facet, args.size, args.bins)

    crops = CropSet(args.crops)
    d = crops.d
    te, _ = video_split(crops.species, crops.video, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    idx_fit = np.sort(rng.permutation(np.where(~te)[0])[: args.fit])
    idx_te = np.sort(rng.permutation(np.where(te)[0])[: args.sample])
    crops.prefetch(np.concatenate([idx_fit, idx_te]))

    rec: dict = {"config": f"L{args.layer}/{args.facet}/{args.size}", "dtype": args.dtype,
                 "torch": torch.__version__}
    if args.device == "cuda" and torch.cuda.is_available():
        rec["gpu"] = torch.cuda.get_device_name(0)
        rec["tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
    print(f"== hardware ==  {rec.get('gpu', args.device)}  |  torch {rec['torch']}  |  "
          f"dtype {args.dtype}  |  TF32 {rec.get('tf32', 'n/a')}  |  config {rec['config']}")
    print(f"   fit set {len(idx_fit)}  |  sample {len(idx_te)}  |  warmup {args.warmup}  iters {args.iters}\n")

    with DenseExtractor(args.model, args.device, args.dtype) as ex:
        # ---------- 1. ONE-TIME template construction (per species, amortised) ----------
        if sync:
            sync()
        t0 = time.perf_counter()
        acc = Accumulator(cfg)
        for b in range(0, len(idx_fit), 64):
            items = crops.batch(idx_fit[b: b + 64])
            for g, it in zip(ex.grid(np.stack([i.image for i in items]), cfg), items):
                acc.add(g, it)
        tmpl = acc.build()
        if sync:
            sync()
        build_s = time.perf_counter() - t0
        rec["template_build"] = {"total_s": build_s, "per_crop_ms": 1e3 * build_s / len(idx_fit),
                                 "n_fit": len(idx_fit), "n_species": len(tmpl.templates)}
        print(f"== ONE-TIME template construction ==")
        print(f"   {build_s:6.1f} s total  ({1e3*build_s/len(idx_fit):.1f} ms/crop, {len(idx_fit)} "
              f"crops, {len(tmpl.templates)} species) -- happens ONCE, then reused every frame\n")

        # ---------- 2. PER-FRAME inference, stage by stage (batch = 1 latency) ----------
        items = crops.batch(idx_te)
        imgs = np.stack([it.image for it in items])
        one = imgs[:1]

        # a representative grid + its foreground/scores, to time the numpy head in isolation
        g0 = ex.grid(one, cfg)[0]
        it0 = items[0]
        fg0 = foreground(g0, it0.instance)

        stages = {}
        # (a) JPEG decode -- one call per DISTINCT crop (size varies), median across crops
        dec = []
        jp = d["jpeg"]
        for i in idx_te:
            t0 = time.perf_counter()
            _ = np.asarray(Image.open(io.BytesIO(jp[i])).convert("RGB"))
            dec.append((time.perf_counter() - t0) * 1e3)
        stages["decode_jpeg"] = _stats(dec)
        # (b) SAM mask unpack (CACHED -- generation is upstream, not our cost)
        if crops.can_mask:
            mk = [int(i) for i in idx_te]
            stages["mask_unpack_cached"] = _reps(lambda: crops._mask(mk[0]),
                                                 warmup=args.warmup, iters=args.iters, sync=None)
        # (c) DINOv3 forward, batch = 1 (the dominant cost)
        stages["dinov3_forward_b1"] = _reps(lambda: ex.grid(one, cfg),
                                            warmup=args.warmup, iters=args.iters, sync=sync)
        # (d) foreground selection (numpy)
        stages["foreground"] = _reps(lambda: foreground(g0, it0.instance),
                                     warmup=args.warmup, iters=args.iters, sync=None)
        # (e) score the 4 candidates vs the template (numpy)
        stages["score_faces"] = _reps(
            lambda: tmpl.score_faces(g0, fg0, it0.face_uv, it0.face_ids, it0.species),
            warmup=args.warmup, iters=args.iters, sync=None)
        # (f) choose the sign + read out the viewpoint tag (trivial)
        s0 = tmpl.score_faces(g0, fg0, it0.face_uv, it0.face_ids, it0.species)
        def _choose_and_tag():
            slot, _m = choose(s0, it0.face_ids, axis=int(it0.geo_axis))
            viewpoint_of(float(d["face_alpha"][it0.i, max(slot, 0)]))
        stages["choose_and_tag"] = _reps(_choose_and_tag,
                                         warmup=args.warmup, iters=args.iters, sync=None)
        rec["per_frame_stages"] = stages

        head = sum(stages[k]["median"] for k in ("foreground", "score_faces", "choose_and_tag"))
        fwd = stages["dinov3_forward_b1"]["median"]
        total = fwd + head
        rec["per_frame_summary"] = {"forward_ms": fwd, "head_ms": head, "compute_total_ms": total,
                                    "fps_b1": 1e3 / total if total else None}

        print("== PER-FRAME inference (batch=1, ms) ==   median [p10-p90]")
        order = ["decode_jpeg", "mask_unpack_cached", "dinov3_forward_b1",
                 "foreground", "score_faces", "choose_and_tag"]
        for k in order:
            if k in stages:
                s = stages[k]
                print(f"   {k:<22s} {s['median']:8.2f}  [{s['p10']:6.2f}-{s['p90']:6.2f}]")
        print(f"   {'-'*22} {'-'*8}")
        print(f"   {'DINOv3 forward':<22s} {fwd:8.2f}")
        print(f"   {'method head (numpy)':<22s} {head:8.2f}   <- foreground+score+choose+tag")
        print(f"   {'COMPUTE TOTAL':<22s} {total:8.2f}   = {1e3/total:5.1f} fps (batch 1)\n")

        # ---------- 3. OPTIONAL sweep: forward throughput over size x dtype x batch ----------
        if args.sweep:
            print("== FORWARD SWEEP ==  ms/crop (median) and crops/s, forward only")
            print(f"   {'size':>4s} {'dtype':>5s} {'b=1':>8s} {'b=8':>8s} {'b=32':>8s} "
                  f"{'b=64':>8s}   {'best fps':>9s}")
            sweep = []
            for dt in ([args.dtype] if args.dtype == "fp32" else [args.dtype, "fp32"]):
                for sz in (224, 448):
                    with DenseExtractor(args.model, args.device, dt) as ex2:
                        c2 = Config(args.layer, args.facet, sz, args.bins)
                        percrop, best_fps = {}, 0.0
                        for bs in (1, 8, 32, 64):
                            batch = imgs[:bs] if len(imgs) >= bs else np.repeat(
                                imgs[:1], bs, axis=0)
                            # A large batch at 448 can exhaust a 16 GB card. Losing the whole run
                            # -- and the JSON, which is written at the very end -- to the LAST
                            # batch size is a poor trade for one missing cell.
                            try:
                                st = _reps(lambda b=batch: ex2.grid(b, c2),
                                           warmup=max(3, args.warmup // 2),
                                           iters=max(20, args.iters // 2), sync=sync)
                            except torch.OutOfMemoryError:
                                torch.cuda.empty_cache()
                                percrop[bs] = None
                                continue
                            pc = st["median"] / bs
                            percrop[bs] = pc
                            best_fps = max(best_fps, 1e3 / pc)
                        sweep.append({"size": sz, "dtype": dt, "ms_per_crop": percrop,
                                      "best_fps": best_fps})
                        print(f"   {sz:>4d} {dt:>5s} " +
                              " ".join(f"{percrop[b]:8.2f}" if percrop.get(b) else f"{'OOM':>8s}"
                                       for b in (1, 8, 32, 64)) +
                              f"   {best_fps:9.1f}")
            rec["forward_sweep"] = sweep

    if args.out:
        import json
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rec, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
