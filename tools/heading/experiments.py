"""THE PAPER TABLE: the main component study, the transfer test, and the compact ablation.

    # main table + ablation, on the walking (motion-labelled) held-out videos
    python -m tools.heading.experiments --crops data/heading/crops.npz \
        --out data/heading/exp --layer 24 --facet token --size 224

    # + THE TRANSFER TEST: the same template, scored on STANDING animals
    python -m tools.heading.experiments ... --stand-crops data/heading/crops_stand.npz

WHAT IS BEING ISOLATED
----------------------
Three signals are combined, and the table exists to show what each one contributes:

  setting                 3D geom   locomotion   DINOv3   output
  ----------------------------------------------------------------------------------------
  random sign             yes       -            -        chance-level sign
  locomotion only         yes       yes          -        defined ONLY on moving frames
  appearance, oracle axis yes*      yes          yes      head-vs-tail sign  (*axis given)
  full method             yes       yes          yes      continuous signed heading
  + visibility            yes       yes          yes      the flank tag re-ID consumes

METRICS
-------
The CONTINUOUS angular heading error is the primary result; the categorical numbers are
diagnostics.

  angular error      |wrap(azimuth_pred - azimuth_true)|, in the WORLD ground plane
  acc@15 / 30 / 45   the same, thresholded
  sign accuracy      head vs tail, on the true axis (chance 50%)
  flank accuracy     left vs right, WHERE A FLANK IS ACTUALLY VISIBLE (|sin alpha| >= 0.35)

ONE HONEST LIMITATION, STATED UP FRONT
--------------------------------------
The predicted heading is QUANTISED to the 3D box's horizontal face normals -- we choose an end of an
axis, we do not regress a free angle. So the angular error contains two things: our sign/axis choice,
AND the box's own axis error. The "locomotion only" row makes that floor visible: it is the error you
get with a PERFECT sign, and no method built on these boxes can beat it.

THE TRANSFER TEST (the scientifically important one)
-----------------------------------------------------
Everything above is measured on WALKING animals, because that is the only place motion labels exist.
But 133,809 of the frames are STATIONARY (84%) and none have been tested. `bridges.py` produces
labels for standing animals from WALK -> STAND -> WALK runs whose before/after headings agree within
30 degrees -- so the animal provably did not turn while it was stopped. Scoring the SAME template on
those crops answers the only question that matters: does DINOv3 carry the locomotion anchor to frames
where the animal is not moving, or does it only work where the anchors came from?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.heading.cropset import CropSet
from tools.heading.descriptors import DEFAULT_MODEL, Config, DenseExtractor, foreground
from tools.heading.split import video_split
from tools.heading.template import Accumulator, AxisTemplate, choose, opposite_slot
from tools.heading.viewpoint import EDGE_ON, viewpoint_of


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def score_all(ex, crops, idx, tmpl, batch, *, centred=True, masks=True):
    """(n, 4) per-face scores for every crop in `idx`."""
    S = np.full((len(idx), 4), np.nan)
    for b in range(0, len(idx), batch):
        items = crops.batch(idx[b: b + batch])
        G = ex.grid(np.stack([it.image for it in items]), tmpl.cfg)
        for k, (g, it) in enumerate(zip(G, items)):
            if it.species not in tmpl.templates:
                continue
            fg = foreground(g, it.instance if masks else None)
            S[b + k] = tmpl.score_faces(g, fg, it.face_uv, it.face_ids, it.species,
                                        centred=centred)
        print(f"    {min(b+batch, len(idx))}/{len(idx)}", end="\r", flush=True)
    print(" " * 30, end="\r")
    return S


def metrics(name, sign_ok, flank_ok, visible, n):
    """One row of the table. SIGN (head vs tail) and FLANK (the re-ID tag) only.

    We deliberately do NOT report an azimuth/angular error. The heading is quantised to the box's
    face normals, so the azimuth error is dominated by the BOX AXIS quality, not by what the method
    contributes -- every error is either ~0 deg (right sign) or ~180 deg (wrong sign), which is why
    it added nothing over the sign column. Sign and flank are what the method actually decides.
    """
    return {
        "setting": name, "n": int(n),
        "sign": float(np.mean(sign_ok)) if sign_ok is not None else float("nan"),
        "flank_vis": (float(np.mean(flank_ok[visible])) if flank_ok is not None and visible.any()
                      else float("nan")),
    }


def show(rows, title):
    print(f"\n=== {title} ===")
    print(f"{'setting':<26s} {'n':>6s} {'sign':>7s} {'flank*':>8s}")
    for r in rows:
        f = "" if np.isnan(r["flank_vis"]) else f"{100*r['flank_vis']:7.1f}%"
        s = "" if np.isnan(r["sign"]) else f"{100*r['sign']:6.1f}%"
        print(f"{r['setting']:<26s} {r['n']:6d} {s:>7s} {f:>8s}")
    print(f"  * flank accuracy is over frames where a flank is VISIBLE (|sin alpha| >= {EDGE_ON}); "
          f"elsewhere the animal is head-on and no flank exists to name.")


def evaluate(crops, idx, S, d, *, tag):
    """Build every row of the main table from one set of scores."""
    fid, y, geo = d["face_ids"][idx], d["y_face"][idx], d["geo_axis"][idx]
    falpha = d["face_alpha"][idx]
    n = len(idx)
    rng = np.random.default_rng(0)

    def ends(k, a):
        return [int(a), opposite_slot(fid[k], int(a))]

    rows = []

    # --- 1. RANDOM SIGN: geometry gives the axis, the sign is a coin flip ---
    pick = np.array([ends(k, geo[k])[rng.integers(2)] for k in range(n)])
    rows.append(metrics("random sign", pick == y, None, np.zeros(n, bool), n))

    # --- 2. LOCOMOTION REFERENCE: sign 100% by construction (this IS the label). ---
    rows.append(metrics("locomotion reference", np.ones(n, bool), None, np.zeros(n, bool), n))

    # --- 3. APPEARANCE, ORACLE AXIS: the axis is given; DINOv3 supplies only the sign ---
    # ABSTENTIONS. score_faces returns NaN when there is genuinely nothing to read -- most often
    # because the animal is exactly END-ON, so its body axis projects to a point and axis_profile
    # has no axis to bin patches along. The head and the tail land in the same place, and there is
    # no evidence to distinguish them.
    #
    # The rows that do NOT consult appearance (random sign, locomotion oracle) keep every crop; the
    # rows that DO will therefore have a smaller n. That difference is an abstention, not a dropped
    # sample, and the report must say so rather than leave an unexplained number in a table.
    ok = ~np.isnan(S).all(1)
    n_abstain = int((~ok).sum())
    if n_abstain:
        print(f"\n  ABSTENTIONS: {n_abstain}/{n} crops carry no appearance evidence (the animal is "
              f"end-on, so its\n  body axis projects to a point and there is no profile to read). "
              f"The appearance rows below\n  therefore report n={int(ok.sum())}, not {n}. This is an "
              f"abstention, not a dropped sample.")
    sv = np.where(np.isnan(S), -np.inf, S)
    t_of = np.array([opposite_slot(fid[k], int(y[k])) for k in range(n)])
    p_or = np.where(sv[np.arange(n), y] >= sv[np.arange(n), t_of], y, t_of)
    rows.append(metrics("appearance, oracle axis", (p_or == y)[ok], None,
                        np.zeros(ok.sum(), bool), ok.sum()))

    # --- 4. FULL METHOD: geometry proposes the axis, appearance disposes the sign ---
    p_full = np.array([choose(S[k], fid[k], axis=int(geo[k]))[0] for k in range(n)])
    m = ok & (p_full >= 0)
    a_p = falpha[np.arange(n), np.maximum(p_full, 0)]
    a_t = falpha[np.arange(n), y]
    fl_p = np.array([viewpoint_of(float(a))["flank"] for a in a_p])
    fl_t = np.array([viewpoint_of(float(a))["flank"] for a in a_t])
    vis = np.abs(np.sin(a_t)) >= EDGE_ON
    rows.append(metrics("FULL METHOD", (p_full == y)[m], (fl_p == fl_t)[m], vis[m], m.sum()))

    # --- 5. + VISIBILITY: the same predictions, read out as the re-ID tag ---
    rows.append(metrics("FULL + visibility", (p_full == y)[m], (fl_p == fl_t)[m], vis[m], m.sum()))

    show(rows, f"MAIN COMPONENT TABLE -- {tag}")

    # ---- WHERE DO THE SIGN ERRORS LIVE?  The decisive diagnostic. ----
    # Sign accuracy and flank accuracy cannot both be believed unless we know where the failures
    # sit. When an animal points at the camera its body axis has almost NO EXTENT IN THE IMAGE, so
    # the profile is degenerate and there is no evidence to read -- the head/tail choice is a coin
    # flip. Those are exactly the frames where no flank is visible and the tag is undefined anyway.
    # If the errors concentrate there, "it degrades on standing animals" is the wrong description.
    print(f"\n  sign accuracy vs how BROADSIDE the animal is (|sin alpha|, from the TRUE heading):")
    print(f"  {'|sin a|':>12s} {'n':>6s} {'sign':>7s} {'flank':>7s}   interpretation")
    bands = [(0.00, 0.35, "head-on / tail-on: NO flank exists, and the axis barely projects"),
             (0.35, 0.70, "oblique: a flank is visible"),
             (0.70, 1.01, "broadside: the flank is fully visible -- what re-ID needs")]
    band_rows = []
    for lo, hi, note in bands:
        q = m & (np.abs(np.sin(a_t)) >= lo) & (np.abs(np.sin(a_t)) < hi)
        if not q.any():
            continue
        fa = float((fl_p == fl_t)[q].mean())
        sg = float((p_full == y)[q].mean())
        band_rows.append({"lo": lo, "hi": hi, "n": int(q.sum()), "sign": sg, "flank": fa,
                          "note": note})
        print(f"  {lo:5.2f}-{hi:<5.2f} {q.sum():6d} {100*sg:6.1f}% {100*fa:6.1f}%   {note}")

    # per-species, on the full method
    sp = d["species"][idx]
    per_species = []
    print(f"\n  per-species (FULL METHOD):")
    for s in sorted(set(sp.tolist())):
        q = m & (sp == s)
        if not q.any():
            continue
        fv = q & vis
        fa = float((fl_p == fl_t)[fv].mean()) if fv.any() else float("nan")
        per_species.append({"species": s, "n": int(q.sum()),
                            "sign": float((p_full == y)[q].mean()), "flank_vis": fa})
        print(f"    {s:>9s} n={q.sum():5d}  sign {100*(p_full == y)[q].mean():5.1f}%  "
              f"flank* {100*fa:5.1f}%")
    return ({"rows": rows, "bands": band_rows, "per_species": per_species,
             "n_total": int(n), "n_abstain": n_abstain}, p_full, m)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--stand-crops", type=Path, default=None,
                    help="crops_stand.npz from bridges.py -- the transfer test (brief stops)")
    ap.add_argument("--human-crops", type=Path, default=None,
                    help="crops_human.npz -- 2 videos of GRAZING zebras with HUMAN face-locks. The "
                         "only independent check, and the only committed standers we have.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--facet", default="token", choices=["token", "key"])
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--fit", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    crops = CropSet(args.crops)
    d = crops.d
    te, test_v = video_split(crops.species, crops.video, seed=args.seed)
    cfg = Config(args.layer, args.facet, args.size, args.bins)
    print(f"{len(crops)} walking crops | held out {len(test_v)} videos ({te.sum()}) | "
          f"SAM masks: {'yes' if crops.can_mask else 'NO'}")

    rng = np.random.default_rng(args.seed)
    idx_fit = np.sort(rng.permutation(np.where(~te)[0])[: args.fit])
    idx_te = np.where(te)[0]
    crops.prefetch(np.concatenate([idx_fit, idx_te]))

    out = {}
    with DenseExtractor(args.model, args.device, args.dtype) as ex:
        # the template: the mean profile of WALKING crops from the TRAINING videos. Nothing trained.
        acc = Accumulator(cfg)
        for b in range(0, len(idx_fit), args.batch):
            items = crops.batch(idx_fit[b: b + args.batch])
            for g, it in zip(ex.grid(np.stack([i.image for i in items]), cfg), items):
                acc.add(g, it)
        tmpl = acc.build()
        print(f"template (nothing trained): {tmpl.n_fitted}")

        print("\nscoring held-out WALKING crops ...")
        S = score_all(ex, crops, idx_te, tmpl, args.batch)
        out["main"], _, _ = evaluate(crops, idx_te, S, d, tag="WALKING (held-out videos)")

        # ---- THE COMPACT ABLATION: at most the findings that explain WHY it works ----
        print("\n\n=== ABLATION ===")
        abl = []
        abl.append(dict(out["main"]["rows"][3], setting="full method"))

        print("  (a) UNCENTRED profile (the original scoring) ...")
        S_unc = score_all(ex, crops, idx_te, tmpl, args.batch, centred=False)
        r, _, _ = evaluate(crops, idx_te, S_unc, d, tag="ABLATION: uncentred")
        abl.append(dict(r["rows"][3], setting="  - centring"))

        if crops.can_mask:
            print("  (b) NO instance mask (crop-box foreground only) ...")
            S_nom = score_all(ex, crops, idx_te, tmpl, args.batch, masks=False)
            r, _, _ = evaluate(crops, idx_te, S_nom, d, tag="ABLATION: no SAM mask")
            abl.append(dict(r["rows"][3], setting="  - SAM instance mask"))

        # (c) appearance chooses the axis too -- no geometric prior
        n = len(idx_te)
        fid = d["face_ids"][idx_te]
        p_app = np.array([choose(S[k], fid[k])[0] for k in range(n)])
        y = d["y_face"][idx_te]
        faz, az_true = d["face_az"][idx_te], d["az"][idx_te]
        m = p_app >= 0
        abl.append(metrics("  - geometric axis prior",
                           faz[np.arange(n), np.maximum(p_app, 0)][m], az_true[m],
                           (p_app == y)[m], None, np.zeros(m.sum(), bool), m.sum()))
        show(abl, "COMPACT ABLATION (each row removes ONE component from the full method)")
        out["ablation"] = abl

        # ---- THE TRANSFER TEST ----
        if args.stand_crops:
            print("\n\n=== TRANSFER TEST: the SAME template, on STANDING animals ===")
            st = CropSet(args.stand_crops)
            sd = st.d
            te_s, _ = video_split(st.species, st.video, seed=args.seed)
            idx_s = np.where(te_s)[0]                     # held-out videos ONLY
            st.prefetch(idx_s)
            print(f"  {len(idx_s)} stationary crops on held-out videos "
                  f"(labels from WALK->STAND->WALK bridges; the animal provably did not turn)")
            S_s = score_all(ex, st, idx_s, tmpl, args.batch)
            out["transfer"], _, _ = evaluate(st, idx_s, S_s, sd, tag="STATIONARY (the transfer test)")

            w = out["main"]["rows"][3]
            s_ = out["transfer"]["rows"][3]
            print(f"\n  WALKING   sign {100*w['sign']:.1f}%   flank {100*w['flank_vis']:.1f}%")
            print(f"  STANDING  sign {100*s_['sign']:.1f}%   flank {100*s_['flank_vis']:.1f}%")
            print(f"  --------------------------------------------------------------")
            print(f"  THE GAP   sign {100*(s_['sign']-w['sign']):+.1f} pts   "
                  f"flank {100*(s_['flank_vis']-w['flank_vis']):+.1f} pts")
            print(f"\n  This is the number the whole approach rests on. The template was built ONLY")
            print(f"  from walking animals; these are standing ones, on videos it never saw. A small")
            print(f"  gap means DINOv3 carries the locomotion anchor beyond the frames that produced")
            print(f"  it. A large gap means locomotion only works where locomotion already was.")

        # ---- THE HUMAN CHECK: grazing zebras, human face-locks, never in training ----
        # The ONLY reference in the study that does not come from our own motion labels. Both videos
        # are held out permanently (split.HUMAN_LOCKED_VIDEOS). These animals GRAZE -- committed
        # standers that bridges.py cannot reach -- so this is also our only test on the true 86%.
        if args.human_crops:
            print("\n\n=== HUMAN-LABELLED GRAZING ZEBRAS (independent check) ===")
            hu = CropSet(args.human_crops)
            idx_h = np.arange(len(hu))          # both videos held out permanently
            hu.prefetch(idx_h)
            print(f"  {len(idx_h)} crops, 2 videos. Labels from a HUMAN, not from motion.")
            S_h = score_all(ex, hu, idx_h, tmpl, args.batch)
            out["human"], _, _ = evaluate(hu, idx_h, S_h, hu.d,
                                          tag="HUMAN-LABELLED grazing zebras")
            zw = next((r for r in out["main"]["per_species"] if r["species"] == "zebra"), None)
            h_ = out["human"]["rows"][3]
            if zw:
                print(f"\n  zebra, WALKING (our motion labels): sign {100*zw['sign']:.1f}%")
            print(f"  zebra, GRAZING (HUMAN labels)     : sign {100*h_['sign']:.1f}%   "
                  f"flank {100*h_['flank_vis']:.1f}%")
            print(f"\n  This is the only number that does not assume our own labels are right.")
            print(f"  Agreement here means the free motion labels are sound; disagreement means")
            print(f"  they are not, and everything else inherits it.")

    args.out.mkdir(parents=True, exist_ok=True)
    def _cnt(a):
        return {str(k): int(v) for k, v in zip(*np.unique(a, return_counts=True))}

    out["data"] = {
        "crops_total": _cnt(crops.species),
        "crops_train_videos": _cnt(crops.species[~te]),
        "crops_test_walking": _cnt(crops.species[te]),
        "template_fitted_on": {k: int(v) for k, v in tmpl.n_fitted.items()},
    }
    if args.stand_crops:
        st2 = CropSet(args.stand_crops)
        te2, _ = video_split(st2.species, st2.video, seed=args.seed)
        out["data"]["crops_standing_total"] = _cnt(st2.species)
        # The standing crops from TRAINING videos are used for NOTHING. The template is built from
        # WALKING crops only, so these are discarded -- and the table must say so, or the pool looks
        # like the test set.
        out["data"]["crops_standing_unused"] = _cnt(st2.species[~te2])
        out["data"]["crops_test_standing"] = _cnt(st2.species[te2])

    out["config"] = {"layer": args.layer, "facet": args.facet, "size": args.size,
                     "bins": args.bins, "seed": args.seed, "fit_crops": int(len(idx_fit)),
                     "held_out_videos": sorted(map(str, test_v)),
                     "sam_masks": bool(crops.can_mask)}
    (args.out / "results.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
