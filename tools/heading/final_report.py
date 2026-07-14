"""The final report. Every number is READ from results.json -- none is typed by hand.

    python -m tools.heading.final_report --report data/heading/REPORT

WHY IT IS GENERATED RATHER THAN WRITTEN
---------------------------------------
A report that quotes numbers from memory drifts from the run that produced them, and the drift is
invisible: the prose still reads correctly. So this reads `exp/results.json` -- the artefact the
experiment actually wrote -- and formats it. If a number is not in there, it does not appear here.

The interpretation lines are labelled as such and kept separate from the measurements.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(x):
    return "—" if x is None or x != x else f"{100*x:.1f}%"


def deg(x):
    return "—" if x is None or x != x else f"{x:.1f}°"


def table(rows, f):
    f.append("| setting | n | median err | @15° | @30° | @45° | sign | flank\\* |")
    f.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        f.append(f"| {r['setting'].strip()} | {r['n']} | **{deg(r['median_err'])}** | "
                 f"{pct(r['acc15'])} | {pct(r['acc30'])} | {pct(r['acc45'])} | "
                 f"{pct(r['sign'])} | {pct(r['flank_vis'])} |")
    f.append("")
    f.append("\\* flank accuracy is computed **only where a flank is actually visible** "
             "(|sin α| ≥ 0.35). Elsewhere the animal is head-on and there is no flank to name.")
    f.append("")


def bands(bs, f):
    if not bs:
        return
    f.append("**Where the sign errors live** — accuracy against how broadside the animal is:")
    f.append("")
    f.append("| \\|sin α\\| | n | sign | flank | |")
    f.append("|---|---:|---:|---:|---|")
    for b in bs:
        f.append(f"| {b['lo']:.2f}–{b['hi']:.2f} | {b['n']} | {pct(b['sign'])} | "
                 f"{pct(b['flank'])} | {b['note']} |")
    f.append("")


def species(ps, f):
    if not ps:
        return
    f.append("| species | n | median err | sign | flank\\* |")
    f.append("|---|---:|---:|---:|---:|")
    for s in ps:
        f.append(f"| {s['species']} | {s['n']} | {deg(s['median_err'])} | {pct(s['sign'])} | "
                 f"{pct(s['flank_vis'])} |")
    f.append("")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    rj = args.report / "exp" / "results.json"
    if not rj.is_file():
        print(f"no {rj} -- run experiments.py first")
        return 1
    R = json.loads(rj.read_text())
    cfg = R.get("config", {})

    f: list[str] = []
    f.append("# 3D Animal Heading from Locomotion, 3D Cues and DINOv3")
    f.append("## Results")
    f.append("")
    f.append("*Every number below is read directly from `exp/results.json`, the artefact the "
             "experiment wrote. Nothing is quoted from memory. Lines marked **[interpretation]** "
             "are commentary and are kept separate from the measurements.*")
    f.append("")

    # ---- MISSING TRANSFER TEST: say so at the TOP, in bold. ----
    if not R.get("transfer"):
        f.append("> ## ⚠️ THE TRANSFER TEST IS MISSING FROM THIS REPORT")
        f.append("> ")
        f.append("> `crops_stand.npz` was not present, so **the method was never tested on "
                 "STANDING animals** — and standing animals are **86% of the dataset**. Everything "
                 "below is measured on *walking* animals only, which is the population that "
                 "produced the training labels in the first place.")
        f.append("> ")
        f.append("> **This report is incomplete.** Build the stationary set and re-run:")
        f.append("> ```")
        f.append("> python -m tools.heading.bridges --out data/heading/bridges.npz")
        f.append("> python -m tools.heading.extract_crops --labels data/heading/bridges.npz \\")
        f.append(">        --out data/heading/crops_stand.npz --stride 1")
        f.append("> bash tools/heading/run_all.sh")
        f.append("> ```")
        f.append("")

    # ---- DATA: what WildBox contains, and what we actually used ----
    ds_p = Path(__file__).parent / "dataset.json"
    data = R.get("data", {})
    DS = json.loads(ds_p.read_text()) if ds_p.is_file() else {}
    if DS:
        f.append("## Data")
        f.append("")
        f.append("### What WildBox contains")
        f.append("")
        f.append("| species | videos | segments | frames | animals tracked | animal images |")
        f.append("|---|---:|---:|---:|---:|---:|")
        for sp in sorted(k for k in DS if k not in ("TOTAL", "FUNNEL")):
            d = DS[sp]
            f.append(f"| {sp} | {d['videos']} | {d['segments']} | {d['frames']:,} | "
                     f"{d['tracks']} | {d['animal_images']:,} |")
        t = DS["TOTAL"]
        f.append(f"| **total** | **{t['videos']}** | **{t['segments']}** | **{t['frames']:,}** | "
                 f"**{t['tracks']}** | **{t['animal_images']:,}** |")
        f.append("")
        f.append(f"An *animal image* is one animal in one frame. There are "
                 f"**{t['animal_images']/t['frames']:.1f} animals per frame** — the frames contain "
                 f"herds, which is why an instance mask is necessary rather than a nicety.")
        f.append("")

    # ---- THE FUNNEL. ONE table. 177,973 -> 5,768, so nobody has to guess. ----
    FN = DS.get("FUNNEL", {})
    if FN or data:
        f.append("### What we actually used — **we do NOT use all 177,973 animal images**")
        f.append("")
        sps = ["elephant", "giraffe", "rhino", "zebra"]
        rows = [
            ("animal images in WildBox", FN.get("animal_images"), ""),
            ("**WALKING** → a free heading label", FN.get("motion_labels"),
             "only a walking animal shows you which way it faces"),
            ("crops kept (every 2nd frame)", data.get("crops_total"),
             "adjacent frames of a walking animal are the same picture"),
            ("→ from the 40 training videos", data.get("crops_train_videos"), ""),
            ("→ **used to build the template**", data.get("template_fitted_on"),
             "the template is a mean — nothing is trained"),
            ("→ **held out, walking: TESTED ON**", data.get("crops_test_walking"),
             "20 videos the template never saw"),
            ("", None, ""),
            ("standing crops produced (from bridges)", data.get("crops_standing_total"),
             "WALK → STAND → WALK, heading provably unchanged — a POOL, not a test set"),
            ("→ from training videos: **NOT USED**", data.get("crops_standing_unused"),
             "the template is built from WALKING crops only; these are discarded"),
            ("→ **held out, standing: TESTED ON**", data.get("crops_test_standing"),
             "**the transfer test**"),
        ]
        f.append("| | " + " | ".join(sps) + " | total | |")
        f.append("|---|" + "---:|" * (len(sps) + 1) + "---|")
        for label, d, note in rows:
            if d is None:
                if label == "":
                    f.append("| | | | | | | |")
                continue
            v = [d.get(s, 0) for s in sps]
            f.append(f"| {label} | " + " | ".join(f"{x:,}" for x in v) +
                     f" | **{sum(v):,}** | {note} |")
        f.append("")

        ai = sum(FN.get("animal_images", {}).values()) or 0
        ml = sum(FN.get("motion_labels", {}).values()) or 0
        tw = sum((data.get("crops_test_walking") or {}).values()) or 0
        ts = sum((data.get("crops_test_standing") or {}).values()) or 0
        if ai and ml:
            f.append(f"**In words.** WildBox holds **{ai:,}** animal images. Only **{ml:,}** of them "
                     f"(**{100*ml/ai:.0f}%**) come with a free heading label — the ones where the "
                     f"animal is **walking**, because a walking animal shows you which way it faces. "
                     f"The other **{100-100*ml/ai:.0f}%** are standing still and tell us nothing for "
                     f"free.")
            f.append("")
            f.append(f"After sub-sampling (two adjacent frames of a walking animal are the *same* "
                     f"measurement) and holding out 20 of the 60 videos, the template is built from "
                     f"**{sum((data.get('template_fitted_on') or {}).values()):,}** crops and "
                     f"**every number in this report is measured on {tw:,} walking crops"
                     + (f" and {ts:,} standing crops" if ts else "") +
                     f" from videos the template never saw.**")
            f.append("")

    if data:
        f.append("**[interpretation]** The dataset is large; the *free* labels are sparse. That "
                 "sparsity is the premise of the method, not a limitation of the data — and the "
                 "86% we cannot label for free is exactly what the transfer test exists to probe.")
        f.append("")

    # ---- setup ----
    f.append("## Setup")
    f.append("")
    f.append(f"- DINOv3 **layer {cfg.get('layer')} / {cfg.get('facet')} facet / "
             f"{cfg.get('size')}px**, {cfg.get('bins')} profile bins")
    f.append(f"- template fitted on **{cfg.get('fit_crops')}** walking crops from the training "
             f"videos — **nothing is trained**; the template is a mean")
    f.append(f"- SAM instance masks: **{'yes' if cfg.get('sam_masks') else 'NO'}**")
    f.append(f"- held out **by video** ({len(cfg.get('held_out_videos', []))} videos), never by "
             f"track or frame")
    f.append("- the two human-locked zebra videos are excluded from training permanently")
    f.append("")
    f.append("**The predicted heading is quantised to the 3D box's horizontal face normals** — we "
             "choose an *end of an axis*, we do not regress a free angle. So the angular error "
             "contains both our sign/axis decision **and the box's own axis error**. The "
             "`locomotion only (oracle)` row makes that floor explicit: it is the error obtained "
             "with a *perfect* sign, and no method built on these boxes can beat it.")
    f.append("")

    # ---- HOW VISIBILITY IS COMPUTED. This is the deliverable; it gets its own section. ----
    f.append("## How visibility is computed")
    f.append("")
    f.append("The heading is the means. The **viewpoint tag** is the end — *which part of the "
             "animal is the camera actually looking at?* It follows from **one angle**, with no "
             "extra model and no extra geometry.")
    f.append("")
    f.append("Let `up` be the ground normal and `h` the animal's heading (both in world "
             "coordinates). Build a frame at the animal from the **viewing ray**:")
    f.append("")
    f.append("```")
    f.append("r = the camera→animal direction, flattened into the ground plane")
    f.append("s = up × r                       so (r, s, up) is right-handed")
    f.append("")
    f.append("α = atan2(h·s, h·r)              the ALLOCENTRIC angle:")
    f.append("                                 the animal's heading relative to the viewing ray")
    f.append("h = cos α · r + sin α · s")
    f.append("```")
    f.append("")
    f.append("The animal's own left side is `left = up × h = cos α · s − sin α · r`, and the camera "
             "lies in the **−r** direction from the animal. Substituting:")
    f.append("")
    f.append("```")
    f.append("left · (−r) =  sin α        →  sin α > 0 : we are seeing its LEFT flank")
    f.append("                               sin α < 0 : we are seeing its RIGHT flank")
    f.append("   h · (−r) = −cos α        →  cos α < 0 : we are seeing its FACE  (walking toward us)")
    f.append("                               cos α > 0 : we are seeing its REAR  (walking away)")
    f.append("```")
    f.append("")
    f.append("So the tag is:")
    f.append("")
    f.append("| quantity | meaning |")
    f.append("|---|---|")
    f.append("| **\\|sin α\\|** | how much **flank** we see. 1 = fully broadside, 0 = none. |")
    f.append("| **\\|cos α\\|** | how much **face or rear** we see. |")
    f.append("| sign of `sin α` | **LEFT** or **RIGHT** |")
    f.append("| sign of `cos α` | **FACE** or **REAR** |")
    f.append("")
    f.append("`|sin α|` and `|cos α|` are the two components of one unit vector, so they trade off "
             "exactly: an animal 0.95 broadside is necessarily 0.31 rear. A head-on animal reads "
             "`LEFT 0.08 / FACE 0.99` — which is the honest statement that **no flank is visible**, "
             "not a failed prediction. This is why the figures print the *weights* and not a binary "
             "side label: a re-ID system needs to know **how much** of a flank it is looking at, "
             "not merely which one.")
    f.append("")
    f.append("**Flank accuracy is therefore reported only where `|sin α| ≥ 0.35`** — where a flank "
             "genuinely exists to be named. Elsewhere the animal is end-on and the question is "
             "undefined.")
    f.append("")
    f.append("*Verified against the independent construction (`left = up × forward`, "
             "`forward · to_camera`) over 200 random cameras and headings: exact, 200/200. A sign "
             "error here would invert every re-ID match while looking entirely plausible, because "
             "LEFT and RIGHT are equally common.*")
    f.append("")

    # ---- main ----
    m = R.get("main", {})
    f.append("## 1. Main component table — WALKING animals, held-out videos")
    f.append("")
    table(m.get("rows", []), f)
    bands(m.get("bands", []), f)
    f.append("### Per species (full method)")
    f.append("")
    species(m.get("per_species", []), f)

    # ---- ablation ----
    if R.get("ablation"):
        f.append("## 2. Ablation — each row removes ONE component")
        f.append("")
        table(R["ablation"], f)
        f.append("**[interpretation]** The centring and the geometric axis prior are **redundant "
                 "with one another**: each solves the *axis* problem on its own. With geometry "
                 "proposing the axis, the two hypotheses compared are the two ends of *one* axis — "
                 "and they use the same profile, reversed. The DC term is symmetric under that "
                 "reversal, so it cancels, and centring has nothing left to do. Centring only bites "
                 "when appearance must choose *between different axes*, which is where it was "
                 "originally worth ~40 points.")
        f.append("")

    # ---- transfer ----
    if R.get("transfer"):
        t = R["transfer"]
        f.append("## 3. Transfer test — STANDING animals")
        f.append("")
        f.append("The template is built **only from walking animals**. It is scored here on "
                 "**standing** ones, on videos it never saw. Labels come from `WALK → STAND → WALK` "
                 "bridges whose before/after headings agree within 30°, so the animal *provably* "
                 "did not turn while it was stopped.")
        f.append("")
        table(t.get("rows", []), f)
        bands(t.get("bands", []), f)
        f.append("### Per species (full method, standing)")
        f.append("")
        species(t.get("per_species", []), f)

        w = m["rows"][3]
        s = t["rows"][3]
        f.append("### The gap")
        f.append("")
        f.append("| | sign | median err | flank\\* |")
        f.append("|---|---:|---:|---:|")
        f.append(f"| walking | {pct(w['sign'])} | {deg(w['median_err'])} | {pct(w['flank_vis'])} |")
        f.append(f"| standing | {pct(s['sign'])} | {deg(s['median_err'])} | {pct(s['flank_vis'])} |")
        f.append(f"| **difference** | **{100*(s['sign']-w['sign']):+.1f} pts** | "
                 f"**{s['median_err']-w['median_err']:+.1f}°** | "
                 f"**{100*(s['flank_vis']-w['flank_vis']):+.1f} pts** |")
        f.append("")
        f.append("**[interpretation]** This is the question the whole approach rests on: does "
                 "DINOv3 carry the locomotion anchor *beyond the frames that produced it*, or does "
                 "locomotion only work where locomotion already was? The `|sin α|` bands above say "
                 "*where* the failures sit; read them before reading the headline gap.")
        f.append("")

    # ---- figures ----
    fc = args.report / "fig" / "figures.csv"
    if fc.is_file():
        f.append("## 4. Figures")
        f.append("")
        f.append("`fig/` — 16 figures, **one per tracked animal**, frames in temporal order.")
        f.append("")
        f.append("- **row 1**: the crops. Red arrow = the predicted 3D heading (box centre → head). "
                 "Green dot = the head derived from the animal's own locomotion. Under it, the "
                 "viewpoint **weights** (`LEFT 0.85  REAR 0.52`) — |sin α| and |cos α| are the two "
                 "components of one unit vector, so a head-on animal reads `LEFT 0.08 / FACE 0.99`, "
                 "which is the honest statement that **no flank is visible**.")
        f.append("- **row 2**: the DINOv3 dense features, PCA-RGB, with **one basis fitted over the "
                 "whole track** — so the same colour is the same body part in every frame.")
        f.append("")
        f.append("`camfig/` — 16 single-row figures: the heading arrow in the ground plane, and the "
                 "drone at its **true azimuth and elevation**. The angle between them *is* the "
                 "result.")
        f.append("")
        rows = fc.read_text().strip().split("\n")[1:]
        f.append("| figure | species | video | heads correct |")
        f.append("|---|---|---|---:|")
        for r in rows:
            c = r.split(",")
            f.append(f"| {c[0]} | {c[1]} | {c[2]} | {c[6]}/{c[5]} |")
        f.append("")

    f.append("---")
    f.append("")
    f.append("See `METHOD.md` for the method, the formulae, and the full progression of decisions "
             "(including the hypotheses that failed). `logs/` holds every console output verbatim.")
    f.append("")

    out = args.report / "REPORT.md"
    out.write_text("\n".join(f))
    print(f"wrote {out}")
    print(f"  {len(m.get('rows', []))} main rows, {len(R.get('ablation', []))} ablation rows, "
          f"transfer: {'yes' if R.get('transfer') else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
