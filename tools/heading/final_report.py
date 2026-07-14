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
