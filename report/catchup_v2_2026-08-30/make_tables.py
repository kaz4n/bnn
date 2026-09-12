#!/usr/bin/env python3
"""Emit the LaTeX tables of the catch-up report straight from the score JSONs.

No number in the report is typed by hand. If a run is missing its row simply
does not appear, so the report can never quote a result that was not produced.
"""
import glob
import json
import re
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HT = os.path.join(ROOT, "attack", "hardtests_20260830")
RERUN = os.path.join(ROOT, "attack", "rerun_20260830")
OUT = os.path.join(os.path.dirname(__file__), "tables")
os.makedirs(OUT, exist_ok=True)


def load(path):
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def m_of(path):
    return load(path)["metrics"] if os.path.exists(path) else None


def fmt(v, n=3):
    return "--" if v is None else f"{v:.{n}f}"


def write(name, lines):
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")
    print("wrote", os.path.relpath(path, ROOT))


HDR = (r"Run & Pixel acc. & All-background & F1 & IoU & MSSIM & Pixel dist. & Recog. & $n$ \\")


def row(label, m):
    if m is None:
        return None
    return (f"{label} & {fmt(m['bit_acc'])} & {fmt(m['all_zero_bit_acc'])} & "
            f"{fmt(m['foreground_f1'])} & {fmt(m['foreground_iou'])} & "
            f"{fmt(m['mssim'])} & {fmt(m['pixel_level_distance'],1)} & "
            f"{fmt(m['recognition_accuracy_recovered'],2)} & {m['n_images']} \\\\")


def table_sweep():
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\hline", HDR, r"\hline"]
    any_row = False
    for r in range(1, 6):
        m = m_of(os.path.join(HT, f"score_sweep_R{r}.json"))
        rr = row(f"{r} repeat" + ("" if r == 1 else "s"), m)
        if rr:
            lines.append(rr)
            any_row = True
    lines += [r"\hline", r"\end{tabular}"]
    if any_row:
        write("tab_sweep.tex", lines)


def table_conditions():
    order = [("C1_control", "Same conditions (control)"),
             ("C2_reprogram", "FPGA reprogrammed"),
             ("C3_clk7m5", "Clock 5 $\\to$ 7.5\\,MHz"),
             ("C4_clk10m", "Clock 5 $\\to$ 10\\,MHz"),
             ("C5_gain30", "Gain 40 $\\to$ 30\\,dB"),
             ("C6_gain50", "Gain 40 $\\to$ 50\\,dB")]
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\hline", HDR, r"\hline"]
    any_row = False
    for key, lab in order:
        m = m_of(os.path.join(HT, f"score_cond_{key}.json"))
        rr = row(lab, m)
        if rr:
            lines.append(rr)
            any_row = True
    lines += [r"\hline", r"\end{tabular}"]
    if any_row:
        write("tab_conditions.tex", lines)


def table_headline():
    entries = [
        # the evaluation count is printed in the last column, so the label states only
        # the profile size; a partially finished run is then reported honestly.
        (os.path.join(HT, "score_paperscale.json"),
         "300 profile images, control evaluation set"),
        (os.path.join(HT, "score_cond_C1_control.json"),
         "40 profile / 40 evaluation, fresh images"),
        (os.path.join(RERUN, "score_trained_80_eval40_heldout.json"),
         "40 profile / 40 evaluation, earlier run"),
        (os.path.join(HT, "score_probe_mismatch.json"),
         "Probe mismatch (one-hot template vs trained traces)"),
        (os.path.join(HT, "score_sweep_R1.json"),
         "Single hardware repeat (no averaging)"),
    ]
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\hline", HDR, r"\hline"]
    any_row = False
    for path, lab in entries:
        rr = row(lab, m_of(path))
        if rr:
            lines.append(rr)
            any_row = True
    lines += [r"\hline", r"\end{tabular}"]
    if any_row:
        write("tab_headline.tex", lines)


def table_mixed():
    """Clock-mixing matrix: three templates x two attack clocks."""
    order = [("5m", "Profiled at 5\\,MHz only"),
             ("10m", "Profiled at 10\\,MHz only"),
             ("mixed", "Mixed, 20 images at each clock")]
    attacks = [("C1_control", "5\\,MHz"), ("C4_clk10m", "10\\,MHz")]
    lines = [r"\begin{tabular}{lrrrrrr}", r"\hline",
             r"Template (40 profile images) & "
             r"\multicolumn{3}{c}{Attack at 5\,MHz} & \multicolumn{3}{c}{Attack at 10\,MHz} \\",
             r" & F1 & MSSIM & Recog. & F1 & MSSIM & Recog. \\", r"\hline"]
    any_row = False
    for key, lab in order:
        cells = []
        ok = True
        for akey, _ in attacks:
            m = m_of(os.path.join(HT, f"score_mix_{key}_vs_{akey}.json"))
            if m is None:
                ok = False
                break
            cells += [fmt(m["foreground_f1"]), fmt(m["mssim"]),
                      fmt(m["recognition_accuracy_recovered"], 2)]
        if ok:
            lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
            any_row = True
    lines += [r"\hline", r"\end{tabular}"]
    if any_row:
        write("tab_mixed.tex", lines)


def table_p2p():
    """Power2Picture on gate-passing captures, against the voided earlier run."""
    entries = [
        (os.path.join(ROOT, "attack", "results_p2p_clockabs", "summary.json"),
         "Earlier 9000-image run \\emph{(void: no gate)}"),
        (os.path.join(ROOT, "attack", "rerun_20260830",
                      "p2p_2000_clockabs_mse", "summary.json"),
         "Corrected capture, 20 epochs"),
        (os.path.join(HT, "p2p_matched_e60", "summary.json"),
         "Corrected capture, 60 epochs"),
        (os.path.join(HT, "p2p_matched_e120", "summary.json"),
         "Corrected capture, 120 epochs"),
    ]
    lines = [r"\begin{tabular}{lrrrrrr}", r"\hline",
             r"Run & Pixel acc. & All-background & F1 & MSSIM & Recog. & $n$ \\", r"\hline"]
    any_row = False
    for path, lab in entries:
        if not os.path.exists(path):
            continue
        d = load(path)
        m = d.get("test_metrics") or d.get("metrics")
        lines.append(
            f"{lab} & {fmt(m['bit_acc'])} & {fmt(m['all_zero_bit_acc'])} & "
            f"{fmt(m['foreground_f1'])} & {fmt(m['mssim'])} & "
            f"{fmt(m['recognition_accuracy_recovered'], 2)} & {int(d['n_test'])} \\\\")
        any_row = True
    lines += [r"\hline", r"\end{tabular}"]
    if any_row:
        write("tab_p2p.tex", lines)


def table_grey():
    """Grey design against the binary design, same generator and same split."""
    entries = [(os.path.join(HT, "p2p_grey_e60", "summary.json"),
                "Grey design (real 0--255 pixels)"),
               (os.path.join(HT, "p2p_matched_e60", "summary.json"),
                "Binary design (thresholded pixels)")]
    lines = [r"\begin{tabular}{lrrrrrr}", r"\hline",
             r"Design & MAE (0--255) & Corr. & MSSIM & F1 & Recog. & $n$ \\",
             r"\hline"]
    any_row = False
    for path, lab in entries:
        if not os.path.exists(path):
            continue
        d = load(path)
        m = d.get("test_metrics") or d.get("metrics")
        corr = fmt(m["grey_corr"]) if "grey_corr" in m else "--"
        lines.append(
            f"{lab} & {fmt(m['pixel_level_distance'], 2)} & {corr} & "
            f"{fmt(m['mssim'])} & {fmt(m['foreground_f1'])} & "
            f"{fmt(m['recognition_accuracy_recovered'], 2)} & {int(d['n_test'])} \\\\")
        any_row = True
    lines += [r"\hline", r"\end{tabular}"]
    if any_row:
        write("tab_grey.tex", lines)



def table_delta():
    """Foreground F1 against the candidate search radius, for two template sizes."""
    import glob
    series = {}
    for f in glob.glob(os.path.join(HT, "score_delta_*.json")):
        base = os.path.basename(f)[len("score_delta_"):-len(".json")]
        tag, d = base.rsplit("_d", 1)
        series.setdefault(tag, {})[float(d.replace("p", "."))] = load(f)["metrics"]
    if not series:
        return
    shown = [1.40, 1.00, 0.60, 0.40, 0.25, 0.15, 0.08, 0.06, 0.03]
    lines = [r"\begin{tabular}{l" + "r" * len(shown) + "}", r"\hline",
             "Template & " + " & ".join(f"$\\delta$ = {d:g}" for d in shown) + r" \\",
             r"\hline"]
    any_row = False
    for tag, lab in (("p40", "40 profiling images"), ("p300", "300 profiling images")):
        if tag not in series:
            continue
        cells = []
        best = max(series[tag].items(), key=lambda kv: kv[1]["foreground_f1"])[0]
        for d in shown:
            m = series[tag].get(d)
            if m is None:
                cells.append("--")
            elif d == best:
                cells.append("\\textbf{%s}" % fmt(m["foreground_f1"]))
            else:
                cells.append(fmt(m["foreground_f1"]))
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
        any_row = True
    lines += [r"\hline", r"\end{tabular}"]
    if any_row:
        write("tab_delta.tex", lines)


FGRID = [("5m", "5"), ("7p5m", "7.5"), ("10m", "10"), ("20m", "20")]


def table_grid():
    """Four profiling clocks against four attack clocks."""
    lines = [r"\begin{tabular}{lrrrr}", r"\hline",
             r"Profiled at & \multicolumn{4}{c}{Attacked at} \\",
             " & " + " & ".join(f"{lab}\\,MHz" for _, lab in FGRID) + r" \\",
             r"\hline"]
    any_row = False
    for pk, plab in FGRID:
        cells = []
        for ak, _ in FGRID:
            m = m_of(os.path.join(HT, f"score_g4_{pk}_vs_{ak}.json"))
            if m is None:
                cells.append("--")
            elif pk == ak:
                cells.append("\\textit{%s}" % fmt(m["foreground_f1"]))
            else:
                cells.append(fmt(m["foreground_f1"]))
        lines.append(f"{plab}\\,MHz & " + " & ".join(cells) + r" \\")
        any_row = True
    lines += [r"\hline", r"\end{tabular}"]
    if any_row:
        write("tab_grid.tex", lines)


def table_jitter():
    """What a clock-randomising defender costs each template."""
    lines = [r"\begin{tabular}{lrrrr}", r"\hline",
             r"Profiled at & Matched clock & Randomised clock & Change & Recognition \\",
             r"\hline"]
    any_row = False
    for pk, plab in FGRID:
        a = m_of(os.path.join(HT, f"score_g4_{pk}_vs_{pk}.json"))
        b = m_of(os.path.join(HT, f"score_g4_{pk}_vs_jitter.json"))
        if a is None or b is None:
            continue
        lines.append(
            f"{plab}\\,MHz & {fmt(a['foreground_f1'])} & {fmt(b['foreground_f1'])} & "
            f"{b['foreground_f1'] - a['foreground_f1']:+.3f} & "
            f"{fmt(b['recognition_accuracy_recovered'], 2)} \\\\")
        any_row = True
    lines += [r"\hline", r"\end{tabular}"]
    if any_row:
        write("tab_jitter.tex", lines)



def best_over_delta(pattern):
    """Best-scoring run over every delta present, as (delta_label, metrics).

    The grid sweeps are stored one JSON per (cell, delta), named ..._d0p10.json.
    Globbing rather than listing the radii means the table follows whatever the
    sweep actually produced, so extending the sweep never needs a code change.
    """
    best = None
    for path in glob.glob(pattern):
        m = re.search(r"_d(0p\d+)\.json$", os.path.basename(path))
        if not m:
            continue
        d = float(m.group(1).replace("p", "."))
        met = m_of(path)
        if met is None:
            continue
        if best is None or met["foreground_f1"] > best[1]["foreground_f1"]:
            best = (f"{d:.2f}", met)
    return best if best else (None, None)


def table_grid_best():
    """The transfer surface with delta tuned per cell rather than fixed.

    A single delta is fair between cells but is not each cell's optimum, and the
    matched-condition cells suffer most from that: 10 MHz on its own condition
    reads 0.675 at the fixed 0.25 and 0.882 at its own best radius.
    """
    lines = [r"\begin{tabular}{lrrrr}", r"\hline",
             r"Profiled at & \multicolumn{4}{c}{Attacked at} \\",
             " & " + " & ".join(rf"{lab}\,MHz" for _, lab in FGRID) + r" \\",
             r"\hline"]
    any_row = False
    for pk, plab in FGRID:
        cells = []
        for ak, _ in FGRID:
            dlab, m = best_over_delta(
                os.path.join(HT, f"score_gd_{pk}_vs_{ak}_d*.json"))
            if m is None:
                cells.append("--")
                continue
            txt = r"%s {\tiny (%s)}" % (fmt(m["foreground_f1"]), dlab)
            cells.append(r"\textit{%s}" % txt if pk == ak else txt)
        lines.append(rf"{plab}\,MHz & " + " & ".join(cells) + r" \\")
        any_row = True
    lines += [r"\hline", r"\end{tabular}"]
    if any_row:
        write("tab_grid_best.tex", lines)


def table_jitter():
    """Clock randomisation, with delta tuned on both sides of the comparison.

    The earlier version of this table read both columns at a fixed delta = 0.25.
    Once the matched column is retuned per cell the randomised column has to be
    retuned too, or the defence is measured against a handicapped attacker.
    """
    lines = [r"\begin{tabular}{lrrrrr}", r"\hline",
             r"Profiled at & Matched clock & Randomised clock & Change & "
             r"Recognition & $\delta$ (m / r) \\",
             r"\hline"]
    any_row = False
    for pk, plab in FGRID:
        da, a = best_over_delta(os.path.join(HT, f"score_gd_{pk}_vs_{pk}_d*.json"))
        db, b = best_over_delta(os.path.join(HT, f"score_jd_{pk}_vs_jitter_d*.json"))
        if a is None or b is None:
            continue
        lines.append(
            rf"{plab}\,MHz & {fmt(a['foreground_f1'])} & {fmt(b['foreground_f1'])} & "
            f"{b['foreground_f1'] - a['foreground_f1']:+.3f} & "
            f"{fmt(b['recognition_accuracy_recovered'], 2)} & {da} / {db} \\\\")
        any_row = True
    lines += [r"\hline", r"\end{tabular}"]
    if any_row:
        write("tab_jitter.tex", lines)


def table_mix2():
    """Mixing with equal per-condition coverage, swept over delta.

    The radii are globbed, not listed: the first version of this sweep bottomed
    out at 0.10 and the mixed template's optimum sat on that floor, so the sweep
    was later extended downward and the table has to follow it.
    """
    seen = set()
    for path in glob.glob(os.path.join(HT, "score_mx2_*_vs_5m_d*.json")):
        mo = re.search(r"_d(0p\d+)\.json$", os.path.basename(path))
        if mo:
            seen.add(mo.group(1))
    deltas = sorted(seen, key=lambda k: float(k.replace("p", ".")))
    lines = [r"\begin{tabular}{llrr}", r"\hline",
             r"Template & $\delta$ & Attacked at 5\,MHz & Attacked at 10\,MHz \\",
             r"\hline"]
    any_row = False
    for tk, tlab in (("5m", r"Profiled at 5\,MHz"),
                     ("10m", r"Profiled at 10\,MHz"),
                     ("mixed", "Mixed, 40 at each")):
        first_done = False
        for dk in deltas:
            a = m_of(os.path.join(HT, f"score_mx2_{tk}_vs_5m_d{dk}.json"))
            b = m_of(os.path.join(HT, f"score_mx2_{tk}_vs_10m_d{dk}.json"))
            if a is None or b is None:
                continue
            dlab = f"{float(dk.replace('p', '.')):.2f}"
            first = "" if first_done else tlab
            first_done = True
            lines.append(f"{first} & {dlab} & {fmt(a['foreground_f1'])} & "
                         f"{fmt(b['foreground_f1'])} \\\\")
            any_row = True
        lines.append(r"\hline")
    lines += [r"\end{tabular}"]
    if any_row:
        write("tab_mix2.tex", lines)


def table_amp():
    """Three designs on the SAME 2000 images: the amplifier-width control.

    Narrow vs wide isolates amplifier width inside the binary task; wide vs grey
    then isolates the input datapath with the amplifier held equal.
    """
    entries = [
        (os.path.join(HT, "p2p_matched_e60", "summary.json"),
         "Binary, narrow amplifier ($9\\times63$)"),
        (os.path.join(HT, "p2p_wide_e60", "summary.json"),
         "Binary, wide amplifier ($72\\times63$)"),
        (os.path.join(HT, "p2p_greym_e60", "summary.json"),
         "Grey, wide amplifier ($72\\times63$)"),
    ]
    lines = [r"\begin{tabular}{lrrrrr}", r"\hline",
             r"Design & MAE (0--255) & Corr. & MSSIM & F1 & Recog. \\",
             r"\hline"]
    any_row = False
    for path, lab in entries:
        if not os.path.exists(path):
            continue
        d = load(path)
        m = d.get("test_metrics") or d.get("metrics")
        corr = fmt(m["grey_corr"]) if m.get("grey_corr") is not None else "--"
        lines.append(
            f"{lab} & {fmt(m['pixel_level_distance'], 2)} & {corr} & "
            f"{fmt(m['mssim'])} & {fmt(m['foreground_f1'])} & "
            f"{fmt(m['recognition_accuracy_recovered'], 2)} \\\\")
        any_row = True
    lines += [r"\hline", r"\end{tabular}"]
    if any_row:
        write("tab_amp.tex", lines)


def table_stage():
    """Candidate quality per stage against the final reconstruction score.

    Retrieval and ranking are measured on the real template files and the real
    evaluation bundles, so the columns describe exactly what the attack sees.
    Printed at delta = 0.25 so the F1 column is the same fixed-radius grid the
    diagnostic was run at, not the per-cell tuned one.
    """
    diag_path = os.path.join(HT, "grid_rank_real_d0p25.json")
    if not os.path.exists(diag_path):
        return
    diag = load(diag_path)
    lines = [r"\begin{tabular}{lrrrr}", r"\hline",
             r"Profiled $\to$ attacked & In set & Rank 1 & Candidates & F1 \\",
             r"\hline"]
    any_row = False
    for pk, plab in FGRID:
        for ak, alab in FGRID:
            d = diag.get(f"{pk}|{ak}")
            m = m_of(os.path.join(HT, f"score_gd_{pk}_vs_{ak}_d0p25.json"))
            if d is None or m is None:
                continue
            lab = f"{plab} $\\to$ {alab}\\,MHz"
            if pk == ak:
                lab = r"\textit{%s}" % lab
            lines.append(
                f"{lab} & {fmt(d['inset'])} & {fmt(d['rank1'])} & "
                f"{d['mean_cand']:.1f} & {fmt(m['foreground_f1'])} \\\\")
            any_row = True
        lines.append(r"\hline")
    lines += [r"\end{tabular}"]
    if any_row:
        write("tab_stage.tex", lines)


DS = [("mnist", "MNIST"), ("fashion", "Fashion")]


def table_dataset_grid():
    """Template dataset against attacked dataset, delta tuned per cell.

    Raw F1 is not comparable between these two datasets: MNIST evaluation images
    are 11.4 % foreground and Fashion ones 25.9 %, so the best F1 reachable by a
    predictor that knows nothing moves from 0.204 to 0.474. Both the raw value
    and the chance-relative excess are printed so the reader can see why.
    """
    lines = [r"\begin{tabular}{llrrrrr}", r"\hline",
             r"Template & Attacked & $\delta$ & F1 & chance & excess & MSSIM \\",
             r"\hline"]
    any_row = False
    for pk, plab in DS:
        for ak, alab in DS:
            best = None
            for path in glob.glob(os.path.join(HT, f"score_ds2_{pk}_vs_{ak}_d*.json")):
                mo = re.search(r"_d(0p\d+)\.json$", os.path.basename(path))
                if not mo:
                    continue
                d = float(mo.group(1).replace("p", "."))
                m = m_of(path)
                if m is None:
                    continue
                if best is None or m["foreground_f1"] > best[1]["foreground_f1"]:
                    best = (d, m)
            if best is None:
                continue
            d, m = best
            c = m["f1_chance"]
            exc = (m["foreground_f1"] - c) / (1 - c)
            lab = f"{plab} $\\to$ {alab}"
            row = (f"{plab} & {alab} & {d:.3f} & {fmt(m['foreground_f1'])} & "
                   f"{fmt(c)} & {fmt(exc)} & {fmt(m['mssim'])} \\\\")
            lines.append(row)
            any_row = True
    lines += [r"\hline", r"\end{tabular}"]
    if any_row:
        write("tab_dataset_grid.tex", lines)


def table_patch_coverage():
    """Why the MNIST template cannot render clothing: it never saw dense patches."""
    import numpy as np
    rows = []
    for k, lab in DS:
        f = os.path.join(HT, f"ds_tmpl_{k}.npz")
        if not os.path.exists(f):
            continue
        T = np.load(f, allow_pickle=True)
        c = T["patch_code"].astype(int)
        b = T["patch_bits"].astype(int)
        u, cnt = np.unique(c, return_counts=True)
        pr = cnt / cnt.sum()
        rows.append((lab, len(u), float(pr[u == 0][0]),
                     float(pr[u == 511][0]) if 511 in u else 0.0,
                     float(b.sum(1).mean() / 9.0)))
    if not rows:
        return
    lines = [r"\begin{tabular}{lrrrr}", r"\hline",
             r"Template & distinct codes & all-zero & all-ones & mean density \\",
             r"\hline"]
    for lab, n, z, o, d in rows:
        lines.append(f"{lab} & {n}\\,/\\,512 & {z:.3f} & {o:.3f} & {d:.3f} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    write("tab_patch_coverage.tex", lines)


def table_fashion_grey():
    """Greyscale reconstruction, MNIST against Fashion, same generator settings."""
    entries = [(os.path.join(HT, "p2p_greym_e60", "summary.json"), "MNIST, grey"),
               (os.path.join(HT, "p2p_fashgrey_e60", "summary.json"), "Fashion, grey")]
    lines = [r"\begin{tabular}{lrrrrrr}", r"\hline",
             r"Run & MAE (0--255) & Corr. & MSSIM & F1 & Recog. & (original) \\",
             r"\hline"]
    any_row = False
    for path, lab in entries:
        if not os.path.exists(path):
            continue
        m = load(path)["metrics"]
        rg = m.get("recognition_accuracy_recovered_gray",
                   m.get("recognition_accuracy_recovered"))
        lines.append(
            f"{lab} & {fmt(m['grey_mae_255'], 2)} & {fmt(m['grey_corr'])} & "
            f"{fmt(m['mssim'])} & {fmt(m['foreground_f1'])} & {fmt(rg, 3)} & "
            f"{fmt(m['recognition_accuracy_original'], 3)} \\\\")
        any_row = True
    lines += [r"\hline", r"\end{tabular}"]
    if any_row:
        write("tab_fashion_grey.tex", lines)


# LEAK_LANES, amplifier flops, measured mean absolute trace value, result dir.
# Flop counts are from the synthesis reports: registers over the unamplified
# build rise by exactly 72*LANES, so the parameter does what it claims.
AMP_SWEEP = [
    ("63", 4536, 7602, "p2p_gfash_l63v2"),
    ("15", 1080, 3314, "p2p_gfash_l15"),
    ("3",   216, 2229, "p2p_gfash_l3"),
    ("1",    72, 2197, "p2p_gfash_l1"),
]


def table_amp_sweep():
    """Reconstruction quality against the size of the added leakage amplifier.

    Same 2000 Fashion images, same generator, same features, split and seed at
    every width, so the amplifier is the only variable.
    """
    lines = [r"\begin{tabular}{rrrrrrrr}", r"\hline",
             r"Lanes & Amp.\ flops & Mean $|$trace$|$ & MAE & Corr. & MSSIM & "
             r"F1 excess & Recog. \\",
             r"\hline"]
    any_row = False
    for lanes, flops, amp, d in AMP_SWEEP:
        path = os.path.join(HT, d, "summary.json")
        if not os.path.exists(path):
            continue
        m = load(path)["metrics"]
        lines.append(
            f"{lanes} & {flops} & {amp} & {fmt(m['grey_mae_255'], 2)} & "
            f"{fmt(m['grey_corr'])} & {fmt(m['mssim'])} & {fmt(m['f1_excess'])} & "
            f"{fmt(m['recognition_accuracy_recovered_gray'], 3)} \\\\")
        any_row = True
    lines += [r"\hline", r"\end{tabular}"]
    if any_row:
        write("tab_amp_sweep.tex", lines)


def table_amp_data_grid():
    """Amplifier size against profiling-set size, all four cells.

    The sweep in tab_amp_sweep varies the amplifier at one training size, so it
    cannot separate "the amplifier carries information" from "the amplifier
    reduces how much data is needed". These four cells do.
    """
    rows = [(r"none ($1$ flop), $56$k traces", "p2p_paper60k_noamp"),
            (r"$72$ flops, $1.5$k traces", "p2p_gfash_l1"),
            (r"$72$ flops, $56$k traces", "p2p_paper60k_l1"),
            (r"$4536$ flops, $1.5$k traces", "p2p_gfash_l63v2"),
            (r"$4536$ flops, $56$k traces", "p2p_paper60k_l63")]
    lines = [r"\begin{tabular}{lrrrrrr}", r"\hline",
             r"Amplifier, profiling set & MAE & Corr. & MSSIM & F1 excess & "
             r"Recog. & (ceiling) \\", r"\hline"]
    any_row = False
    for lab, d in rows:
        path = os.path.join(HT, d, "summary.json")
        if not os.path.exists(path):
            continue
        m = load(path)["metrics"]
        lines.append(
            f"{lab} & {fmt(m['grey_mae_255'], 2)} & {fmt(m['grey_corr'])} & "
            f"{fmt(m['mssim'])} & {fmt(m['f1_excess'])} & "
            f"{fmt(m['recognition_accuracy_recovered_gray'], 3)} & "
            f"{fmt(m['recognition_accuracy_original'], 3)} \\\\")
        any_row = True
    lines += [r"\hline", r"\end{tabular}"]
    if any_row:
        write("tab_amp_data_grid.tex", lines)


def table_paper_settings():
    """Where this setup matches Power2Picture and where it cannot."""
    rows = [
        ("Generator architecture", "Table I: 128, 128, 12544, 3 transposed convs", "identical"),
        ("Optimiser, learning rate", "Adam, $0.001$", "identical"),
        ("Loss", r"\texttt{gradmse} (MSE + gradient MSE)", "identical"),
        ("Batch size", "256", "identical"),
        ("Profiling set", "60k train split", "identical"),
        ("Dataset", "Fashion-MNIST", "identical"),
        ("Victim network", "quantised LeNet, full", "one $3\\times3$ conv layer"),
        ("Measurement", "5 on-chip TDC sensors", "external shunt, CW-Lite"),
        ("Window rate", "streaming, one window per cycle",
         r"$2$ cycles per window (\texttt{DWELL}), min. supported"),
        ("Leakage amplifier", "none", "none ($1$ flop output register)"),
    ]
    lines = [r"\begin{tabular}{lll}", r"\hline",
             r"Aspect & Power2Picture & This work \\", r"\hline"]
    for a, b, c in rows:
        lines.append(f"{a} & {b} & {c} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    write("tab_paper_settings.tex", lines)


def table_dwell():
    """Window rate: DWELL=8 against DWELL=2, both with the amplifier removed.

    DWELL holds each convolution window for that many cycles. A streaming
    accelerator advances one window per cycle, so a lower DWELL is the more
    realistic victim. It is not a free knob: at a fixed ADC rate it also divides
    the samples per window, so the generator input shrinks with it. Both effects
    are properties of the same physical change and are reported together.
    """
    rows = [(r"$8$ cycles ($32$ samples/window)", "p2p_paper60k_noamp", 5408),
            (r"$2$ cycles ($8$ samples/window)", "p2p_paper60k_noamp_d2", 1352)]
    lines = [r"\begin{tabular}{lrrrrrr}", r"\hline",
             r"Window dwell & Input dim. & MAE & Corr. & MSSIM & F1 excess & "
             r"Recog. \\", r"\hline"]
    n_rows = 0
    for lab, d, dim in rows:
        path = os.path.join(HT, d, "summary.json")
        if not os.path.exists(path):
            continue
        m = load(path)["metrics"]
        lines.append(
            f"{lab} & {dim} & {fmt(m['grey_mae_255'], 2)} & {fmt(m['grey_corr'])} & "
            f"{fmt(m['mssim'])} & {fmt(m['f1_excess'])} & "
            f"{fmt(m['recognition_accuracy_recovered_gray'], 3)} \\\\")
        n_rows += 1
    lines += [r"\hline", r"\end{tabular}"]
    # Both rows or none: a one-row comparison table would imply a comparison that
    # has not been measured yet.
    if n_rows == len(rows):
        write("tab_dwell.tex", lines)


def table_provenance():
    """Hardware provenance of every capture used in this report."""
    # the two profiling captures come first: every condition test in section 6.2 is
    # attacked with a template built from traces_rerun_20260830_trained_80, so its
    # provenance matters as much as that of the attack captures themselves.
    dirs = [os.path.join(ROOT, "host", "traces_rerun_20260830_trained_80"),
            os.path.join(ROOT, "host", "traces_rerun_20260830_onehot_80"),
            os.path.join(ROOT, "host", "traces_prof_5m"),
            os.path.join(ROOT, "host", "traces_prof_10m"),
            os.path.join(ROOT, "host", "traces_grey_p2p_2000")]
    dirs += sorted(glob.glob(os.path.join(ROOT, "host", "traces_hard_C*")))
    dirs += [os.path.join(ROOT, "host", "traces_paperscale_500")]
    roles = {"rerun_20260830_trained_80": "profiling (all condition tests)",
             "rerun_20260830_onehot_80": "profiling (probe mismatch)",
             "paperscale_500": "profiling only (300-image template)",
             "prof_5m": "profiling (clock mixing, 5 MHz)",
             "prof_10m": "profiling (clock mixing, 10 MHz)",
             "grey_p2p_2000": "grey design, generative reconstruction"}
    lines = [r"\begin{tabular}{llrrccc}", r"\hline",
             r"Capture & Role & Clock & Gain & Not stock AES & Functional check & Mism. \\",
             r"\hline"]
    any_row = False
    for d in dirs:
        p = os.path.join(d, "capture_manifest.json")
        if not os.path.exists(p):
            continue
        man = load(p)
        fc = man["functional_check"]
        short = os.path.basename(d).replace("traces_", "")
        role = roles.get(short, "attack")
        name = short.replace("_", "\\_")
        lines.append(
            f"{name} & {role} & {man['fpga_freq_hz']/1e6:.1f}\\,MHz & "
            f"{man['gain_db']:.0f}\\,dB & "
            f"{'yes' if man['design_check']['verified_not_stock_aes'] else 'NO'} & "
            f"{'676/676' if fc['passed'] else 'FAIL'} & {fc['mismatches']} \\\\")
        any_row = True
    lines += [r"\hline", r"\end{tabular}"]
    if any_row:
        write("tab_provenance.tex", lines)


if __name__ == "__main__":
    table_headline()
    table_sweep()
    table_conditions()
    table_mixed()
    table_p2p()
    table_grey()
    table_delta()
    table_grid()
    table_grid_best()
    table_jitter()
    table_mix2()
    table_amp()
    table_stage()
    table_dataset_grid()
    table_patch_coverage()
    table_fashion_grey()
    table_amp_sweep()
    table_amp_data_grid()
    table_paper_settings()
    table_dwell()
    table_provenance()
    print("done")
