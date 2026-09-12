#!/usr/bin/env python3
"""Generate every LaTeX table in the paper from the result JSONs.

No number in the paper is typed by hand. A result whose JSON is absent simply does not
get a row, so the paper can never quote a figure that was not produced. Same discipline
as report/catchup_v2_2026-08-30/make_tables.py.
"""
import json
import os

R = "experiments/rgb_sca/results"
OUT = os.path.join(os.path.dirname(__file__), "tables")
os.makedirs(OUT, exist_ok=True)

HLINE = r"\hline"
NL = r"\\"


def tex(v):
    """Escape LaTeX specials. Underscores in values like live_cw305_chipwhisperer are
    the common trap -- they abort the build with a hundred cascading errors."""
    out = str(v).replace("\\", "/")
    for ch in ("_", "&", "%", "#"):
        out = out.replace(ch, "\\" + ch)
    return out


def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def table(colspec, header, rows):
    return "\n".join([
        r"\begin{tabular}{" + colspec + "}",
        HLINE,
        header + " " + NL,
        HLINE,
        *rows,
        HLINE,
        r"\end{tabular}",
        "",
    ])


def write(name, text):
    open(os.path.join(OUT, name), "w").write(text)
    print(f"  wrote {name}")


# ---------------------------------------------------------- hardware results
d = load(f"{R}/hardware_20260912/results.json")
if d:
    rows = []
    for mode in ("summed", "serial"):
        for g in ("control", "natural"):
            r = d["modes"].get(mode, {}).get(g)
            if not r:
                continue
            t, p, s = r["trace"], r["prior_only"], r["summary"]
            cp = t["channel_permutation"]
            chroma = f"{s['chroma_advantage'][0]:+.2f}/{s['chroma_advantage'][1]:+.2f}"
            rows.append(
                f"{mode} & {g} & {t['mae_pooled']:.2f} & {p['mae_pooled']:.2f} & "
                f"{s['advantage_over_prior_pooled_mae']:+.2f} & "
                f"{s['luma_advantage']:+.2f} & {chroma} & "
                f"{cp['swap_penalty_ratio']:.3f} & "
                f"{sum(t['mssim_per_channel']) / 3:.3f} " + NL)
    write("tab_hardware.tex", table(
        "llrrrrcrr",
        "Dataflow & Group & Trace MAE & Prior MAE & Advantage & Luma adv. & "
        "Chroma adv. & Swap ratio & MSSIM",
        rows))

# ---------------------------------------------------------- data scaling
d = load(f"{R}/data_scaling.json")
if d:
    ns = sorted({int(n) for grp in d.values() for n in grp})
    rows = []
    for g in ("natural", "control"):
        if g not in d:
            continue
        cells = " & ".join(f"{d[g][str(n)]['mssim']:.3f}" if str(n) in d[g] else "--"
                           for n in ns)
        rows.append(f"{g} & {cells} " + NL)
    write("tab_scaling.tex", table(
        "l" + "r" * len(ns),
        "Group & " + " & ".join(f"$n={n}$" for n in ns),
        rows))

# ---------------------------------------------------------- anchor validation
rows = []
for tag, path in (("Scalar i.i.d.\\ Gaussian", f"{R}/anchor_validation_56k/results.json"),
                  ("Sampled real residuals", f"{R}/anchor_sampled_56k/results.json")):
    a = load(path)
    if a:
        rows.append(f"{tag} & {a['simulated']['mssim_mean']:.3f} & "
                    f"{a['hardware_anchor']['mssim']:.3f} & {a['abs_gap']:.3f} " + NL)
if rows:
    write("tab_anchor.tex", table(
        "lrrr", "Noise model & Simulated MSSIM & Measured MSSIM & Gap", rows))

# ---------------------------------------------------------- provenance
d = load(f"{R}/hardware_20260912/results.json")
if d:
    c = d["capture"]
    fc = c.get("functional_checks", {})
    rows = [
        f"Bitstream SHA-256 & \\texttt{{{c['bitstream_sha256'][:32]}\\ldots}} " + NL,
        f"Design signature & \\texttt{{{tex(c['design_signature'])}}} " + NL,
        f"Capture mode & \\texttt{{{tex(c['capture_mode'])}}} " + NL,
        f"FPGA clock & {c['fpga_freq_hz'] / 1e6:.1f}\\,MHz " + NL,
        f"ADC samples per cycle & {c['samples_per_cycle']} " + NL,
        f"Dwell & {c['dwell']} cycles " + NL,
        f"Gain & {c['gain_db']:.0f}\\,dB " + NL,
        f"Averaging & {c['avg']} " + NL,
        f"Deployed kernel index & {c['kernel_index']} " + NL,
        "Functional-check mismatches & "
        + ", ".join(f"{k}: {v}" for k, v in fc.items()) + " " + NL,
    ]
    write("tab_provenance.tex", table("ll", "Property & Value", rows))

print("done")
