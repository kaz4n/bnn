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
# The parallel dataflow was captured in a separate session (same bitstream, mode is a
# runtime register) so its results live in their own file. Merge rather than duplicate.
d = load(f"{R}/hardware_20260912_rev3/results.json")
dp = load(f"{R}/hardware_parallel_rev3/results.json")
if d and dp:
    d = {**d, "modes": {**d["modes"], **dp.get("modes", {})}}
if d:
    rows = []
    for mode in ("summed", "parallel", "serial"):
        for g in ("control", "natural"):
            r = d["modes"].get(mode, {}).get(g)
            if not r:
                continue
            t, p, s = r["trace"], r["prior_only"], r["summary"]
            cp = t["channel_permutation"]
            chroma = f"{s['chroma_advantage'][0]:+.2f}/{s['chroma_advantage'][1]:+.2f}"
            # Report the EXCESS swap penalty over the no-input prior, not the raw one:
            # a constant predictor already scores +1.05 on CIFAR because the dataset's
            # channel distributions differ, so the raw figure overstates colour recovery.
            exc = cp.get("excess_swap_penalty_over_prior")
            exc_s = f"{exc:+.2f}" if exc is not None else "--"
            rows.append(
                f"{mode} & {g} & {t['mae_pooled']:.2f} & {p['mae_pooled']:.2f} & "
                f"{s['advantage_over_prior_pooled_mae']:+.2f} & "
                f"{s['luma_advantage']:+.2f} & {chroma} & "
                f"{cp['swap_penalty']:+.2f} & {exc_s} & "
                f"{sum(t['mssim_per_channel']) / 3:.3f} " + NL)
    write("tab_hardware.tex", table(
        "llrrrrcrrr",
        "Dataflow & Group & Trace MAE & Prior MAE & Advantage & Luma adv. & "
        "Chroma adv. & Swap & Excess & MSSIM",
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
d = load(f"{R}/hardware_20260912_rev3/results.json")
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

# ---------------------------------------------------------- corrected Phase 0
d = load(f"{R}/phase0_20260912_corrected/results.json")
old = load(f"{R}/phase0_20260911/results.json")
if d and old:
    rows = []
    for df in ("serial", "parallel", "summed"):
        o = old.get("recovery", {}).get(df, {}).get("noise_0.0", {})
        n = d.get("recovery", {}).get(df, {}).get("noise_0.0", {})
        if not o or not n:
            continue
        rows.append(f"{df} & {o['advantage_over_prior_pooled']:+.2f} & "
                    f"{n['advantage_over_prior_pooled']:+.2f} " + NL)
    if rows:
        write("tab_phase0.tex", table(
            "lrr",
            "Dataflow & Position split, defective & Scene-disjoint, corrected", rows))

# ---------------------------------------------------------- overfit functionality
d = load(f"{R}/overfit_check/results.json")
if d:
    rows = []
    for name in ("synthetic", "control", "natural"):
        a = d["arms"].get(name)
        if not a:
            continue
        c = a["loss_curve"]
        rows.append(f"{name} & {a['train_mae']:.2f} & {a['train_mssim']:.3f} & "
                    f"{c[0]['loss']:.3f} & {c[-1]['loss']:.4f} & "
                    f"{a['spread_ratio']:.2f} " + NL)
    if rows:
        write("tab_overfit.tex", table(
            "lrrrrr",
            "Arm & Train MAE & Train MSSIM & Loss start & Loss end & Spread ratio",
            rows))

print("done")
