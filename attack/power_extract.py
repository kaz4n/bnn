"""
P3 -- power extraction front-end.

Two extraction modes exist:

* direct: synchronous captures where each cycle has a fixed sample count.
* paper: paper Section 5 style analog cleanup:
  DC restoration, low-pass filtering, Pearson/template cycle alignment,
  RC exponential curve fitting, and trailing-power subtraction.

Paper mode needs several samples per cycle. One-sample RO-sensor captures are already
per-cycle features, so auto mode keeps them on the direct path.
"""
import numpy as np

try:
    from scipy import optimize, signal
except Exception:  # pragma: no cover - scipy is expected in this repo env
    optimize = None
    signal = None


def extract_per_cycle(traces, samples_per_cycle, n_cycles, reduce="sum"):
    """
    traces: [n_kernels, n_samples] raw synchronous trace.
    Returns [n_kernels, n_cycles] per-cycle power.
    For real CW traces, set samples_per_cycle = adc_mul and align cycle 0 to the trigger
    edge (the trigger spans exactly the conv, so sample 0 == cycle 0).
    """
    nk = traces.shape[0]
    usable = n_cycles * samples_per_cycle
    t = traces[:, :usable].reshape(nk, n_cycles, samples_per_cycle)
    if reduce == "sum":
        return t.sum(axis=2)
    if reduce == "mean":
        return t.mean(axis=2)
    if reduce == "peak":
        return t.max(axis=2)
    if reduce == "ptp":
        return np.ptp(t, axis=2)
    if reduce == "rms":
        return np.sqrt(np.mean(t * t, axis=2))
    if reduce == "abs_sum":
        return np.abs(t).sum(axis=2)
    raise ValueError(reduce)


def dc_restore_highpass_inverse(trace, sample_rate_hz=2.5e9, cutoff_hz=250.0):
    """Undo the paper's equivalent measurement high-pass filter approximation."""
    x = np.asarray(trace, dtype=np.float64)
    if signal is None or sample_rate_hz <= 0 or cutoff_hz <= 0 or x.size < 2:
        return x
    tau = 1.0 / (2.0 * np.pi * cutoff_hz)
    step = 1.0 / sample_rate_hz
    alpha = step / tau
    beta = np.exp(-step / tau)
    # h[0]=1, h[n>0]=-alpha*beta**n. Invert x = r*h with a first-order IIR.
    return signal.lfilter([1.0, -beta], [1.0, -beta * (1.0 + alpha)], x)


def low_pass_filter(trace, sample_rate_hz=2.5e9, cutoff_hz=60e6, order=5):
    """Paper uses a 60 MHz low-pass filter to suppress oscilloscope noise."""
    x = np.asarray(trace, dtype=np.float64)
    if signal is None or sample_rate_hz <= 0 or cutoff_hz <= 0 or x.size < 8:
        return x
    nyq = 0.5 * sample_rate_hz
    # CW-Lite analog captures cannot realize the paper's 60 MHz cutoff when sampling at
    # ~105 MS/s. Use the highest safe cutoff rather than silently disabling filtering.
    cutoff_hz = min(float(cutoff_hz), 0.45 * sample_rate_hz)
    if cutoff_hz >= 0.95 * nyq:
        cutoff_hz = 0.9 * nyq
    sos = signal.butter(order, cutoff_hz / nyq, btype="lowpass", output="sos")
    padlen = min(3 * (2 * len(sos) + 1), x.size - 1)
    return signal.sosfiltfilt(sos, x, padlen=padlen)


def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return -np.inf if denom <= 1e-18 else float(np.dot(a, b) / denom)


def align_cycle_windows(trace, samples_per_cycle, n_cycles, search_radius=None, template=None):
    """
    Locate cycle starts by correlating a one-cycle template around each nominal start.

    The paper chooses a typical one-cycle waveform manually. Here we use the average
    nominal cycle as the default template, which makes the step reproducible.
    """
    x = np.asarray(trace, dtype=np.float64)
    spc = int(samples_per_cycle)
    if spc < 3:
        return [(j * spc, (j + 1) * spc) for j in range(n_cycles)]
    usable = min(x.size, n_cycles * spc)
    chunks = x[:usable].reshape(-1, spc)
    if template is None:
        template = np.median(chunks, axis=0)
    template = np.asarray(template, dtype=np.float64)
    if template.size != spc:
        raise ValueError(f"template length {template.size} != samples_per_cycle {spc}")
    if search_radius is None:
        search_radius = max(1, spc // 2)
    windows = []
    last = 0
    for j in range(n_cycles):
        nominal = j * spc
        lo = max(last, nominal - search_radius)
        hi = min(x.size - spc, nominal + search_radius)
        best = nominal
        best_r = -np.inf
        for st in range(lo, hi + 1):
            r = _pearson(x[st:st + spc], template)
            if r > best_r:
                best_r = r
                best = st
        windows.append((best, best + spc))
        last = best + 1
    return windows


def reduce_cycle_segment(segment, reduce="sum", template=None):
    """Collapse one aligned cycle into one power feature."""
    seg = np.asarray(segment, dtype=np.float64)
    if reduce == "sum":
        return float(np.sum(seg))
    if reduce == "mean":
        return float(np.mean(seg))
    if reduce == "peak":
        return float(np.max(seg))
    if reduce == "ptp":
        return float(np.ptp(seg))
    if reduce == "rms":
        return float(np.sqrt(np.mean(seg * seg)))
    if reduce == "abs_sum":
        return float(np.sum(np.abs(seg)))
    if reduce == "template_amp":
        if template is None:
            raise ValueError("template_amp requires a cycle template")
        tpl = np.asarray(template, dtype=np.float64)
        if tpl.size != seg.size:
            tpl = np.interp(np.linspace(0, tpl.size - 1, seg.size), np.arange(tpl.size), tpl)
        s = seg - seg.mean()
        t = tpl - tpl.mean()
        den = float(np.dot(t, t))
        return 0.0 if den <= 1e-18 else float(np.dot(s, t) / den)
    if reduce == "residual_rms":
        if template is None:
            raise ValueError("residual_rms requires a cycle template")
        tpl = np.asarray(template, dtype=np.float64)
        if tpl.size != seg.size:
            tpl = np.interp(np.linspace(0, tpl.size - 1, seg.size), np.arange(tpl.size), tpl)
        s = seg - seg.mean()
        t = tpl - tpl.mean()
        den = float(np.dot(t, t))
        amp = 0.0 if den <= 1e-18 else float(np.dot(s, t) / den)
        r = s - amp * t
        return float(np.sqrt(np.mean(r * r)))
    raise ValueError(reduce)


def _rc_model(t, base, amp, tau_rise, tau_fall, peak):
    peak = max(float(peak), 1.0)
    tau_rise = max(float(tau_rise), 1e-6)
    tau_fall = max(float(tau_fall), 1e-6)
    y = np.empty_like(t, dtype=np.float64)
    left = t <= peak
    y[left] = base + amp * (1.0 - np.exp(-t[left] / tau_rise))
    y_peak = base + amp * (1.0 - np.exp(-peak / tau_rise))
    y[~left] = base + (y_peak - base) * np.exp(-(t[~left] - peak) / tau_fall)
    return y


def fit_rc_cycle(segment):
    """Fit one cycle to a charge/discharge RC curve and return model parameters."""
    y = np.asarray(segment, dtype=np.float64)
    n = y.size
    if optimize is None or n < 4 or np.allclose(y, y[0]):
        return None
    t = np.arange(n, dtype=np.float64)
    base0 = float(np.percentile(y, 10))
    peak0 = int(np.argmax(np.abs(y - base0)))
    amp0 = float(y[peak0] - base0)
    if abs(amp0) < 1e-12:
        amp0 = float(y.max() - y.min()) or 1.0
    p0 = [base0, amp0, max(1.0, n / 8.0), max(1.0, n / 3.0), max(1.0, peak0)]
    bounds = ([-np.inf, -np.inf, 1e-3, 1e-3, 1.0],
              [np.inf, np.inf, max(1.0, n * 4.0), max(1.0, n * 8.0), max(1.0, n - 1.0)])
    try:
        popt, _ = optimize.curve_fit(_rc_model, t, y, p0=p0, bounds=bounds, maxfev=200)
        return popt
    except Exception:
        return None


def generate_trailing_trace(params, cycle_len, tail_len):
    if params is None or tail_len <= 0:
        return np.zeros(max(0, tail_len), dtype=np.float64)
    base, amp, tau_rise, tau_fall, peak = params
    end_t = max(cycle_len - 1, 0)
    y_end = _rc_model(np.asarray([end_t], dtype=np.float64), *params)[0]
    t = np.arange(1, tail_len + 1, dtype=np.float64)
    return (y_end - base) * np.exp(-t / max(float(tau_fall), 1e-6))


def extract_per_cycle_paper(traces, samples_per_cycle, n_cycles, sample_rate_hz=2.5e9,
                            lowpass_cutoff_hz=60e6, dc_cutoff_hz=250.0,
                            do_dc_restore=True, do_lowpass=True, do_align=True,
                            do_curve_fit=True, tail_cycles=3, reduce="sum",
                            min_curve_fit_spc=8):
    """Paper Section 5 per-cycle power extraction for multi-sample analog traces."""
    traces = np.asarray(traces)
    nk = traces.shape[0]
    spc = int(samples_per_cycle)
    if spc < 4:
        return extract_per_cycle(traces, spc, n_cycles, reduce=reduce)

    out = np.zeros((nk, n_cycles), dtype=np.float32)
    use_curve_fit = do_curve_fit and spc >= int(min_curve_fit_spc)
    tail_len = max(0, int(tail_cycles * spc)) if use_curve_fit else 0
    for ki in range(nk):
        x = traces[ki].astype(np.float64, copy=True)
        if do_dc_restore:
            x = dc_restore_highpass_inverse(x, sample_rate_hz, dc_cutoff_hz)
        if do_lowpass:
            x = low_pass_filter(x, sample_rate_hz, lowpass_cutoff_hz)
        if do_align:
            windows = align_cycle_windows(x, spc, n_cycles)
            usable = min(x.size, n_cycles * spc)
            template = np.median(x[:usable].reshape(-1, spc), axis=0)
        else:
            windows = [(j * spc, (j + 1) * spc) for j in range(n_cycles)]
            usable = min(x.size, n_cycles * spc)
            template = np.median(x[:usable].reshape(-1, spc), axis=0)
        work = x.copy()
        for j, (st, ed) in enumerate(windows):
            if ed > work.size:
                break
            seg = work[st:ed]
            if use_curve_fit:
                params = fit_rc_cycle(seg)
                trail = generate_trailing_trace(params, ed - st, min(tail_len, work.size - ed))
                out[ki, j] = float(seg.sum() + trail.sum())
                if trail.size:
                    work[ed:ed + trail.size] -= trail
            else:
                out[ki, j] = reduce_cycle_segment(seg, reduce, template)
    return out


def remap_extended_to_output(pc, ew, ksize, line=28, cycle0_offset=0):
    """Map per-cycle power of the extended-raster conv (EW*EW cycles) to the 28x28 output
    grid the attack uses. Output pixel (oy,ox) is emitted at extended cycle
    (oy+K-1)*EW + (ox+K-1)."""
    nk = pc.shape[0]
    kk = ksize - 1
    first = kk * ew + kk - int(cycle0_offset)
    last = (line - 1 + kk) * ew + (line - 1 + kk) - int(cycle0_offset)
    if first < 0 or last >= pc.shape[1]:
        raise ValueError(
            "capture does not contain the complete first-to-last output span: "
            f"need cycle indexes {first}..{last}, have 0..{pc.shape[1] - 1}. "
            "The legacy capture_cycles='output' files with n_cycles=784 are "
            "truncated for the extended-raster design."
        )
    out = np.zeros((nk, line * line), dtype=np.float32)
    for oy in range(line):
        for ox in range(line):
            idx = (oy + kk) * ew + (ox + kk) - int(cycle0_offset)
            if 0 <= idx < pc.shape[1]:
                out[:, oy * line + ox] = pc[:, idx]
    return out


def load_and_extract(npz_path, reduce="sum", frontend="auto", sample_rate_hz=None,
                     lowpass_cutoff_hz=None, dc_cutoff_hz=250.0,
                     do_dc_restore=True, do_lowpass=True, do_align=True,
                     do_curve_fit=True, tail_cycles=3, min_curve_fit_spc=8):
    d = np.load(npz_path)
    spc = int(d["samples_per_cycle"])
    ncyc = int(d["n_cycles"])
    source = str(d["trace_source"]) if "trace_source" in d.files else ""
    clock_mode = str(d["clock_mode"]) if "clock_mode" in d.files else ""
    if sample_rate_hz is None:
        sample_rate_hz = float(d["sample_rate_hz"]) if "sample_rate_hz" in d.files else 2.5e9
    if lowpass_cutoff_hz is None:
        lowpass_cutoff_hz = 60e6
    # The paper Section 5 front-end (DC restore / low-pass / align / RC curve-fit) exists
    # ONLY to reconstruct per-cycle power from an ASYNCHRONOUS smeared analog trace. It must
    # NOT run on phase-locked synchronous captures (SETUP_PLAN C3): those already give clean
    # per-cycle samples, and the §5 filters would be computed for the wrong sample rate.
    # So in auto mode, use Section 5 only for explicitly asynchronous analog captures.
    # A sync-x4 capture is still tagged "analog" by the host, but must stay on the direct
    # phase-locked path.
    is_async_analog = source == "analog" and clock_mode not in ("sync-x4", "sync")
    use_paper = frontend == "paper" or (frontend == "auto" and is_async_analog)

    # New captures retain every repetition as [kernel, repeat, sample]. Extract/alignment
    # happens per repeat, then features are combined. This avoids irreversible pointwise
    # averaging of independently-clocked asynchronous traces.
    raw = d["repeats"] if "repeats" in d.files else d["traces"]
    if raw.ndim == 2:
        repeat_views = [raw]
    elif raw.ndim == 3:
        repeat_views = [raw[:, ri, :] for ri in range(raw.shape[1])]
    else:
        raise ValueError(f"expected traces [kernel,sample] or repeats "
                         f"[kernel,repeat,sample], got {raw.shape}")

    extracted = []
    for trace_view in repeat_views:
        if use_paper:
            one = extract_per_cycle_paper(trace_view, spc, ncyc, sample_rate_hz,
                                          lowpass_cutoff_hz, dc_cutoff_hz,
                                          do_dc_restore, do_lowpass, do_align,
                                          do_curve_fit, tail_cycles, reduce,
                                          min_curve_fit_spc)
        else:
            one = extract_per_cycle(trace_view, spc, ncyc, reduce)
        extracted.append(one)
    pc = extracted[0] if len(extracted) == 1 else np.median(np.stack(extracted), axis=0)
    capture_cycles = str(d["capture_cycles"]) if "capture_cycles" in d.files else "extended"
    # "output" in legacy host files means a physical interval beginning at the first
    # output; it is not a compacted output-only stream. It therefore still needs the
    # extended-raster mapping. Only an explicitly compacted format may skip this step.
    if "ew" in d.files and capture_cycles != "output_compacted":
        ew = int(d["ew"]); ks = 2 * ((ew - 28) // 2) + 1
        cycle0_offset = int(d["cycle0_offset"]) if "cycle0_offset" in d.files else 0
        pc = remap_extended_to_output(pc, ew, ks, cycle0_offset=cycle0_offset)
    return pc, d["image"], int(d["label"]), d["kernel_ids"]
