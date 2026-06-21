"""
P3 -- power extraction front-end (REPLACES paper S5).

Paper S5 (DC restore, curve fitting, alignment) exists only because they sampled
ASYNCHRONOUSLY at 2.5 GHz and had to reconstruct per-cycle power from a smeared analog
trace. ChipWhisperer-Lite samples SYNCHRONOUSLY (phase-locked to the device clock), so
each clock cycle maps to a fixed integer number of samples and we recover per-cycle
power directly by reducing each cycle's samples. No DC restoration / curve fitting.

This is the only place the capture method matters; S6/S7 are unchanged.
"""
import numpy as np


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
    if reduce == "peak":
        return t.max(axis=2)
    raise ValueError(reduce)


def load_and_extract(npz_path, reduce="sum"):
    d = np.load(npz_path)
    pc = extract_per_cycle(d["traces"], int(d["samples_per_cycle"]),
                           int(d["n_cycles"]), reduce)
    return pc, d["image"], int(d["label"]), d["kernel_ids"]
