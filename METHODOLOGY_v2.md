# Methodology v2 — why real-silicon recovery failed, and the power-only path forward

Supersedes the root-cause section of `FINAL_RESULTS.md`. Written after a no-hardware
simulation study (`attack/trace_sim.py` new flags; `scratchpad/sim_confound*.py`) that
isolated the actual failure mechanism. See memory `dilution-rootcause`.

## Corrected root cause

`FINAL_RESULTS.md` blamed **per-cycle power extraction / CW-Lite inter-cycle smear**. The
simulation says that is the *secondary* effect. What actually reproduces the measured
`corr(power, window) ≈ 0.1`:

| Mechanism (injected into the sim) | corr | Reaches HW 0.1? |
|---|---|---|
| baseline (clean) | 0.81 | — |
| inter-cycle smear, τ=10 cycles | 0.36 | no |
| **global-activity dilution, 10×** | **0.06** | **yes** |
| dilution 100× + common-mode subtraction, 5% per-trace jitter | 0.09 | yes |

The wall is **data-independent global activity** (clock tree, USB, unused fabric) that is
~100–1000× the 0.57 mW conv, and specifically its **trace-to-trace variability**:

```
dilute=100×, mean-subtracted, sweep per-trace jitter:
  0.0% → corr 0.57   (mean cancels, signal returns)
  1.0% → corr 0.26
  5.0% → corr 0.09   (= the hardware floor)
```

The **mean** of the global term cancels under common-mode subtraction; its **variability**
does not. This is why averaging never helped (it removes *random* noise, not the
*correlated* global term) and why the paper needed **SAKURA-G** (a tiny, SCA-isolated
Spartan-6 with no large global term) — the 2.5 GHz scope alone was never the deciding factor.

## Two prior conclusions that don't hold

1. **"16× amplifier had no effect → fundamental."** The amp scales the *conv* signal, so a
   genuinely working 16× amp would *raise* signal-vs-global and *help* against dilution.
   HLS did unroll 16 lanes (RTL confirmed), but whether Vivado kept them distinct and
   whether they moved *measured* power is **unverified**. Check it: `fanout_variance_check.py`.
   Until then the "16× null" is not load-bearing evidence.
2. **"Common-mode subtraction is the power-only fix."** Insufficient alone — it cancels the
   global mean, not its variability (see table). It only works *after* the global activity
   is made reproducible.

## Plan (gated; cheapest-decisive first)

**Done, no hardware:**
- [x] Sim confound study → dilution-variability is the mechanism.
- [x] `trace_sim.py` gains `smear_tau`, `dilution`, `dilution_jitter` (default 0 = old
      behavior) so future sim results model the real regime honestly.
- [x] `attack/tvla.py` — model-free leakage test (validated on sim).
- [x] `host/fanout_variance_check.py` — amp-reality bench check (ready).

**When the bench is connected:**
1. **`fanout_variance_check.py`** — build a `FANOUT=1` bitstream alongside the `f16` one
   (`set FANOUT=1` in `run_hls.tcl` / `run_vivado.tcl`), compare data-dependent energy.
   If ratio ≈ 1, the amp is broken → rebuild distinct lanes; this may itself lift recovery.
2. **`tvla.py --fixed DIR --random DIR`** on raw single traces. `|t| < 4.5` everywhere →
   dead channel, stop (power-only cannot win; EM probe or quieter target required).
   `|t| > 4.5` → channel alive, proceed. (Necessary, not sufficient — TVLA survives the
   common-mode term that kills recovery.)
3. **Fabric-quiescing + deterministic capture** (checklist below) → re-measure *recovery*.
4. **Lever A** (idle clocks between conv cycles; `run_*.tcl` "build both") → removes the
   secondary smear term and lets the PDN settle between cycles.

## Fabric-quiescing checklist — the power-only lever that targets the mechanism

Goal: during the trigger window, make competing fabric activity **small AND identical
every trace**, so its mean cancels under common-mode subtraction and its variability
approaches the conv's own. Sim (`tvla.py --sim`, jitter 5%→0%) shows this restores
separability.

- [ ] No register / USB traffic while the scope is armed. Write image+kernel, *then* arm,
      *then* pulse GO — never `fpga_write` during capture.
- [ ] Clock-gate or hold every block except layer-1 conv during the conv (no free-running
      counters, no unused BRAM churn, no debug/ILA cores in the design).
- [ ] Single clock domain; stop unused PLL outputs.
- [ ] Constant, held inputs on all unrelated pins.
- [ ] Fixed, deterministic capture timing: identical GO-to-trigger latency every trace
      (so the global activity lands in the same samples every time — that's what lets
      subtraction cancel it).
- [ ] Temperature/supply stable across the profiling **and** eval captures.
- [ ] **Interleave** profiling and eval capture order (do not capture 300 then 200 in
      separate time blocks — `run_on_hardware.py` currently splits `files[:300]`/`[300:]`,
      which are captured sequentially → drift confound).

## If power-only still falls short

The sim is unambiguous that spatial isolation is the reliable fix. With no EM gear on the
bench, an **H-field near-field probe + LNA into the CW measure-in** is the highest-ROI
acquisition — it removes the global term at the source rather than fighting its variability.
This is the concrete escalation if the checklist + amp-fix don't reach paper recovery.
