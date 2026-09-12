# RGB input recovery — Phase 0 (architectural feasibility, simulation only)

Started 11 September 2026.

**Scope.** This is the power side-channel RGB study. It is *not* the intentional Bayer
sampling benchmark in `experiments/rgb_instrumented/`, and results from the two must
never be pooled or compared. Phase 0 is **simulation only**: no bitstream was built or
programmed, no ChipWhisperer capture was performed, no FPGA setting was changed. Every
result record is stamped `"source": "simulation"`, `"hardware_used": false`.

## Why simulation first

The feasibility review (`report/RGB_FEASIBILITY_AND_EVIDENCE_AUDIT_2026-09-11.md`,
unresolved question 2) identified a study-design problem: on this bench a negative RGB
result cannot distinguish

- **architectural loss** — colour is destroyed by the first-layer channel accumulation, in
  which case no bench improvement ever helps; from
- **bench loss** — the CW-Lite samples 4×/cycle against the 2.5 GS/s used by Wei et al.,
  whose §5 recovers per-cycle power by curve-fitting each cycle's RC transient.

Phase 0 measures the architectural ceiling at zero noise, where bench loss is absent by
construction. A result here is an upper bound on anything the hardware can do.

## What is built

| File | Purpose |
|---|---|
| `rgb_forward.py` | RGB first-layer conv + Hamming-distance power model, three channel dataflows |
| `rgb_probe.py` | The two probes and their controls |
| `run_phase0.py` | Driver; writes `results.json` |
| `tests/` | 29 tests, mostly targeting the controls |

### The channel dataflow is the variable that matters

A first-layer convolution sums over input channels: 27 values (3×3×3) produce one
accumulator value. Whether the per-channel contributions stay observable depends on what
the hardware *registers*, which is an architecture choice:

- **`serial`** — channels iterate over cycles (C cycles per output pixel). Each channel
  occupies its own cycle. Common on resource-constrained FPGAs that reuse one MAC array.
- **`parallel`** — all C·K² products held in their own registers, one adder tree.
- **`summed`** — only the accumulator is registered; products combinational. Worst case.

### The two probes

1. **`distinguisher()`** — exact, no learning. Does permuting the colour channels change
   the power trace at all? If traces are bit-identical, channel identity is provably
   unobservable and the hypothesis is dead regardless of model or bench.
2. **`recovery_probe()`** — ridge regression from local power features to the centre
   pixel's (R,G,B), scored against two controls absent from the published literature:
   **prior-only** (best constant predictor from training data) and **shuffled** (real
   traces paired with the wrong images). Split is by image, never by pixel.

## Phase 0 result (2026-09-11)

`results/phase0_20260911/results.json`. 4 source photos at 32×32 + 16 rolled copies,
9 random ±1 kernels, 3×3.

**Probe 1 — channel identity is observable in all three dataflows**, including the
worst case, with a control that confirms the mechanism:

| Dataflow | Trace changes under R↔B swap? | Same, with channel-*symmetric* kernels |
|---|---|---|
| `serial` | yes (80–94% of cycles) | **yes** — separability comes from cycle ordering |
| `parallel` | yes | **no** — identical traces |
| `summed` | yes | **no** — identical traces |

The control is the informative half. Where only sums are registered, channel identity is
carried by the *kernel's per-channel weight asymmetry*: if k[R]=k[G]=k[B], permuting
channels leaves the accumulator unchanged and colour is invisible. Trained first-layer
kernels are generically asymmetric; one-hot probe kernels may not be.

**Probe 2 — a linear read-out beats the prior in every dataflow, ordered as predicted.**
Pooled MAE advantage over prior-only, 0–255 units:

| Dataflow | noise 0.0 | 0.1 | 0.5 | 1.0 |
|---|---:|---:|---:|---:|
| `serial` | **+9.42** | +8.71 | +6.14 | +4.23 |
| `parallel` | +3.85 | +3.75 | +2.46 | +1.50 |
| `summed` | +1.56 | +1.59 | +1.40 | +0.95 |

The shuffled control collapses to prior-only in all cases (within 0.3%), confirming the
only signal path is the trace.

## What this does and does not establish

**Does:** per-channel information survives the first-layer accumulation and is accessible
to a *linear* read-out at zero noise. RQ1 is answered affirmatively at the architecture
level. A linear probe is deliberately weak, so this is a lower bound on extractable
information.

**Does not:**

- Say anything about the CW305/CW-Lite bench. The noise sweep is i.i.d. Gaussian, a crude
  stand-in — **not** a bench model. PDN smear, dilution, jitter and finite sampling are
  Phase 1 and deliberately absent so architectural and bench loss cannot be confused.
- Demonstrate reconstruction. Best pooled MAE is ~30/255 against a prior of ~40 — an
  advantage, nowhere near faithful colour recovery.
- Generalize across scenes. **There are 4 independent scenes.** The 16 rolled copies are
  not independent and are labelled as such in the result provenance. This is a pilot.
- Hold for blue. Under `parallel` and `summed` the per-channel advantage is concentrated
  in R (+8.25, +2.97); B is flat or slightly negative (−0.67, +0.64). Pooling hides this,
  which is why per-channel scores are never pooled away.

## Correction to the 11 September review

`report/RGB_FEASIBILITY_AND_EVIDENCE_AUDIT_2026-09-11.md` §C.1 argued that under channel
summing "channel identity is not separately observable from one accumulation." Probe 1
shows that is **too strong**: it holds only for channel-symmetric kernels. With asymmetric
weights the accumulator does change under a channel permutation. The review's conclusion
(RGB fidelity is unestablished by the supplied papers) is unaffected — but the stated
mechanism was wrong, and the corrected version is kernel-dependent, not universal.

---

# Phase 1 — calibrated against real hardware (2026-09-11)

`rgb_bench.py`, `run_phase1.py`, `results/phase1_20260911/`. Still simulation, but the
noise level is **measured**, not invented. Records are stamped
`source: simulation_calibrated_to_hardware` with the calibration capture's bitstream hash.

## Calibration

Read-only against `host/traces_gfash_noamp_train60k` — a real zero-amplifier grey capture
(`live_cw305_chipwhisperer`, `avg: 1`) on which the grayscale attack demonstrably works
(MSSIM 0.724, recognition 0.746). That makes it a **sufficiency anchor**: its noise level
is known to be enough for grayscale recovery.

Writing `feature = a·model + residual`, the measured correlation `r` fixes
`k = σ_noise/(a·σ_model) = sqrt(1/r² − 1)`.

| Featurization | corr | k |
|---|---:|---:|
| mean of all 32 samples (**currently used**) | 0.236 | 4.11 |
| samples 6–13 (**measured best**) | **0.347** | **2.70** |
| samples 8–11 | 0.333 | 2.83 |

**Actionable now, independent of RGB:** the data-dependent activity is confined to a few
samples early in each dwell period (corr 0.306 at sample 8, ~0.00 after sample 16).
Averaging the whole dwell dilutes it. Switching to samples 6–13 is a **1.47× correlation
gain for free** on the existing grayscale pipeline — no recapture, no rebuild.

Genuine inter-window smear is also present: measured lag-1 correlation is 0.330 where the
model's own window overlap predicts only 0.264, an excess of +0.066.

## Phase 1 result, and how to read it

Advantage over prior-only, pooled MAE, at the calibrated `k` and its averaging series
(`k_eff = k/√N`), 300 independent CIFAR-10 scenes:

| Arm | avg=1 | avg=10 | avg=50 | avg=100 | avg=500 |
|---|---:|---:|---:|---:|---:|
| **grayscale reference** | +0.45 | +1.08 | +1.31 | +1.35 | +1.39 |
| RGB `serial` | +0.04 | +0.29 | +0.73 | +1.03 | +1.88 |
| RGB `parallel` | +0.66 | +0.95 | +1.10 | +1.19 | +1.43 |
| RGB `summed` | +0.89 | +2.29 | +2.81 | +2.91 | +3.00 |

**The grayscale arm is the whole point of this table.** It scores only +0.45 at avg=1 —
yet the real attack on that exact capture reaches MSSIM 0.724. So the linear probe
**under-reads by a large factor**, and small absolute advantages here are not evidence of
infeasibility. Read the RGB rows *relative to the grayscale row*, not in absolute terms.
On that reading, `parallel` and `summed` are at or above the grayscale arm at every
averaging level.

**Honest limitations of Phase 1:**

- The linear ridge probe is not sensitive enough to be decisive at bench noise. The
  grayscale control proves this. A nonlinear reconstructor is required for a sharp answer —
  which is what the published attacks actually use.
- The dataflow ordering **reverses** from Phase 0 (`serial` best at zero noise, worst at
  avg=1). Mechanism: serial spreads the same work over 3× more cycles, so per-cycle signal
  is smaller. Whether that is a real disadvantage or an artifact of scaling noise
  per-dataflow is **not settled** — `k` was measured on a single-pass grey design and its
  transfer to a serial design is an assumption, not a measurement.
- Noise is modelled as a scalar SNR with an optional smear kernel. No PDN transfer
  function, thermal drift or trigger jitter.

---

# Phase 2 — hardware (in progress)

## Done: RGB core RTL, verified in simulation

`cw305/rgb_leakage_core.sv` + `cw305/tb_rgb_leakage_core.sv`. Three-channel 3×3 first
layer, 32×32 RGB input, 30×30 valid output, 8-bit pixels × ±1 weights. **No leakage
amplifier** — one registered `leak_sink` flop, matching the zero-amplifier grey baseline.

**The dataflow is a runtime mode, not a build-time parameter.** One bitstream measures all
three conditions. This is deliberate: reprogramming the CW305 over an already-configured
FPGA silently fails on this board, which is exactly how the 2026-09-06 amplifier sweep
produced four captures of the same bitstream (`attack/hardtests_20260830/AMP_SWEEP_VOID.md`).
Switching mode by register removes that failure mode.

The modes differ in *registered state*, not arithmetic — `MODE_PARALLEL` clocks all 27
product registers, `MODE_SUMMED` holds them so only the accumulator toggles, `MODE_SERIAL`
gives each channel its own dwell period (the design response to the measured smear).

xsim, IMG_SIDE=8, DWELL=4:

```
mode 0: 36 outputs in 433 cycles
mode 1: 36 outputs in 145 cycles
mode 2: 36 outputs in 145 cycles
PASS: all three modes bit-identical and matching the golden model
```

The mode invariant is load-bearing: if modes differed arithmetically, a cross-mode
hardware comparison would measure the wrong thing entirely.

## Done: register file, top level, build script, capture host

| File | What |
|---|---|
| `cw305/cw305_reg_leakage_rgb.sv` | RGB register file — 3072-byte image store, 27-bit kernel, mode register, signature page |
| `cw305/tb_cw305_reg_leakage_rgb.sv` | Integration testbench through the real USB register path |
| `cw305/cw305_leakage_rgb_top.sv` | CW305 top level |
| `cw305/cw305_leakage_rgb.xdc` | Names `crypto_clk` so the constraint is not silently dropped |
| `build/run_vivado_leakage_rgb.tcl` | Build; fails loudly on negative slack |
| `host/cw305_rgb_capture.py` | Capture with positive design identification and per-mode functional check |

**Image store: three planes, not one array.** Distributed RAM gives one read port per
copy, so a single 3072-byte array with 27 taps would replicate to 82,944 bytes of LUTRAM
and not fit. But every tap reads exactly one channel plane, and a plane is 1024 bytes — a
power of two. Three arrays with nine read ports each is 27,648 bytes, ~3.9× the grey
design rather than 27×. Channel select is `byteaddr[PLANE_W+1:PLANE_W]`, free wiring.

**Memory map collision caught in review.** Scaling the grey design's `OUT_BASE` to 4096
would have spanned `reg_address` 32–46 and collided with GO(33), STAT(34) and SIG(35),
which are decoded as `byteaddr >> 7`. The pages are now separated explicitly and the map
is documented in the register file header.

**Positive design identification.** The grey capture script proves the board is *not* the
stock AES image. That is weaker than it looks — any previously-loaded custom bitstream
also passes, which is how the amplifier sweep captured one bitstream four times. This
design answers `RGBSCA01` on page 35 and the capture script refuses to run otherwise.

xsim integration result:

```
RGB register file integration (IMG_SIDE=8, DWELL=4)
  signature OK: RGBSCA01
  mode 0: 36 outputs checked
  mode 1: 36 outputs checked
  mode 2: 36 outputs checked
PASS: signature, image round-trip and all three modes correct
```

`IMG_SIDE` is a parameter and the plane decode derives from it, so the small simulated
design exercises the identical logic as the synthesised 32×32 build.

## Line-buffer rework (2026-09-11) — fits, verified

The random-access design below did not fit. It was replaced by a line buffer, which is
also the structure Wei et al. actually attack. Out-of-context synthesis,
`xc7a100tftg256-2`, `IMG_SIDE=32`, `DWELL=8`:

| Resource | window-port | **line buffer** | grey baseline |
|---|---:|---:|---:|
| Slice LUTs | 73,174 (115%) | **1,572 (2.5%)** | 13,910 (22%) |
| F7 Muxes | 29,374 (93%) | 34 | — |
| Block RAM | 0 | 1.5 tiles | 0 |
| SRLC32E | 0 | 48 (line buffers) | — |
| DSPs | 0 | **0** | 0 |

46× smaller than the failed version, and smaller than the grey baseline.

| File | What |
|---|---|
| `cw305/rgb_linebuf_core.sv` | Line-buffer core; supersedes `rgb_leakage_core.sv` |
| `cw305/tb_rgb_linebuf_core.sv` | Unit testbench incl. segment framing |
| `cw305/cw305_reg_linebuf_rgb.sv` | BRAM pixel store; supersedes `cw305_reg_leakage_rgb.sv` |
| `cw305/tb_cw305_reg_linebuf_rgb.sv` | Integration testbench through the USB path |

Both pass xsim: all three modes bit-identical to the golden convolution, segment framing
exact, signature `RGBLBUF1`, pixel round-trip correct.

### Bitstream built and timing-clean (2026-09-11)

`build/cw305_leakage_rgblb_d8_s32.bit`, 703,804 bytes, sha256 `95b2d4d7cbbfc1d5…`

| | |
|---|---|
| WNS / WHS | **+3.829 ns / +0.065 ns** |
| Failing endpoints | **0** setup, **0** hold (3,993 analyzed) |
| Slice LUTs | 1,511 (2.38%) — 1,133 logic + 378 memory |
| Slice Registers | 899 (0.71%) |
| Block RAM | 1.5 tiles | 
| DSPs | 0 |

The `leakage_crypto_clk` constraint **was applied**, not silently dropped — zero
`Synth 8-3321` warnings, and the clock appears in the timing report with its 40 ns period
and 3,993 analyzed endpoints. That was the specific failure mode the grey design
documented, where the constraint named a net that did not exist at that level and the
convolution logic was timed against the wrong clock.

Hold slack of +0.065 ns is positive but tight; worth re-checking if `DWELL` or the clock
frequency changes.

### Three problems this rework surfaced, any of which would have invalidated results

1. **`prod_reg` was being deleted by synthesis.** The product registers drove nothing, so
   Vivado removed all 432 flops — which would have made `parallel` and `summed`
   electrically identical and silently destroyed the comparison the core exists to make.
   They are now folded into the `leak_sink` XOR reduction, which keeps them alive without
   touching `out_data`. Register count went 362 → 580.
2. **The MAC was mapping to DSPs.** A DSP48 has a different power signature from a LUT
   adder, and the grey baseline uses 0 DSPs, so the comparison would not have been like
   for like. `(* use_dsp48 = "no" *)` — note the 2016.4 spelling; `use_dsp` is silently
   ignored.
3. **A one-position lag in the line buffer.** Registering `pix_addr` *and* the BRAM output
   costs two cycles, so every window shifted in the previous pixel. Caught only because
   the testbench uses an asymmetric random image and checks every output against a golden
   model — a symmetric or constant test image would have passed.

### Trace length forced segmented capture

A full 32×32 scan at DWELL=8 is 32,768 ADC samples, and 98,304 in serial mode, against a
CW-Lite limit near 24,573. Shrinking the image or collapsing DWELL to 2 would have
destroyed the settling structure the measured featurization depends on. Instead the core
always scans the whole image — keeping the line buffer coherent — and frames only the
trigger over `[seg_start, seg_start+seg_len)`. The host reassembles a full scan from 2
segments (summed/parallel) or 5 (serial). Geometry and DWELL are preserved; only
acquisition time grows.

---

## Superseded: the random-access design DID NOT FIT

Out-of-context synthesis, `xc7a100tftg256-2`, `IMG_SIDE=32`, `DWELL=8`:

| Resource | Used | Available | % |
|---|---:|---:|---:|
| **Slice LUTs** | **73,174** | 63,400 | **115.4** |
| LUT as Logic | 72,844 | 63,400 | 114.9 |
| **LUT as Memory** | **330** | 19,000 | **1.7** |
| F7 Muxes | 29,374 | 31,700 | 92.7 |
| Slice Registers | 24,837 | 126,800 | 19.6 |
| Block RAM | 0 | 135 | 0.0 |

**The three-plane decomposition did not work, and the utilization says exactly why.**
`LUT as Memory` is 1.7% — Vivado inferred essentially **no distributed RAM at all**.
It built 27 large multiplexers out of logic instead (F7 Muxes at 92.7% is the tell).

The prediction of "~3.9× the grey design" was right in magnitude and wrong in
consequence: the grey design's nine 784-to-1 muxes already cost real area, and 27
1024-to-1 muxes overflow the part. The error was assuming an array with one write port
and nine *asynchronous, independently-addressed* read ports would infer LUTRAM. It does
not at this width.

### The fix is also the more faithful design: a line buffer

Random access into the image store was never the right structure. Wei et al. state the
attack target plainly — "the actual attack target is the structure of line buffer where
we exploit the power consumption with the sliding convolutional window" (§ Applicability).
A line buffer holds two image rows plus the 3×3 window in shift registers and advances
one column per output pixel:

- storage becomes 3 channels × 2 rows × 32 bytes ≈ 192 bytes of shift register plus 27
  window registers, instead of 27 wide random-access read ports;
- the image itself moves to block RAM (3072 bytes ≈ 1 tile, currently 0 used), streamed
  one byte per cycle;
- and the leakage structure becomes the one both supplied papers actually attack, rather
  than an approximation of it.

This is a redesign of the core's input interface (streaming pixels instead of a
`window_bytes` port), not a parameter change. The mode-register mechanism, signature
page, memory map, capture host and generator are all unaffected.

## Remaining

1. **Rebuild the core around a line buffer**, then re-run the out-of-context synthesis
   check before attempting a bitstream. The build script
   (`build/run_vivado_leakage_rgb.tcl`) already fails loudly on negative slack; it should
   not be run until utilization is under 100%.
2. **Program + capture** — **requires a human.** Reprogramming over a configured FPGA
   silently fails on this board: power-cycle it (cutting power; USB RST is not enough)
   and program while blank. The CW305 currently holds the Bayer benchmark bitstream.
   `cw305_rgb_capture.py` verifies the `RGBSCA01` signature before capturing and refuses
   otherwise, so a failed reprogram is a loud error rather than a quiet wrong result.
## Done: 3-channel generator with controls

`experiments/rgb_sca/rgb_generator.py`. Power2Picture's generator widened to three
channels and 32×32 (seed 8×8, upsample 8→16→32; layer types, kernel sizes, strides and
padding unchanged, so it is a widening rather than a redesign).

Three output channels prove nothing by themselves — that is a property of the last
transposed convolution, not of the trace — so the module runs **three arms together** and
refuses to present a headline number alone:

| Arm | Input | What it establishes |
|---|---|---|
| `trace` | real traces, real pairing | the claim |
| `prior_only` | every sample gets the *same* mean feature vector | the number the trace arm must beat |
| `shuffled` | real traces, wrong images | should match prior-only; if not, something other than the trace leaks |

Plus a **channel-permutation check** at evaluation: score against R↔B-swapped truth. If
the error does not rise, channel identity was not recovered no matter how good the pooled
score looks.

**The colour metric that matters.** Natural-image colour is strongly predictable from
luminance, so a model could recover structure from the trace and *colourise it from the
prior* — scoring well on pooled PSNR while carrying no colour information. Errors are
therefore split into luminance (Y) and chrominance (Cb, Cr), and `chroma_over_luma` is
reported. Recovering Y while failing Cb/Cr is grayscale recovery from an RGB-input
accelerator — a real finding, and explicitly **not** RGB reconstruction.

Featurization defaults to `settle` (the measured 1.47× window), expressed in
samples-per-window units so it tracks `dwell` and `samples_per_cycle`.

47 tests pass, including an end-to-end run driven by Phase 0 simulated traces that
asserts the controls behave: the trace arm beats prior-only, the shuffled arm does not,
and shuffled lands near prior-only. That validates the pipeline before any bitstream
exists.

## Run the hardware simulations

```bash
cd /c/Xilinx/Vivado/2016.4/bin && ./xvlog -sv cw305/rgb_leakage_core.sv cw305/cw305_reg_leakage_rgb.sv cw305/tb_cw305_reg_leakage_rgb.sv && ./xvlog cw305/cdc_pulse.v && ./xelab tb_cw305_reg_leakage_rgb -s regsim && ./xsim regsim -R
```

## Run

```bash
python -m pytest experiments/rgb_sca/tests -q
```

```bash
python -m experiments.rgb_sca.run_phase1 --n-images 300 --out experiments/rgb_sca/results/phase1_20260911
```

## Run

```bash
python -m pytest experiments/rgb_sca/tests -q
```

```bash
python -m experiments.rgb_sca.run_phase0 --size 32 --out experiments/rgb_sca/results/phase0_20260911
```

Supply your own scenes — strongly recommended, given the 4-scene limitation — with
`--images-npz FILE` holding an `images` array of shape `(N,3,H,W)`, uint8.

---

# HARDWARE RESULTS — 12 September 2026

**Real RGB reconstructions from measured CW305 power side-channel traces.**

Bitstream `cw305_leakage_rgblb_d8_s32.bit`, signature `RGBLBUF1`, **no leakage amplifier**.
8,022 traces (4,011 per dataflow), functional check **0 mismatches** in both modes — run on
a natural photograph, not a solid patch, so it actually exercises the line buffer.

Capture: `host/traces_rgb_full` · Analysis: `results/hardware_20260912`

## Results

| mode | group | advantage | luma adv | chroma adv | swap ratio | MSSIM |
|---|---|---:|---:|---|---:|---:|
| summed | control (tinted) | **+14.02** | **+15.91** | −0.80 / −0.65 | 0.994 | **0.56** |
| summed | natural (CIFAR-10) | −4.97 | −5.85 | +0.25 / −0.04 | 1.026 | 0.05 |
| serial | control | +11.35 | +13.80 | −1.62 / −2.38 | 0.985 | 0.52 |
| serial | natural | −3.79 | −4.62 | −0.08 / −0.10 | 1.031 | 0.06 |

## What this establishes

**1. Luminance recovery from real RGB-accelerator traces, better than forecast.**
+14.02 MAE advantage over the analytic prior, MSSIM 0.566, against a simulated forecast of
+9.53 / 0.462. The image grid shows recognizable garment silhouettes recovered from
physically measured power.

**2. Colour is not recovered.** Chroma advantage is negative in **five of six** cells
(summed/natural shows a small **positive** Cb advantage of +0.25), and on the **control**
group — the only group that reconstructs at all — the channel-permutation ratio is
0.985–0.998, i.e. scoring against channel-swapped truth is no worse than the correct
pairing. On the natural group the ratio exceeds 1.0 (1.026–1.031), but that group does
not reconstruct, so the ratio is not evidence of colour recovery there.

With random per-image tints, chroma is unpredictable from structure or prior, so this is
a controlled negative, not a modelling artifact.

> **Corrected 12 September 2026.** This paragraph previously read "negative in all four
> cells" and "ratio is at or below 1.0", contradicting the table directly above it and
> the underlying JSON. External review caught it. The conclusion — faithful colour
> recovery is unestablished — is unchanged, but the supporting description was wrong and
> overstated.
>
> A further caveat from the same review: a positive swap penalty is **not** by itself
> evidence of measurement-derived colour, because an input-independent constant predictor
> also incurs one when the dataset's channel distributions differ. `rgb_generator.py` now
> reports `prior_swap_penalty` and `excess_swap_penalty_over_prior` alongside it; only the
> excess is attributable to the measurement.

**3. Natural photographs fail** — worse than predicting the mean image (MSSIM 0.05). The
grayscale-CIFAR control established this is scene complexity, not colour.

**4. The serial dataflow did NOT help, contradicting simulation.** Phase 0 measured serial
as the most separable dataflow at zero noise, and the validated tint probe agreed. On
hardware it is worse than summed (+11.35 vs +14.02, chroma more negative). Trust the
hardware. Likely cause: serial spreads the same work over 3× the cycles, lowering
per-cycle signal — invisible to a zero-noise probe.

**5. The real bench beats the model.** Measured corr **0.523 → k = 1.63**, against the grey
reference design's 0.347 / k = 2.70. The line-buffer design is 9× smaller (1,511 vs 13,910
LUTs), so far less competing activity dilutes the signal. Future forecasts should
recalibrate on this design.

## Why the control group was essential

Without it, the natural-image null is uninterpretable — broken capture, or genuinely hard
problem? The control group recovers recognizable garments **from the same capture session**,
proving the bitstream, bench, capture path and generator all work. That is what converts a
null into a finding.

## Honest scope of the claim

This is **grayscale recovery from an RGB-input accelerator**, not RGB reconstruction. It is
a real and distinct result — and the pre-registered reading from the feasibility review's
§E.6, not a retrofit. Claiming RGB reconstruction would require a positive chroma advantage
and a swap penalty above 1.0; neither is present.

The measured SNR requirement for colour is ~4× better than the grey design and ~2.5× better
than this one. The most realistic lever is sampling rate: Wei et al. used 2.5 GS/s, the
CW-Lite gives 105 MS/s at 4 samples/cycle.
