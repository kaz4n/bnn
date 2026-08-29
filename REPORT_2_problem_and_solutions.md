# BNN Power Side-Channel Recreation: Why This Bench Falls Short, and How to Fix It

Recreation of Wei et al., *"I Know What You See"* (ACSAC '18), on a CW305 Artix-7 target with
a ChipWhisperer-Lite. This report states the problem, gives the evidence for the root cause,
lists what we tested and ruled out, and recommends hardware changes that would let the attack
succeed. The headline recommendation is a smaller, SCA-isolated FPGA.

Companion file `REPORT_1_original_failure_analysis.md` holds the earlier conclusion. This
report corrects and extends it.

---

## 1. What we set out to reproduce

The paper recovers MNIST input images from the power trace of a binarized CNN accelerator.
The attack targets the first convolution layer. Its line buffer processes one pixel per clock
cycle, so each cycle's power leaks the pixel window under the kernel. Two attacks read that
leak: background detection (Section 6) marks low-power cycles as background, and the power
template (Section 7) matches a per-cycle power vector across nine kernels to recover pixel
values.

The paper's bench and yours differ in the two ways that decide the outcome:

| | Paper (Wei et al.) | Your bench |
|---|---|---|
| Target FPGA | Spartan-6 LX75 | Artix-7 XC7A100T (about 100k logic cells) |
| Board | SAKURA-G (SCA-isolated, shunt + amplifier) | CW305 (SCA target, but a large die) |
| Scope | Tektronix at 2.5 GS/s | ChipWhisperer-Lite at 105 MS/s |

## 2. What works, and what does not

The attack software reproduces the paper in simulation. With clean traces at paper sizes (300
profiling images, 200 evaluation images), the template attack reaches recognition around 0.72
to 0.9 and background detection around 0.84, against the paper's 0.898 and 0.816. The full
hardware pipeline also works: the layer-1 conv computes bit-exact on silicon, the trigger
fires, and synchronous capture returns clean per-cycle samples.

Real-silicon recovery is where it breaks. Running the same attacks on measured CW305 power
gives template recognition around 0.13 and background around 0.255. The recovered images carry
almost no digit structure.

## 3. Root cause: signal dilution, not sampling

The per-cycle power barely tracks the image. The correlation between per-cycle power and the
pixel window sits near 0.1 on the external ADC and near 0.33 on an on-chip sensor. The attack
needs correlation near 0.8.

A simulation study pinned the mechanism. Adding a large, data-independent global-activity term
to the trace simulator (the clock tree, static current, and unused fabric of a big chip)
reproduces the measured collapse to 0.1 once the term reaches roughly 100 times the conv's own
per-cycle signal. The floor is the term's trace-to-trace variability, not its average. A
constant global offset cancels under mean subtraction; a global term that wobbles a few percent
between captures does not, and a few percent of a term 100 times larger than the signal still
buries it.

This reinterprets the earlier "16x amplifier had no effect" result. The prior report read that
as a sampling-bandwidth wall. It is a dilution signature. Amplifying the conv's switching 16x
raises both the wanted signal and the coherent contaminant that rides with it, so their ratio
holds and correlation does not move. The channel is not noise-limited; more averaging would
lift a noise-limited channel, and it does not.

The paper avoided this because its Spartan-6 LX75 is a small die and SAKURA-G routes the core
supply through a shunt into an amplifier. On that setup the conv is a large fraction of the
measured power. On the A100T the conv is a small fraction, so the same attack starts far below
the recovery threshold.

## 4. What we tested and ruled out on this bench

Every bench-side lever was tried and measured:

- **Trace averaging.** Correlation on the on-chip sensor rises to about 0.28 at 50 averages
  and plateaus near 0.33 at 800. Sixteen times more averaging buys 17 percent. Averaging
  cannot reach 0.8.
- **On-chip RO/TDC voltage sensor.** This reads local supply droop and gave correlation around
  0.33, roughly twice the external ADC. It remains dilution-limited because it sits on the same
  low-impedance power grid and reads the chip's global droop.
- **Sensor and conv co-location.** We built a bitstream that floorplans the sensor into the same
  clock region as the conv (226 sensor cells and 17,058 conv cells pinned to CLOCKREGION
  X0Y0:X0Y2). Measured correlation was 0.267, against 0.272 for the default placement. No change.
  Clock-region placement does not isolate local IR-drop over the conv.
- **Kernel averaging and spatial post-processing.** Averaging the sensor across nine kernels,
  Gaussian smoothing, and morphological cleanup raised background pixel accuracy from 0.52 to
  0.58 but left recognition near 0.24. Post-processing cannot recover signal the sensor never
  captured.
- **A finer oscillation-counting sensor.** A redesigned sensor that counts ring-oscillator
  cycles per window synthesizes out of context but gets pruned in the full design, because a
  counter clocked by an unconstrained ring-oscillator net is not a valid clock for the tools.
- **Clock-gating the idle fabric.** The design has no meaningful gateable logic during the
  trigger window. The USB and register logic already sit idle. The dilution comes from the
  clock tree, static current, and the conv's own data-independent control, none of which can be
  gated without stopping the conv.

The pattern is consistent. The bench is dilution-limited, and no change to sampling,
processing, or on-chip placement moves that limit.

## 5. Solutions, ranked by how directly they fix dilution

### 5.1 Use a smaller, SCA-isolated FPGA (the real fix)

Dilution is a ratio problem: the conv's power divided by the whole chip's power. Shrink the
denominator and the ratio rises toward the paper's regime. This is the paper's own design
choice, and it is the most reliable path.

Options, from closest-to-paper to most-modern:

- **Spartan-6 on SAKURA-G.** A direct reproduction of the paper's bench. If the goal is to
  match the paper exactly, this is it.
- **A small NewAE SCA target**, such as a CW312T-A35 daughterboard (Artix-7 XC7A35, about a
  third of the A100T) on a CW312/CW340 baseboard, or a CW305-A35 if you want to stay in the
  CW305 family. These route the FPGA core supply through a measurement shunt built for SCA, and
  the smaller die cuts the dilution denominator. Check the current NewAE catalog for exact part
  numbers and shunt options.
- **Any small Spartan or Artix on an SCA board** with a dedicated core-supply shunt and a
  low-noise measurement path.

The attack software and the trained BNN kernels port unchanged. Only the target board changes.
This is the recommendation if you want to reproduce the paper's result rather than study the
CW305's limits.

### 5.2 Add an EM near-field probe to the existing CW305

An H-field probe held over the conv region, into a low-noise amplifier, into the
ChipWhisperer-Lite measure input, couples to local current density rather than the shared
supply rail. It sidesteps the global droop that defeated the on-chip sensor, because it reads
the field above the conv logic rather than the grid voltage the whole die shares.

This is the cheapest change that targets the actual mechanism, at the cost of a probe and an
LNA (roughly a few hundred dollars) plus the effort of positioning. The pblock work from this
session already pins the conv to a known clock region, which narrows where to place the probe.
Expect to iterate on probe position and height.

### 5.3 Higher-resolution capture (CW-Husky)

A ChipWhisperer-Husky gives 12-bit samples, a higher rate, and streaming, against the Lite's
10-bit path. This helps the extraction axis, so it improves any channel's SNR. On its own it
does not address dilution, so it will not bridge the gap on the A100T. Pair it with 5.1 or 5.2.

### 5.4 The combination that matches a modern reproduction

A small SCA-isolated Artix or Spartan target, measured through a shunt, captured on a Husky,
reproduces the paper's setup with current hardware. That is the configuration most likely to
land recognition in the paper's 0.8 to 0.9 band.

## 6. Recommendation

Pick by goal.

To reproduce the paper's result, buy a small SCA-isolated target. A CW312T-A35 (or a SAKURA-G
with a Spartan-6) is the pragmatic choice, and the dilution arithmetic works in your favor on a
smaller die.

To salvage the CW305 you already own, add an EM probe and LNA. It is the one bench change that
attacks dilution rather than working around it, and the conv is already floorplanned to help
you find the spot.

Either way, the attack code is finished and validated. Only the measurement front end needs to
change. The simulation shows the algorithm recovers images once the channel reaches the
correlation a smaller die or a local probe provides.

---

## Appendix A: The portable software win

The template attack (Section 7) had regressed in the working tree. Its candidate metric had
changed to an L1 distance with strict intersection, and at the tuned radius that starved the
candidate sets and dropped clean-sim recognition to 0.32. Restoring the validated squared-L2
distance with the union fallback at radius 0.1 brought clean-sim recognition back to 0.72 at
300 profiling and 50 evaluation images. This fix is now the default in `attack/evaluate.py` and
`attack/run_on_hardware.py`, and it holds regardless of which measurement front end you choose.

The `run_on_hardware.py --invert` flag handles a droop sensor's polarity: more switching lowers
an RO count, so background reads as high count and needs the sign flipped before Section 6.

## Appendix B: Build-flow notes for future rebuilds

These cost real time this session, so they are worth recording.

- Launch Vivado from PowerShell, not a git-bash background shell. The `launch_runs` worker
  process fails to spawn from git-bash and stalls at the synthesis queue.
- Do not define `NO_RO_SENSOR` in `cw305/cw305_reg_bnn.v`. That macro disables the sensor and
  makes every rebuild sensorless. It was a leftover from the external-capture path.
- Ring-oscillator loops need `BITSTREAM.GENERAL.CRC DISABLE` to load on the CW305, since the
  loops break the configuration readback CRC. `cw305/allow_ro_loops.tcl` handles this.
- When implementing from a synthesis checkpoint, re-read the pin XDCs first. The synth
  checkpoint carries timing constraints but not the physical LOC and IOSTANDARD constraints,
  and the bitstream DRC fails without them.
- The laptop's Modern Standby suspends long builds, and the programmatic keep-awake is
  unreliable. Running implementation from a completed synthesis checkpoint keeps each stage
  short enough to finish between suspends.
