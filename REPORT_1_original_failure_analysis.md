# Final results — Power side-channel attack on a BNN accelerator (CW305 + ChipWhisperer-Lite)

Recreation of Wei et al., *"I Know What You See"* (ACSAC'18) on CW305 Artix-7 + CW-Lite,
porting the bnn-fpga line-buffer conv (Zhao et al.) and the §6/§7 attacks.

## What was achieved ✅

| Phase | Status |
|---|---|
| **P2 train** | BNN trained on MNIST (GPU): Model 1 (3×3) 93%, Model 2 (5×5) 98.3%, golden MLP 98.8%. 64 binarized layer-1 kernels exported. |
| **P3 attack code** | §5-replacement front-end, §6 background detection (Otsu threshold), §7 power template (KDTree + Algorithm 2). |
| **P3 in simulation** | **Reproduces the paper**: 3×3 template recog **0.935** (paper 0.898), 5×5 **0.785** (0.790); background 3×3 0.77, 5×5 0.655 (paper 0.816/0.646). |
| **P0 bench** | CW-Lite + CW305 capture proven (stock AES trace, synchronous ADC lock, trigger). |
| **P1 FPGA** | Layer-1 conv (HLS→Vivado→CW305), **bit-exact on silicon** (all-ones + real MNIST, multiple kernels). Trigger + synchronous power capture working. |

The full pipeline runs end-to-end on real hardware. The **algorithmic experiment is
reproduced** (simulation hits paper-level recovery).

## What did NOT work — real-silicon recovery ❌ (and why)

Running §6/§7 on **measured** CW305 power gave recog ~0.15 (paper ~0.90). Recovery fails
because the per-cycle power barely correlates with the image: corr(power, pixel-window)
≈ **0.1–0.2**, far below what recovery needs.

This was investigated exhaustively. **corr stayed ~0.1–0.2 across every change:**

- **Conv microarchitecture** (the leakage source): preload+mux → addressed line buffer →
  true shift-register line buffer (paper Fig 9c) → **16× parallel leakage amplifier**
  (distinct unmergeable lanes). All bit-exact on silicon; none raised corr.
- **Sampling**: synchronous 4×/cycle and async 21×/cycle oversampling (paper §5 style).
- **Feature**: window magnitude vs Hamming-distance (transition) activity; alignment/offset scans.
- **Capture**: gain 30–50 dB, averaging 50–500.

### Root cause: the measurement bench, not the design
The decisive evidence is the **16× amplifier**: multiplying the data-dependent switching
16× did **not** change correlation. So the limit is **not** leakage magnitude — it is
**per-cycle power extraction**:

1. **CW-Lite ADC = 105 MS/s, 4 samples/cycle (synchronous).** The paper used a **2.5 GHz**
   scope (~thousands of samples/cycle) and §5 *curve-fits the RC power transient of each
   cycle* to recover clean per-cycle power. With 4 samples/cycle that extraction is
   impossible — AC-coupling + power-distribution-network capacitance **smear neighbouring
   cycles together**.
2. **CW305 is a large XC7A100T.** A sub-mW conv's data-dependent power is a tiny fraction
   of total chip power (clock tree, static, USB). The paper used a small **Spartan-6 LX75
   on SAKURA-G**, an SCA-isolated board with an amplified, dedicated power-measurement path.

These are fundamental properties of the CW305 + CW-Lite bench, not fixable in RTL.

## Bottom line
- **Implementation is paper-faithful**: shift-register line buffer = bnn-fpga / Fig 9c;
  conv bit-exact on silicon; capture matches §5's intent (synchronous, and async tested).
- **Algorithmic reproduction succeeds** (simulation = paper-level recovery).
- **Real-silicon recovery is blocked by the CW305+CW-Lite SNR / per-cycle-extraction floor**,
  proven not to be a design/code defect (16× amplification had no effect).
- **Faithful real-silicon recovery would require** a SAKURA-G-class SCA target + a
  GHz-rate oscilloscope (the paper's bench), which CW-Lite cannot emulate.

## Repro
- Simulation (paper-level): `python attack/evaluate.py --ksize 3 --n-profile 300 --n-eval 200`
- Hardware: build `build/run_vivado.tcl` → `host/cw305_func_check.py` (bit-exact conv) →
  `host/cw305_bnn_capture.py` → `attack/run_on_hardware.py` (SNR-limited, as documented).
