# Recreating "I Know What You See" on ChipWhisperer-Lite (CW1173) + CW305 Artix-7

Porting the power side-channel attack of Wei et al. (ACSAC '18) from the paper's
original setup (Spartan-6 LX75 / SAKURA-G / Tektronix MDO3034 @ 2.5 GHz) to your
setup (CW305 Artix-7 target + ChipWhisperer-Lite capture, Vivado/Vivado HLS 2016.4).

---

## Part 1 — Investigation results (the "missing parts", resolved)

### 1.1 Did the paper edit bnn-fpga? — YES, heavily.

Stock `cornell-zhang/bnn-fpga`:
- Target: **Zedboard (Zynq-7000 SoC, ARM + fabric)**, flow = **Xilinx SDSoC** 2016.4/2017.1.
- Dataset: **CIFAR-10 only**. Input 32×32×3. VGG-style 6 conv-layer net
  ({128-128-256-256-512-512} channels) + FC layers.
- First conv layer is hardwired **3×3**.
- Weights shipped as zip (Google Drive); the host binarizes + reorders them at runtime.

Paper's target model (Table 1):
- Dataset: **MNIST**, input **28×28×1**.
- **4 layers**, first layer **64 kernels**, kernel **3×3 (Model 1)** and **5×5 (Model 2)**.
- Line buffer: line size 28, input channel 1.

=> The paper retargeted bnn-fpga from CIFAR-10 to MNIST: changed input dims
(32×32×3 → 28×28×1), reduced depth (6 conv → 4 layers), set layer-1 to 64 kernels,
and **added a 5×5 conv variant** (stock bnn-fpga has no 5×5). They did NOT ship these
edits or the weights — we rebuild them.

### 1.2 Did the paper edit / use BinaryNet? — Used in TWO different roles. Resolved.

`MatthieuCourbariaux/BinaryNet` (Theano/Lasagne):
- `mnist.py` = **MLP** (3 hidden FC layers, ~4096 units). **No conv layer.** ~0.96% error.
- `cifar10.py` = VGG-style **ConvNet** (the architecture bnn-fpga implements).
- `svhn.py` = ConvNet.

Key resolution of the confusion:
- The BinaryNet **MNIST MLP is the paper's "golden reference" classifier**, NOT the
  attacked network. Paper §6.3: *"we use a multi-layer perceptron network [12] with an
  accuracy of 99.2% as a golden reference to evaluate the cognitive quality."* That is
  the BinaryNet MLP. It is only used in software to score recovered images (recognition
  accuracy).
- The **attacked** network is the paper's own custom **4-layer binarized CNN on MNIST**.
  It is neither BinaryNet-mnist (MLP) nor bnn-fpga-CIFAR (6 conv). It reuses the
  BinaryNet *binarization method* but with a custom small architecture.

### 1.3 Which dataset trained on? — MNIST. Settled.

Table 1 reports 99.42% / 99.27% test accuracy "on MNIST datasets", 1st layer =
convolution. Training and testing are both MNIST. (CIFAR-10 is named only in §B as
future work.) No ambiguity.

### 1.4 Net consequence for you

You must supply, from scratch:
1. A binarized 4-layer MNIST CNN (layer-1 = 64× 3×3) — train it yourself.
2. An RTL/HLS layer-1 line-buffer conv unit retargeted to 28×28×1.
3. A CW305 host/register harness (replaces the SDSoC ARM data movers — which do not
   exist on bare Artix-7).
4. A synchronous-capture re-implementation of the attack (paper §5 is replaced, §6/§7
   port directly — see Part 2).

---

## Part 2 — Three hard constraints that reshape the whole build

### C1. You have Vivado + Vivado HLS, NOT SDSoC.

bnn-fpga's stock build is SDSoC-only, so it cannot build as-is on ANY board with your
tools. Path forward:
- Run **Vivado HLS 2016.4** on the accelerator C++ (`cpp/accel/`) to synthesize RTL.
- Wrap the emitted RTL in a **plain Vivado 2016.4** project.
- Gotcha: the HLS source contains `#pragma SDS ...` directives (SDSoC-specific). Plain
  HLS ignores them — you must add your own `#pragma HLS INTERFACE` (AXI-Lite / ap_ctrl)
  pragmas, or for the scoped layer-1-only build, simple `ap_none`/BRAM interfaces.

### C2. CW305 is bare Artix-7 (XC7A100T-2FTG256), no ARM core.

SDSoC auto-generated every ARM↔fabric DMA mover. None of that exists on CW305. You
rebuild host I/O over the **CW305 USB register interface**:
- Use ChipWhisperer's CW305 reference Verilog (`cw305_top.v`, `cw305_usb_reg_fe`, the
  register-bus example) as the shell.
- Map registers: write the 28×28 input image + the 64 kernels into BRAM via registers;
  a "start" register kicks off layer-1 conv; assert the **trigger pin (`tio_trigger`/
  `T_trig`)** at conv start; read the output feature maps back via registers.
- A100T fabric ≥ Zynq-7020, so resource fit is not the blocker — the **interface
  rebuild** is the work.

### C3. ChipWhisperer-Lite is synchronous (~105 MS/s max) — paper §5 does NOT port, and that is GOOD.

- The paper's §5 (2.5 GHz async scope, DC-component restoration with T=0.4 ns, per-cycle
  curve fitting, power alignment) exists *only because* they sampled asynchronously and
  had to reconstruct per-cycle power from a smeared analog trace.
- CW-Lite does **synchronous, clock-phase-locked capture**: the ADC samples locked to
  the DUT clock (`HS2` clock fed back). You get **clean power per clock cycle directly**.
  So **§5 is replaced by CW's synchronous capture, not reproduced** — a simplification.
- Action: **clock the BNN down to ~10–25 MHz** so synchronous capture is comfortable.
  CW-Lite can sample at an integer multiple of the DUT clock (e.g. ADC = 4× clock) for a
  few samples/cycle; sum/peak per cycle to get the per-cycle power scalar the attack
  needs. Verify exact max ADC rate + clkgen multiplier in CW docs for your firmware.
- **§6 (background detection) and §7 (power template) port cleanly.** The line buffer
  still processes **1 pixel/cycle**, so the cycle ↔ pixel-window mapping that both
  attacks depend on is fully intact. Only the front-end (how you turn a trace into a
  per-cycle power vector) changes.

---

## Part 3 — Scoping decisions (what makes this tractable)

1. **Build layer-1 ONLY in hardware.** The attack targets solely the first conv layer's
   line buffer. The recognition metric runs a *separate software classifier* on the
   recovered input image — the FPGA never needs to output a class. Layers 2–4 are
   unspecified in the paper and irrelevant to input recovery. Skip them in RTL.
2. **3×3 (Model 1) first.** bnn-fpga is hardwired 3×3 and Model 1 gives the best results
   (89.8% template / 81.6% background). Defer 5×5 (Model 2, needs modified conv unit,
   worse results) until 3×3 works end-to-end.
3. **Do NOT block on 99.42% accuracy.** The attack works against *any* fixed, known set
   of 64 binarized kernels — profiling (active adversary) adapts to whatever is deployed.
   Train a reasonable BNN, deploy its real layer-1 kernels, use the BinaryNet MLP as the
   golden software classifier. Training quality ≠ attack quality.
4. **De-risk the weakest link FIRST** (Phase 0 below).

---

## Part 4 — Phased plan (ordered by risk)

### Phase 0 — Prove the CW305 capture path (do this before anything else)

Goal: a synchronous, triggered power trace off the CW305, captured by CW-Lite, in
software. If this bench path fails, nothing downstream matters.

Steps:
1. Install ChipWhisperer (Python `chipwhisperer` package + correct CW-Lite firmware).
2. Flash and run a **stock CW305 example** (the bundled AES-on-CW305 target) end-to-end:
   program the CW305 with the provided bitstream, capture a trace via `scope.capture()`,
   confirm trigger + synchronous sampling work.
3. Confirm: clkgen → CW305 X6 clock path, `scope.clock.adc_src`, trigger on `tio4`.

Exit criterion: a clean averaged AES power trace plotted from your CW305. Do not proceed
until this works.

### Phase 1 — Layer-1 line-buffer conv in HLS → Vivado → CW305

1. Extract the layer-1 conv (line buffer + 3×3 binarized MAC + binarize/sign activation)
   from `cpp/accel/`. Strip SDSoC pragmas; reconfigure for **28×28×1 input, line size 28,
   64 output kernels, 3×3**.
2. Vivado HLS 2016.4: C-sim against a golden software conv (bit-exact), then synthesize
   to RTL with explicit HLS INTERFACE pragmas.
3. In Vivado 2016.4: instantiate the RTL inside the **CW305 register-bus shell**. Wire:
   image BRAM (reg-written), kernel BRAM (reg-written), start reg, **trigger asserted for
   exactly the layer-1 conv window**, output BRAM (reg-read).
4. Constrain to XC7A100T-2FTG256, clock ~10–25 MHz, generate `.bit`.
5. Functional check over USB: write image+kernels, run, read back feature maps, compare
   bit-exact to the HLS C-sim.

Exit criterion: HW feature maps == software golden, and trigger fires on layer-1 conv.

### Phase 2 — Train the BNN + export artifacts

1. Train a binarized 4-layer MNIST CNN (layer-1 = 64× 3×3, sign activations, ±1 weights).
   - Easiest modern path: a Larq / PyTorch BNN reproduction of BinaryNet rather than the
     decade-old Theano repo (Theano + old CUDA is painful to stand up in 2026). The
     binarization math is identical; only the framework differs. (If you want strict
     fidelity to BinaryNet, the original Theano/Lasagne `cifar10.py` is the reference for
     the conv binarization layers — adapt its architecture to 28×28×1 / 4 layers.)
2. Export the **64 binarized 3×3 layer-1 kernels** in the bit-packed order the HW expects
   (match bnn-fpga's weight reorder, or pick your own order and make HW match it).
3. Train/obtain the **BinaryNet MNIST MLP** (or any ~99% MLP) as the **golden classifier**
   for the recognition-accuracy metric.
4. Pick 500 MNIST test images (paper: 500 total; 300 to build template, 200 to evaluate).

Exit criterion: kernels load into the Phase-1 HW and produce correct layer-1 outputs;
golden MLP scores clean MNIST at ~99%.

### Phase 3 — Port the attack (§5-replacement + §6 + §7)

**§5 replacement (front-end):** For each captured trace, convert to a **per-cycle power
vector**: with synchronous capture at N samples/cycle, take sum (or peak) per cycle.
Identify the conv window via the trigger (start known; length = total cycles for one
28×28 feature map ≈ image-pixel count + line-buffer fill latency). No DC restoration / no
curve fitting needed.

**§6 Background detection (passive adversary):**
- Capture one trace per kernel (or just one kernel — paper shows kernel choice barely
  matters).
- Build histogram of per-cycle power; pick threshold by max decrease in cycle count
  (Eq. 4). Paper's threshold landed at ~0.5 (rescale to your units).
- Cycles below threshold = background pixels → black/white silhouette.
- Metric: pixel-level accuracy (Eq. 5, background marker) + recognition accuracy via
  golden MLP. Paper: ~86% pixel, ~81.6% recognition (3×3).

**§7 Power template (active adversary):**
- Profile: capture power for each of **9 kernels** (paper uses 9, not all 64) over the
  300 template images. For each cycle, store (related pixels in the K×(K+1)=12-pixel
  window, power-feature-vector ρ of length 9). That is the power template.
- Attack (200 eval images): for each cycle, split ρ into groups (paper: 3 groups of 3),
  search template within distance δ (paper: δ=1.0), intersect groups → pixel candidates.
- Reconstruct: Algorithm 2 (greedy variance-minimizing selector), average candidates per
  pixel.
- Metric: pixel distance (Eq. 7) + recognition accuracy. Paper: ~89.8% recognition (3×3),
  template ~44 MB for 300 images.

Exit criterion: recovered images visually match Fig. 11; recognition accuracy in the
80–90% ballpark for 3×3.

### Phase 4 (optional) — Model 2 (5×5) and mammographic/DDSM extension

Only after 3×3 works. Requires a 5×5 conv unit (modify the line buffer to 5 line
registers + 5×5 MAC) and re-profiling. Expect lower accuracy (paper: 79% template,
64.6% background), matching the paper's reported degradation.

---

## Part 5 — Key numbers to reproduce (targets from the paper)

| Quantity | Paper value (3×3 / Model 1) |
|---|---|
| Layer-1 kernels | 64 |
| Kernels used for template | 9 |
| Template images / eval images | 300 / 200 (500 total) |
| Related pixels per cycle (3×3) | K×(K+1) = 12 |
| Background detection threshold | ~0.5 (rescale) |
| Template δ / grouping | δ=1.0, 3 groups of 3 |
| Pixel-level acc (background, 3×3) | ~86.2% |
| Recognition acc (background, 3×3) | ~81.6% |
| Recognition acc (template, 3×3) | ~89.8% |
| Golden classifier | BinaryNet MNIST MLP, ~99.2% |

---

## Part 6 — Biggest risks / unknowns

1. **Synchronous-capture SNR on layer-1 conv.** Layer-1 conv unit power is tiny
   (paper: 0.57 mW conv unit). On bare A100T without SAKURA-G's amplifier path, the
   per-cycle signal may be near CW-Lite's noise floor. Mitigations: average many traces
   per (image,kernel), low clock, minimize unrelated fabric activity during the trigger
   window (gate/idle everything except layer-1).
2. **Weight bit-ordering** between your trainer and the HW BRAM layout — easy to get
   silently wrong; verify with bit-exact C-sim vs HW.
3. **HLS pragma rework** after stripping SDSoC — budget real time here.
4. **CW-Lite ADC multiplier limits** — confirm max samples/cycle achievable at your DUT
   clock in CW docs; it caps your per-cycle resolution.

---

## Sources
- Paper: *I Know What You See* (ACSAC '18), arXiv:1803.05847.
- bnn-fpga: https://github.com/cornell-zhang/bnn-fpga (Zedboard/SDSoC, CIFAR-10).
- BinaryNet: https://github.com/MatthieuCourbariaux/BinaryNet (MNIST=MLP, CIFAR/SVHN=ConvNet).
- BinaryNet paper: https://arxiv.org/abs/1602.02830.
- ChipWhisperer docs: https://chipwhisperer.readthedocs.io/
