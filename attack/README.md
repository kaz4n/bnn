# P3 — Power side-channel attack (S5-replacement + S6 + S7)

Reproduces the attack of Wei et al. "I Know What You See" on synchronous CW-Lite traces.
Runs **now** on simulated traces; swaps to real CW305 captures with no code change (same
`.npz` format). See `../SETUP_PLAN.md` Phase 3.

## Modules
| File | Paper section | Role |
|---|---|---|
| `trace_sim.py` | — | Hamming-distance CMOS power model → traces (no bench needed) |
| `power_extract.py` | **replaces S5** | synchronous per-cycle power (no DC restore/curve fit) |
| `common.py` | S7.2 | cycle↔pixel scan order, K×(K+1) related-pixel window |
| `attack_background.py` | **S6** | passive: threshold (Eq.4) → background silhouette (Eq.5) |
| `power_template.py` | **S7** | active: template build, candidate gen, Algorithm 2, Eq.7 |
| `evaluate.py` | S6.3/S7.3 | driver + metrics + recognition via golden MLP |

## Run
```bash
# quick demo (random kernels ok):
python evaluate.py --ksize 3 --n-profile 20 --n-eval 8
# faithful sizes (paper: 300 profile / 200 eval, 9 kernels, delta=1.0, 3 groups of 3):
python evaluate.py --ksize 3 --n-profile 300 --n-eval 200 --n-kernels 9 \
    --kernels ../training/artifacts/model_3x3/layer1_kernels.npy
python evaluate.py --ksize 5 ...   # Model 2
```
Outputs `results/recovered_*.npz` (original/background/template images) + `summary.json`
(metrics vs paper targets).

## Why S5 is replaced, not reproduced
Paper S5 (DC restoration, curve fitting, 2.5 GHz alignment) compensates an ASYNCHRONOUS
scope. CW-Lite samples SYNCHRONOUSLY (phase-locked to the device clock), so per-cycle
power is read directly by reducing each cycle's samples — a strict simplification that
loses no attack-relevant information (the line buffer still emits 1 pixel/cycle, so S6/S7
are identical). This is the only capture-method-dependent step.

## Swapping in real traces
`host/cw305_bnn_capture.py` writes `img####.npz` with `traces [n_kernels, n_samples]`,
`samples_per_cycle`, `n_cycles`, `image`, `label`. Replace `trace_sim` calls in
`evaluate.py` with `power_extract.load_and_extract(npz)`. Set `samples_per_cycle = CW
adc_mul`, and align cycle 0 to the trigger edge (trigger spans exactly the conv).

## Faithfulness notes
- Power model is the standard DPA Hamming-distance model on the conv datapath (products +
  accumulator). It reproduces both paper effects: background→low power, across-kernel
  feature-vector encodes pixels. Real silicon leakage is noisier but carries the same
  signal; `--noise` controls the sim SNR.
- `reconstruct()` is a faithful adaptation of Algorithm 2 (greedy, consistency/variance
  minimizing, then per-pixel average of selected candidates).
- Defaults match S7.3: 9 kernels, delta=1.0, 3 groups of 3.
- Recognition uses the BinaryNet MLP golden classifier from P2 (paper S6.3), not the
  attacked net.
```
