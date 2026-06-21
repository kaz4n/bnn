# P2 — BNN training & weight export

Reproduces the binarized networks of Wei et al. "I Know What You See" (ACSAC'18).
See `../SETUP_PLAN.md` for the full project plan.

## What this produces

| Script | Network | Role | Paper acc |
|---|---|---|---|
| `train_attacked_cnn.py --ksize 3` | 4-layer BNN, layer-1 = 64× **3×3** | **Model 1** (attacked) | 99.42% |
| `train_attacked_cnn.py --ksize 5` | 4-layer BNN, layer-1 = 64× **5×5** | **Model 2** (attacked) | 99.27% |
| `train_golden_mlp.py` | BinaryNet MLP (3×4096) | **golden classifier** for recognition-accuracy metric | 99.2% |

`bnn_layers.py` — self-contained BinaryNet binarization (STE sign, weight clip).
larq is **not** used (it breaks on Keras 3).

## Run

```bash
pip install -r requirements.txt
python train_attacked_cnn.py --ksize 3 --epochs 40
python train_attacked_cnn.py --ksize 5 --epochs 40
python train_golden_mlp.py --epochs 50
```

## Key outputs (per model, under `artifacts/model_KxK/`)

- `layer1_kernels.npy` — **(64, K, K) int8 in {−1,+1}** — the ground-truth first-layer
  kernels. This is what the FPGA (P1) must load and what the side-channel attack (P3)
  profiles. The whole experiment hinges on these 64 kernels.
- `layer1_kernels_packed.bin` — bit-packed for FPGA BRAM (1 bit/weight, row-major,
  MSB-first, `1`=+1 / `0`=−1). 3×3 → 2 bytes/kernel (128 B total); 5×5 → 4 bytes (256 B).
- `layer1_kernels.txt` — human-readable.
- `layer1_bn.npz` — layer-1 BatchNorm (gamma, beta, mean, var, epsilon). Needed only if
  P1 runs the full classifier in HW; the attack itself doesn't use it.
- `weights.weights.h5`, `metrics.json`.

## Design notes (faithfulness)

- **First conv layer takes REAL pixels, binary kernels** — exactly the line-buffer op the
  paper attacks (paper feeds pixels 0..255). Input scale used in training does not affect
  the binary kernels or the attack.
- `padding='same'` keeps the feature-map width at 28 → matches the paper's "line size 28".
- 4 weight layers (2 conv + 2 dense) with layer-1 = 64 kernels matches Table 1. Layers 2–4
  are unspecified in the paper; this architecture reaches the reported accuracy. They do
  not affect the attack (which only uses layer-1 power), but are trained so the model is a
  real, accurate classifier as in the paper.
- Both Model 1 (3×3) and Model 2 (5×5) are produced — full faithful reproduction.

## Bit-ordering ⚠️

`layer1_kernels_packed.bin` ordering (row-major, MSB-first) is a **convention**. P1 must
load kernels into BRAM in the SAME order the conv RTL expects. Verify with a bit-exact
C-sim-vs-hardware check (SETUP_PLAN.md Phase 1).
