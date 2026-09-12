# Resuming the amplifier sweep (needs someone at the board)

Everything except the captures is done. The four bitstreams exist and are verified.

## Why it must be supervised

This CW305 does not reprogram over an already-configured FPGA and reports success
when it refuses. The four grey bitstreams have identical register maps, so neither
the stock-AES page signature nor the byte-ramp probe distinguishes them. The only
discriminator is mean absolute trace amplitude. The 2026-09-06 attempt produced four
captures of the same bitstream and would have looked like a clean result.

## Procedure, per LEAK_LANES value in 63 15 3 0

1. Cut power to the CW305. USB RST is not enough.
2. Program while blank:
   `cw.target(None, cw.targets.CW305, bsfile="build/cw305_leakage_grey_d8_l<L>.bit",
    fpga_id="100t", force=True, slurp=False)`
3. Capture ~20 images and record mean abs trace.
4. It must differ clearly from the previous LANES value. If it matches, the reprogram
   did not take: power cycle and retry rather than continuing.
5. Only then run the full capture:
   `bash attack/hardtests_20260830/run_amp_sweep.sh <L>`

Expect amplitude to fall with LANES. At LANES=63 it is about 7500 (mean abs).

## Then, board-free

The P2P step in run_amp_sweep.sh already pins `--split random --seed 0` with
1500/200/300, so the train/val/test indices are identical at every point; verify
`train_indices` match across the four `summary.json` files before comparing metrics.
Report against the Fashion chance baselines (f1_chance 0.474), not the MNIST grey row.

## What to delete first

`host/traces_gfash_l{63,15,3,0}` and `attack/hardtests_20260830/p2p_gfash_l*` are the
void captures and their results. Remove them before rerunning so they cannot be mixed
with good data. See AMP_SWEEP_VOID.md.
