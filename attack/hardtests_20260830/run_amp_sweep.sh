#!/usr/bin/env bash
# How much of this attack is the accelerator, and how much is the amplifier I added?
#
# Every grey result so far comes from a design carrying LEAK_LANES=63, i.e. a
# 72x63 = 4536-flop register bank whose only purpose is to toggle on the MAC term
# and make the core leak. No accelerator in either paper has such a circuit. So
# the published numbers describe an attack on hardware built to be attacked.
#
# This sweeps the bank down to nothing:
#   LANES=63 -> 4536 flops   (what every previous grey result used)
#   LANES=15 -> 1080
#   LANES=3  ->  216
#   LANES=0  ->    0         no bank at all; only the real convolution switches
#
# LANES=0 is the honest baseline and the number that should be quoted against the
# papers. The earlier finding that the attack has "no signal without the
# amplifier" was never measured on the grey design, so this also tests it.
#
# VERIFICATION MATTERS HERE. All four bitstreams are grey designs with identical
# register maps, so neither the AES page signature nor the byte-ramp probe can
# tell them apart, and reprogramming a configured FPGA silently fails on this
# board. The discriminator is the mean absolute trace amplitude, which must fall
# monotonically as LANES falls. Two captures with equal amplitude mean a
# reprogram did not take and the same design was measured twice.
set -u
PY="C:/Users/narut/ChipWhisperer/cwenv/Scripts/python.exe"
ROOT="C:/Users/narut/OneDrive/Desktop/Project/bnn"
HT="$ROOT/attack/hardtests_20260830"
FASHION_W="$ROOT/training/artifacts/golden_mlp_fashion/weights.weights.h5"

for L in "$@"; do
  BS="$ROOT/build/cw305_leakage_grey_d8_l${L}.bit"
  OUT="$ROOT/host/traces_gfash_l${L}"
  [ -f "$BS" ] || { echo "missing bitstream for LANES=$L"; continue; }

  if [ ! -d "$OUT" ]; then
    cd "$ROOT/host" || exit 1
    "$PY" cw305_p2p_capture.py --bitstream "$BS" \
        --images fashion_test.npz --grey --kidx 0 \
        --out "traces_gfash_l${L}" \
        --start-index 2000 --n-images 2000 --shard-size 500 \
        --avg 1 --dwell 8 --freq 5e6 --gain 40 --shuffle-seed 8302028 \
        > "/tmp/cap_l${L}.log" 2>&1 \
      || { echo "LANES=$L CAPTURE FAILED (see /tmp/cap_l${L}.log)"; continue; }
  fi

  # identical split at every point, so a quality change is not a different test set
  cd "$ROOT/attack" || exit 1
  R="$HT/p2p_gfash_l${L}"
  [ -d "$R" ] || BNN_RECOGNIZER_WEIGHTS="$FASHION_W" "$PY" p2p_generator.py \
      --traces "$OUT" --out "$R" \
      --features clock_abs --loss mse --split random \
      --epochs 60 --batch 128 --lr 1e-3 \
      --val-count 200 --test-count 300 --seed 0 --require-hardware \
      > "/tmp/p2p_l${L}.log" 2>&1 \
    || { echo "LANES=$L P2P FAILED"; continue; }
  echo "  LANES=$L done"
done
echo "AMP SWEEP DONE"
