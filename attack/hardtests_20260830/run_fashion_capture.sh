#!/usr/bin/env bash
# Fashion-MNIST capture, matched cell-for-cell against the existing 10 MHz MNIST cell.
#
# Wei et al. state that the recovery template is "independent from the chosen
# dataset if the template is built in a way avoids overfitting", but the only
# evidence is one mammography image shown qualitatively with no metrics, and
# they list harder datasets as future work. Power2Picture did evaluate
# Fashion-MNIST, but only for the generative attack, where it degraded sharply
# (MSSIM 0.62 -> 0.38, pixel distance 32 -> 77, recognition 96.4 % -> 53.0 %).
# Nobody has run a template attack on Fashion-MNIST.
#
# Testing that claim needs a 2x2 dataset grid, not a single cross run: a
# MNIST-template-on-Fashion-traces result alone confounds "the template failed
# to transfer" with "Fashion is simply harder". The Fashion->Fashion diagonal is
# the control that separates them, exactly as the matched diagonal did for the
# clock grid in Section 8.3.
#
# Every setting below is copied from traces_g4p_10m/capture_manifest.json so the
# only variable against the existing MNIST cell is the dataset.
#
# Note the probe kernels stay the MNIST-trained layer-1 kernels. That is
# deliberate: it holds the victim accelerator completely fixed so the only
# thing changing is the distribution of the input images, which is what the
# dataset-independence claim is actually about. Swapping in Fashion-trained
# kernels as well would change the accelerator and the inputs at once, and is
# a separate experiment.
set -eu
PY="C:/Users/narut/ChipWhisperer/cwenv/Scripts/python.exe"
ROOT="C:/Users/narut/OneDrive/Desktop/Project/bnn"
cd "$ROOT/host"

COMMON="--bitstream ../build/cw305_leakage_d8_l63.bit \
        --images fashion_test.npz \
        --probe-kernels file \
        --n-kernels 9 --avg 5 --dwell 8 --threshold 127 \
        --freq 10e6 --gain 40 --shuffle-seed 8302028"

# profiling set: images 5300-5339, same indices the MNIST template used
"$PY" cw305_leakage_capture.py $COMMON \
    --out traces_fash_p_10m --start-index 5300 --n-images 40

# evaluation set: images 5200-5239, disjoint from the profiling set
"$PY" cw305_leakage_capture.py $COMMON --no-program \
    --out traces_fash_e_10m --start-index 5200 --n-images 40

echo "FASHION CAPTURE DONE"
