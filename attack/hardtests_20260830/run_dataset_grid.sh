#!/usr/bin/env bash
# The 2x2 dataset transfer grid: {MNIST, Fashion} template x {MNIST, Fashion} traces.
#
# Wei et al. claim the power template is dataset-independent, because it maps a
# power vector to the pixels inside a convolution window rather than to anything
# semantic. Their evidence is a single mammography image with no metrics. This
# grid puts a number on it.
#
# The two off-diagonal cells are the claim; the two diagonals are what make them
# readable. MNIST->Fashion on its own cannot distinguish "the template did not
# transfer" from "Fashion is a harder image set", and Fashion is measurably
# harder: 31.5 % foreground against 13.4 %, all 512 of the 3x3 patch codes
# occurring against 434, and patch entropy 3.68 bits against 2.69.
#
# Radii are swept and the sweep is deliberately wide. Two conclusions in this
# project were previously drawn from an optimum sitting on the edge of the swept
# range (Sections 8.3 and 8.5), so the grid is scored over nine values and the
# aggregator refuses to report a cell whose best value is not bracketed.
set -u
PY="C:/Users/narut/ChipWhisperer/cwenv/Scripts/python.exe"
ROOT="C:/Users/narut/OneDrive/Desktop/Project/bnn"
H="$ROOT/host"
HT="$ROOT/attack/hardtests_20260830"
cd "$ROOT/attack" || exit 1

DELTAS="0.02 0.03 0.04 0.06 0.10 0.15 0.25 0.40 0.60"

# ---------- Fashion template and bundle (MNIST ones already exist) -----------
[ -d "$H/traces_fash_p_10m" ] || { echo "missing traces_fash_p_10m -- run run_fashion_capture.sh"; exit 1; }
[ -d "$H/traces_fash_e_10m" ] || { echo "missing traces_fash_e_10m -- run run_fashion_capture.sh"; exit 1; }

T="$HT/ds_tmpl_fashion.npz"
[ -f "$T" ] || "$PY" active_power_template_cw305.py build-template \
    --traces "$H/traces_fash_p_10m" --profile-count 40 --feature mean_abs \
    --out "$T" || exit 1

B="$HT/ds_bundle_fashion"
[ -d "$B" ] || "$PY" active_power_template_cw305.py make-bundle \
    --traces "$H/traces_fash_e_10m" --start-index 0 --count 40 \
    --out "$B" --truth-out "$HT/ds_truth_fashion.json" || exit 1
echo "fashion template and bundle ready"

# MNIST side reuses the existing 10 MHz cell unchanged, so the only variable
# between the two rows is the dataset.
ln -sfn "$HT/g4_tmpl_10m.npz" "$HT/ds_tmpl_mnist.npz" 2>/dev/null || \
    cp -f "$HT/g4_tmpl_10m.npz" "$HT/ds_tmpl_mnist.npz"
[ -d "$HT/ds_bundle_mnist" ] || cp -r "$HT/g4_bundle_10m" "$HT/ds_bundle_mnist"
[ -f "$HT/ds_truth_mnist.json" ] || cp -f "$HT/g4_truth_10m.json" "$HT/ds_truth_mnist.json"

# ---------- 2x2 x 9 radii = 36 attack runs -----------------------------------
for P in mnist fashion; do
  for A in mnist fashion; do
    for D in $DELTAS; do
      DT=${D/./p}
      S="$HT/score_ds_${P}_vs_${A}_d${DT}.json"
      [ -f "$S" ] && continue
      "$PY" active_power_template_cw305.py attack \
          --template "$HT/ds_tmpl_${P}.npz" --bundle "$HT/ds_bundle_${A}" \
          --out "$HT/ds_${P}_${A}_${DT}" --feature mean_abs --delta "$D" \
          --group-size 3 >/dev/null 2>&1 || { echo "  ${P}->${A} d=${D} ATTACK FAILED"; continue; }
      "$PY" score_all.py --kind active --results "$HT/ds_${P}_${A}_${DT}" \
          --truth "$HT/ds_truth_${A}.json" --name "ds_${P}_${A}_${D}" \
          --out "$S" >/dev/null 2>&1 || { echo "  ${P}->${A} d=${D} SCORE FAILED"; continue; }
      rm -rf "$HT/ds_${P}_${A}_${DT}"
    done
    echo "  cell ${P} -> ${A} swept"
  done
done
echo "DATASET GRID DONE"
