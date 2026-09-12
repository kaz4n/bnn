#!/usr/bin/env bash
# Dataset grid, second pass. Fixes two defects in the first run.
#
# 1. The recogniser. score_all.py builds GoldenMLP() with no argument, which
#    loads the MNIST digit weights. The first pass therefore scored recovered
#    clothing with a handwritten-digit classifier and reported 0.025 recognition
#    on the ground-truth Fashion images, where the Fashion recogniser gets 0.842.
#    The weights now follow the ATTACKED dataset, which is the one the true
#    images come from.
#
# 2. The radius floor. fashion->fashion peaked at delta = 0.02, the bottom of the
#    swept range, so its optimum was not bracketed and could not be quoted. The
#    range is extended downward. This is the third time in this project that an
#    optimum has landed on the edge of a sweep, so the aggregator checks for it
#    explicitly rather than relying on me noticing.
set -u
PY="C:/Users/narut/ChipWhisperer/cwenv/Scripts/python.exe"
ROOT="C:/Users/narut/OneDrive/Desktop/Project/bnn"
HT="$ROOT/attack/hardtests_20260830"
cd "$ROOT/attack" || exit 1

MNIST_W="$ROOT/training/artifacts/golden_mlp/weights.weights.h5"
FASHION_W="$ROOT/training/artifacts/golden_mlp_fashion/weights.weights.h5"
[ -f "$FASHION_W" ] || { echo "missing Fashion recogniser weights"; exit 1; }

DELTAS="0.005 0.01 0.015 0.02 0.03 0.04 0.06 0.10 0.15 0.25 0.40 0.60"

for P in mnist fashion; do
  for A in mnist fashion; do
    if [ "$A" = "fashion" ]; then W="$FASHION_W"; else W="$MNIST_W"; fi
    for D in $DELTAS; do
      DT=${D/./p}
      S="$HT/score_ds2_${P}_vs_${A}_d${DT}.json"
      [ -f "$S" ] && continue
      "$PY" active_power_template_cw305.py attack \
          --template "$HT/ds_tmpl_${P}.npz" --bundle "$HT/ds_bundle_${A}" \
          --out "$HT/ds2_${P}_${A}_${DT}" --feature mean_abs --delta "$D" \
          --group-size 3 >/dev/null 2>&1 || { echo "  ${P}->${A} d=${D} ATTACK FAILED"; continue; }
      BNN_RECOGNIZER_WEIGHTS="$W" "$PY" score_all.py --kind active \
          --results "$HT/ds2_${P}_${A}_${DT}" --truth "$HT/ds_truth_${A}.json" \
          --name "ds2_${P}_${A}_${D}" --out "$S" >/dev/null 2>&1 \
          || { echo "  ${P}->${A} d=${D} SCORE FAILED"; continue; }
      rm -rf "$HT/ds2_${P}_${A}_${DT}"
    done
    echo "  cell ${P} -> ${A} swept"
  done
done
echo "DATASET GRID 2 DONE"
