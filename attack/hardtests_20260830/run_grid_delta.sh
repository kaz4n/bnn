#!/usr/bin/env bash
# Per-cell delta sweep for the four-frequency transfer grid.
#
# Section 8.3 reports the grid at a single delta = 0.25. That is the tuned optimum
# for a 40-image template attacking its own condition, so it is fair between cells,
# but it is not each cell's own best value: the 10 MHz diagonal reads 0.675 there
# and 0.872 at delta = 0.10. Since Section 8.1 showed the optimum moves with the
# template, and the cross-condition cells are a different problem again, every cell
# deserves its own sweep before the surface is published.
#
# 16 cells x 6 radii = 96 attack runs. No board needed; templates and evaluation
# bundles already exist from run_all_four.sh.
set -u
PY="C:/Users/narut/ChipWhisperer/cwenv/Scripts/python.exe"
ROOT="C:/Users/narut/OneDrive/Desktop/Project/bnn"
HT="$ROOT/attack/hardtests_20260830"
cd "$ROOT/attack" || exit 1

FREQS="5m 7p5m 10m 20m"
DELTAS="0.06 0.10 0.15 0.25 0.40 0.60"

for P in $FREQS; do
  [ -f "$HT/g4_tmpl_${P}.npz" ] || { echo "missing template g4_tmpl_${P}.npz"; exit 1; }
done
for A in $FREQS; do
  [ -d "$HT/g4_bundle_${A}" ] || { echo "missing bundle g4_bundle_${A}"; exit 1; }
done

n=0
for P in $FREQS; do
  for A in $FREQS; do
    for D in $DELTAS; do
      DT=${D/./p}
      S="$HT/score_gd_${P}_vs_${A}_d${DT}.json"
      n=$((n+1))
      [ -f "$S" ] && continue
      "$PY" active_power_template_cw305.py attack \
          --template "$HT/g4_tmpl_${P}.npz" --bundle "$HT/g4_bundle_${A}" \
          --out "$HT/gd_${P}_${A}_${DT}" --feature mean_abs --delta "$D" \
          --group-size 3 >/dev/null 2>&1 || { echo "  ${P}->${A} d=${D} ATTACK FAILED"; continue; }
      "$PY" score_all.py --kind active --results "$HT/gd_${P}_${A}_${DT}" \
          --truth "$HT/g4_truth_${A}.json" --name "gd_${P}_${A}_${D}" \
          --out "$S" >/dev/null 2>&1 || { echo "  ${P}->${A} d=${D} SCORE FAILED"; continue; }
      # recovered images are regenerable from the template; keep only the score
      rm -rf "$HT/gd_${P}_${A}_${DT}"
    done
    echo "  cell ${P} -> ${A} swept"
  done
done
echo "GRID DELTA SWEEP DONE ($n cells x radii)"
