#!/usr/bin/env bash
# H2..H4 -- condition transfer. Every condition captures the SAME 40 images
# (5200..5239); only the hardware condition changes. All are attacked with the
# ONE profile built hours earlier from img0780..0819 at 5 MHz / 40 dB / avg 5.
set -u
PY="C:/Users/narut/ChipWhisperer/cwenv/Scripts/python.exe"
ROOT="C:/Users/narut/OneDrive/Desktop/Project/bnn"
HT="$ROOT/attack/hardtests_20260830"
PROFILE="$ROOT/attack/rerun_20260830/profile_trained_80_p40.npz"
cd "$ROOT/attack" || exit 1

for C in C1_control C2_reprogram C3_clk7m5 C4_clk10m C5_gain30 C6_gain50; do
  echo "=============== $C ==============="
  TR="$ROOT/host/traces_hard_${C}"
  "$PY" active_power_template_cw305.py make-bundle \
      --traces "$TR" --start-index 0 --count 40 \
      --out "$HT/cond_bundle_${C}" --truth-out "$HT/cond_truth_${C}.json" >/dev/null || exit 1
  "$PY" active_power_template_cw305.py attack \
      --template "$PROFILE" --bundle "$HT/cond_bundle_${C}" \
      --out "$HT/cond_results_${C}" --feature mean_abs --delta 1.0 --group-size 3 >/dev/null || exit 1
  "$PY" score_all.py --kind active --results "$HT/cond_results_${C}" \
      --truth "$HT/cond_truth_${C}.json" --name "cond_${C}" \
      --out "$HT/score_cond_${C}.json" >/dev/null || exit 1
  echo "$C done"
done
echo "CONDITIONS DONE"
