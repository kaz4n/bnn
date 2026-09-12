#!/usr/bin/env bash
# H6 -- probe mismatch. The template is built from captures that used one-hot
# probe kernels; the attack traces were captured while the FPGA ran the real
# trained kernels. This asks whether the attacker must know the deployed weights.
set -u
PY="C:/Users/narut/ChipWhisperer/cwenv/Scripts/python.exe"
ROOT="C:/Users/narut/OneDrive/Desktop/Project/bnn"
HT="$ROOT/attack/hardtests_20260830"
cd "$ROOT/attack" || exit 1

"$PY" active_power_template_cw305.py build-template \
    --traces "$ROOT/host/traces_rerun_20260830_onehot_80" --profile-count 40 \
    --feature mean_abs --out "$HT/mismatch_profile_onehot_p40.npz" >/dev/null || exit 1
"$PY" active_power_template_cw305.py attack \
    --template "$HT/mismatch_profile_onehot_p40.npz" \
    --bundle "$ROOT/attack/rerun_20260830/bundle_trained_80_eval40_heldout" \
    --out "$HT/mismatch_results" --feature mean_abs --delta 1.0 --group-size 3 >/dev/null || exit 1
"$PY" score_all.py --kind active --results "$HT/mismatch_results" \
    --truth "$ROOT/attack/rerun_20260830/truth_trained_80_eval40_heldout.json" \
    --name "probe_mismatch_onehot_template_vs_trained_traces" \
    --out "$HT/score_probe_mismatch.json" >/dev/null || exit 1
echo "MISMATCH DONE"
