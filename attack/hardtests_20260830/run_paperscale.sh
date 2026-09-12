#!/usr/bin/env bash
# H5 -- paper-scale run: 300 profile images / 200 evaluation images, the split
# Wei et al. use. Same board, same 9 probe kernels, same delta and grouping.
set -u
PY="C:/Users/narut/ChipWhisperer/cwenv/Scripts/python.exe"
ROOT="C:/Users/narut/OneDrive/Desktop/Project/bnn"
HT="$ROOT/attack/hardtests_20260830"
TR="$ROOT/host/traces_paperscale_500"
cd "$ROOT/attack" || exit 1

"$PY" active_power_template_cw305.py build-template \
    --traces "$TR" --profile-count 300 --feature mean_abs \
    --out "$HT/paper_profile_p300.npz" || exit 1
echo "template built"
"$PY" active_power_template_cw305.py make-bundle \
    --traces "$TR" --start-index 6300 --count 200 \
    --out "$HT/paper_bundle_e200" --truth-out "$HT/paper_truth_e200.json" || exit 1
echo "bundle built"
"$PY" active_power_template_cw305.py attack \
    --template "$HT/paper_profile_p300.npz" --bundle "$HT/paper_bundle_e200" \
    --out "$HT/paper_results_e200" --feature mean_abs --delta 1.0 --group-size 3 || exit 1
echo "attack done"
"$PY" score_all.py --kind active --results "$HT/paper_results_e200" \
    --truth "$HT/paper_truth_e200.json" --name "paperscale_p300_e200" \
    --out "$HT/score_paperscale.json" || exit 1
echo "PAPERSCALE DONE"
