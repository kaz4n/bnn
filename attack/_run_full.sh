done_check(){ python -c "import json;import sys;d=json.load(open('results_'$1'x'$1'/summary.json'));sys.exit(0 if d['n_eval']>=200 else 1)" 2>/dev/null; }
for K in 3 5; do
  echo "### MODEL ${K}x${K} ###"
  until done_check $K; do
    python -u evaluate.py --ksize $K --n-profile 300 --n-eval 200 --n-kernels 9 --noise 0.1 --out results_${K}x${K}
  done
done
echo "### P3 FULL DONE ###"
