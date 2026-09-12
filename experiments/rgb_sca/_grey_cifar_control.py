"""Disambiguation control: is the negative RGB result about COLOUR, or about SCENE
COMPLEXITY? Runs the same generator on CIFAR LUMINANCE at the same bench noise.

If grayscale CIFAR also fails, the limit is scene complexity and says nothing about
colour. If grayscale CIFAR succeeds while RGB fails, colour is the thing being lost.
"""
import json, time, numpy as np, torch
from torch import nn
torch.set_num_threads(4)
from experiments.rgb_sca import rgb_forward as F, rgb_generator as G, rgb_noise as NZ
from experiments.rgb_sca.run_phase1_generator import _Args
from experiments.rgb_sca.run_phase1 import load_cifar

imgs,_ = load_cifar(20000, 32, 0)
grey = (0.299*imgs[:,0]+0.587*imgs[:,1]+0.114*imgs[:,2]).astype(np.uint8)[:,None,:,:]
g3 = np.repeat(grey, 3, axis=1)
ch = NZ.characterize_capture("host/traces_gfash_noamp_train60k", 1500)
kg = F.random_kernels(1, C=1, K=3, seed=1)
feats = NZ.simulate_features_sampled(grey, kg, "summed", ch, n_avg=1, seed=0)
print(f"grey-CIFAR features {feats.shape}", flush=True)
a = _Args(20,128,1e-3,0)
res,_ = G.run_all_arms(feats, g3, a, torch, nn)
t,p,s = res["trace"], res["prior_only"], res["summary"]
out = {"kind":"grey_cifar_disambiguation","source":"simulation_calibrated_to_hardware",
       "hardware_used":False,"trace":t,"prior_only":p,"summary":s,
       "mssim_mean":float(np.mean(t["mssim_per_channel"]))}
json.dump(out, open("experiments/rgb_sca/results/grey_cifar_control.json","w"), indent=1, default=float)
print(f"GREY-CIFAR  trace MAE {t['mae_pooled']:.2f}  prior {p['mae_pooled']:.2f}  "
      f"advantage {s['advantage_over_prior_pooled_mae']:+.2f}  MSSIM {out['mssim_mean']:.3f}")
print("  reference: fashion-MNIST anchor MSSIM 0.668 | CIFAR-RGB MSSIM 0.074")
