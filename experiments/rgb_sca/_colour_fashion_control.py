"""The experiment that actually isolates COLOUR recovery.

CIFAR failed in grayscale as badly as in RGB (MSSIM 0.077 vs 0.074, both ~zero advantage
over the prior), so that negative was about scene complexity and said nothing about
colour. The validated anchor -- Fashion-MNIST, same kernel, same bench noise -- reaches
MSSIM 0.668, so spatial content at THAT difficulty is recoverable on this bench.

So: take the exact anchor images and add colour, changing nothing else.

    rgb[c] = fashion_grey * tint[c],   tint ~ U(0.2, 1.0)^3, drawn per image

Two properties make this the right test:

1. Spatial difficulty is identical to the validated anchor, so a failure cannot be blamed
   on scene complexity -- the one confound that invalidated the CIFAR result.
2. The tint is RANDOM per image, so chrominance is genuinely unpredictable from structure,
   from luminance, or from the dataset prior. A model cannot score on chroma by
   colourising plausibly; the only way to get the tint right is to read it from the trace.

That second property is what a reviewer would demand. On natural images a generator can
infer colour from semantics (sky is blue, foliage green) and appear to recover colour
while carrying no colour information at all. Random tints remove that escape route.

Read the outcome as:
  chroma advantage ~ 0, luma advantage > 0  -> grayscale recovery from an RGB-input
                                               accelerator. A real finding, NOT RGB
                                               reconstruction.
  chroma advantage > 0 and swap penalty > 0 -> colour genuinely recovered from the side
                                               channel. That is the novel claim.
"""
import json, numpy as np, torch
from torch import nn
torch.set_num_threads(4)
from experiments.rgb_sca import rgb_forward as F, rgb_generator as G, rgb_noise as NZ
from experiments.rgb_sca.run_phase1_generator import _Args

N = 20000
fash = np.load("host/fashion_train.npz")["images"][:N].astype(np.float64)
rng = np.random.default_rng(0)
tint = rng.uniform(0.2, 1.0, size=(N, 3))
rgb = np.clip(fash[:, None, :, :] * tint[:, :, None, None], 0, 255).astype(np.uint8)
print(f"colour-fashion {rgb.shape}, random tint per image", flush=True)

ch = NZ.characterize_capture("host/traces_gfash_noamp_train60k", 1500)
kern = F.random_kernels(1, C=3, K=3, seed=1)
# Valid geometry matching the residual pool, as the anchor does.
pw = np.stack([F.per_cycle_power(im, kern[0], "summed") for im in rgb])
pw = pw[:, :ch["n_windows_pool"]]
feats = NZ.simulate_features_sampled(rgb, kern, "summed", ch, n_avg=1, seed=0, power=pw)
print(f"features {feats.shape}", flush=True)

res, recs = G.run_all_arms(feats, rgb, _Args(20, 128, 1e-3, 0), torch, nn)
t, p, s = res["trace"], res["prior_only"], res["summary"]
cp = t["channel_permutation"]
saved = G.save_reconstructions(recs, "experiments/rgb_sca/results/colour_fashion/images",
                               "summed", n_show=12, seed=0)
out = {"kind": "colour_fashion_isolation", "source": "simulation_calibrated_to_hardware",
       "hardware_used": False, "trace": t, "prior_only": p, "summary": s,
       "images": saved, "mssim_mean": float(np.mean(t["mssim_per_channel"]))}
json.dump(out, open("experiments/rgb_sca/results/colour_fashion/results.json", "w"),
          indent=1, default=float)

print(f"\nCOLOUR-FASHION  (anchor grayscale reference: MSSIM 0.668)")
print(f"  trace MAE/ch {[round(v,2) for v in t['mae_per_channel']]}  pooled {t['mae_pooled']:.2f}")
print(f"  prior MAE/ch {[round(v,2) for v in p['mae_per_channel']]}  pooled {p['mae_pooled']:.2f}")
print(f"  shuffled {res['shuffled']['mae_pooled']:.2f}  matches prior: {s['shuffled_matches_prior']}")
print(f"  ADVANTAGE pooled {s['advantage_over_prior_pooled_mae']:+.2f}")
print(f"  LUMA adv {s['luma_advantage']:+.2f}   CHROMA adv {[round(v,2) for v in s['chroma_advantage']]}")
print(f"  swap penalty {cp['swap_penalty']:+.3f} (ratio {cp['swap_penalty_ratio']:.4f})")
print(f"  MSSIM/ch {[round(v,3) for v in t['mssim_per_channel']]}")
print(f"  images: {saved['grid']}")
