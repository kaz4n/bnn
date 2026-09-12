#!/usr/bin/env python3
"""Capture the same images under several conditions, interleaved.

Why this exists
---------------
Glamocanin et al. (DATE 2023) show that when power traces for different classes
are acquired one class after another, a model can learn the slow thermal drift
between the blocks instead of the leakage, giving "misleading and overly
optimistic" results. Their remedy is to interleave the recordings so the drift is
spread evenly over every class. They also note the effect is largest exactly when
the leakage is weak -- which is this project's case, since the convolution unit is
tiny next to the rest of the fabric.

My clock-transfer captures were consecutive: all 40 images at 5 MHz, then all 40
at 10 MHz. That is the acquisition pattern their paper warns about, so part of
the measured cross-clock gap could be thermal rather than electrical. This script
re-captures the same images with the conditions interleaved per image, so the two
runs can be compared and the thermal explanation ruled in or out.

It reuses the helpers of cw305_leakage_capture.py rather than copying them, so
the authenticity gate, the functional check and the file format stay identical.
"""
import argparse
import importlib.util
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_capture_module():
    path = os.path.join(HERE, "cw305_leakage_capture.py")
    spec = importlib.util.spec_from_file_location("cw305_leakage_capture", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cap = _load_capture_module()


def set_clock(scope, freq):
    """Retune the capture clock and wait for the ADC to relock."""
    scope.clock.adc_src = "clkgen_x4"
    scope.clock.clkgen_freq = freq
    time.sleep(0.15)
    scope.clock.reset_adc()
    deadline = time.time() + 3.0
    while time.time() < deadline:
        if scope.clock.adc_locked:
            return
        time.sleep(0.05)
    raise RuntimeError(f"ADC did not relock after switching to {freq:g} Hz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freqs", nargs="+", type=float, required=True,
                    help="conditions to interleave, e.g. --freqs 5e6 10e6")
    ap.add_argument("--out-prefix", required=True,
                    help="one directory per frequency is written: <prefix>_<MHz>m")
    ap.add_argument("--images", default="mnist_test.npz")
    ap.add_argument("--kernels",
                    default="../training/artifacts/model_3x3/layer1_kernels.npy")
    ap.add_argument("--probe-kernels", choices=["onehot", "file"], default="file")
    ap.add_argument("--start-index", type=int, default=5300)
    ap.add_argument("--n-images", type=int, default=40)
    ap.add_argument("--n-kernels", type=int, default=9)
    ap.add_argument("--avg", type=int, default=5)
    ap.add_argument("--threshold", type=int, default=127)
    ap.add_argument("--gain", type=float, default=40)
    ap.add_argument("--dwell", type=int, default=8)
    ap.add_argument("--grey", action="store_true")
    ap.add_argument("--fpga-id", default="100t")
    ap.add_argument("--force", action="store_true",
                    help="overwrite images that were already captured")
    args = ap.parse_args()

    import chipwhisperer as cw

    imgs = np.load(args.images)["images"]
    kernel_bits = cap.load_kernel_bits(args.kernels, args.n_kernels, args.probe_kernels)
    packed = [cap.pack_kernel_le(k) for k in kernel_bits]

    scope = cw.scope()
    scope.default_setup()
    scope.gain.db = args.gain
    target = cw.target(None, cw.targets.CW305, bsfile=None,
                       fpga_id=args.fpga_id, force=False)

    design_check = cap.verify_custom_design_loaded(target)

    spc = 4
    spw = args.dwell * spc
    n_windows = cap.N_WINDOWS
    nsamp = None

    dirs = {}
    for f in args.freqs:
        tag = f"{f / 1e6:g}".replace(".", "p")
        d = f"{args.out_prefix}_{tag}m"
        os.makedirs(d, exist_ok=True)
        dirs[f] = d

    # One functional check per condition, before any trace is kept.
    func = {}
    first_img = cap.prepare_image(imgs[args.start_index], args.threshold, args.grey)
    for f in args.freqs:
        set_clock(scope, f)
        func[f] = cap.run_functional_check(target, first_img, kernel_bits[0],
                                           grey=args.grey)
        print(f"functional check @ {f/1e6:g} MHz: "
              f"{'PASS' if func[f]['passed'] else 'FAIL'} "
              f"({func[f]['mismatches']} mismatches)")

    stop = min(args.start_index + args.n_images, len(imgs))
    order = np.arange(args.start_index, stop, dtype=np.int32)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    scope.adc.samples = 0  # set on first arm below
    print(f"interleaved capture: images={len(order)} conditions={len(args.freqs)} "
          f"avg={args.avg} kernels={args.n_kernels}")

    for pos, n in enumerate(order):
        n = int(n)
        img_bits = cap.prepare_image(imgs[n], args.threshold, args.grey)
        # The inner loop is over CONDITIONS, not images: every image is captured
        # at every frequency back to back, so the slow thermal ramp of the whole
        # session lands on both conditions in the same proportion.
        for f in args.freqs:
            outp = os.path.join(dirs[f], f"img{n:04d}.npz")
            if os.path.exists(outp) and not args.force:
                continue
            set_clock(scope, f)
            if nsamp is None:
                nsamp = n_windows * spw
                scope.adc.samples = nsamp + 512
            cap.fpga_write_region(target, cap.REG_IMAGE,
                                  img_bits.reshape(-1).astype(np.uint8).tolist())
            repeats = np.zeros((args.n_kernels, args.avg, scope.adc.samples),
                               dtype=np.float32)
            for kid in range(args.n_kernels):
                cap.fpga_write_region(target, cap.REG_KERNEL, packed[kid])
                for r in range(args.avg):
                    scope.arm()
                    cap.fpga_write_region(target, cap.REG_GO, [1])
                    if scope.capture():
                        raise RuntimeError("capture timeout")
                    repeats[kid, r] = np.asarray(scope.get_last_trace(),
                                                 dtype=np.float32)
                    cap.wait_idle(target)
            np.savez_compressed(
                outp,
                traces=repeats.mean(axis=1).astype(np.float32),
                repeats=repeats,
                image=img_bits.astype(np.uint8),
                label=np.int64(np.load(args.images)["labels"][n]),
                kernel_bits=kernel_bits.astype(np.uint8),
                dwell=np.int32(args.dwell),
                samples_per_cycle=np.int32(spc),
                samples_per_window=np.int32(spw),
                presamples=np.int32(0),
            )
        if (pos + 1) % 10 == 0:
            print(f"  {pos + 1}/{len(order)} images")

    for f in args.freqs:
        manifest = {
            "kind": "cw305_leakage_capture",
            "acquisition": "interleaved",
            "acquisition_note": (
                "every image captured at every condition back to back, so thermal "
                "drift is spread evenly across conditions (Glamocanin et al., "
                "DATE 2023)"),
            "interleaved_with": [x for x in args.freqs if x != f],
            "design": "cw305_leakage_grey_top" if args.grey else "cw305_leakage_top",
            "input_mode": "grey" if args.grey else "binary",
            "fpga_freq_hz": f,
            "gain_db": args.gain,
            "avg": args.avg,
            "n_kernels": args.n_kernels,
            "dwell": args.dwell,
            "out_side": cap.OUT_SIDE,
            "n_windows": n_windows,
            "samples_per_cycle": spc,
            "samples_per_window": spw,
            "presamples": 0,
            "threshold": args.threshold,
            "probe_kernels": args.probe_kernels,
            "start_index": args.start_index,
            "trace_source": "cw_lite_analog_sync_x4",
            "design_check": design_check,
            "functional_check": func[f],
            "started_utc": started,
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "complete",
        }
        with open(os.path.join(dirs[f], "capture_manifest.json"), "w",
                  encoding="utf-8") as fp:
            json.dump(manifest, fp, indent=2)
        print("wrote", dirs[f])

    target.dis()
    scope.dis()


if __name__ == "__main__":
    main()
