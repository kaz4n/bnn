#!/usr/bin/env python3
"""Capture CW-Lite traces for cw305_leakage_top.

The core holds each 3x3 valid MNIST window for DWELL clocks. With CW-Lite
extclk_x4 at 5 MHz, every window occupies 32 ADC samples and the full 26x26
window stream fits inside the Lite capture buffer.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np

LINE = 28
OUT_SIDE = 26
N_WINDOWS = OUT_SIDE * OUT_SIDE
REG_IMAGE = 0
REG_OUTPUT = 16
REG_KERNEL = 32
REG_GO = 33
REG_STATUS = 34


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def binarize_image(img, threshold):
    return (np.asarray(img, dtype=np.uint8) > threshold).astype(np.uint8)


def load_kernel_bits(path, n_kernels, probe_kernels):
    if probe_kernels == "onehot":
        if n_kernels != 9:
            raise ValueError("onehot probe mode requires --n-kernels 9")
        return np.eye(9, dtype=np.uint8)
    kernels = np.load(path)
    if kernels.ndim != 3 or kernels.shape[1:] != (3, 3):
        raise ValueError(f"expected kernels shaped (N,3,3), got {kernels.shape}")
    return (kernels[:n_kernels].reshape(n_kernels, 9) > 0).astype(np.uint8)


def pack_kernel_le(bits9):
    value = 0
    for i, bit in enumerate(np.asarray(bits9, dtype=np.uint8).reshape(9)):
        value |= (int(bit) & 1) << i
    return [value & 0xFF, (value >> 8) & 0x01]


def golden_valid_conv(img_bits, kern_bits):
    out = np.zeros(N_WINDOWS, dtype=np.int8)
    idx = 0
    for y in range(OUT_SIDE):
        for x in range(OUT_SIDE):
            patch = img_bits[y:y + 3, x:x + 3].reshape(-1)
            matches = int(np.count_nonzero(patch == kern_bits))
            out[idx] = np.int8(2 * matches - 9)
            idx += 1
    return out


def wait_idle(target, timeout_s=1.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if (target.fpga_read(REG_STATUS, 1)[0] & 1) == 0:
            return
        time.sleep(0.002)
    raise TimeoutError("FPGA did not go idle")


def run_functional_check(target, img_bits, kern_bits):
    target.fpga_write(REG_IMAGE, img_bits.reshape(-1).astype(np.uint8).tolist())
    target.fpga_write(REG_KERNEL, pack_kernel_le(kern_bits))
    target.fpga_write(REG_GO, [1])
    wait_idle(target)
    hw = np.frombuffer(bytes(target.fpga_read(REG_OUTPUT, N_WINDOWS)), dtype=np.int8)
    golden = golden_valid_conv(img_bits, kern_bits)
    if not np.array_equal(hw, golden):
        bad = int(np.nonzero(hw != golden)[0][0])
        raise RuntimeError(
            f"functional check failed at out[{bad}]: hw={int(hw[bad])} "
            f"golden={int(golden[bad])}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bitstream", default="../build/cw305_leakage_d8_l63.bit")
    ap.add_argument("--images", default="mnist_test.npz")
    ap.add_argument("--kernels", default="../training/artifacts/model_3x3/layer1_kernels.npy")
    ap.add_argument("--probe-kernels", choices=["onehot", "file"], default="onehot",
                    help="onehot gives identifiable 3x3 patches; file uses trained conv kernels")
    ap.add_argument("--out", default="traces_leakage")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--n-images", type=int, default=60)
    ap.add_argument("--n-kernels", type=int, default=9)
    ap.add_argument("--avg", type=int, default=8)
    ap.add_argument("--threshold", type=int, default=127)
    ap.add_argument("--freq", type=float, default=5e6)
    ap.add_argument("--dwell", type=int, default=8)
    ap.add_argument("--gain", type=float, default=40)
    ap.add_argument("--margin-samples", type=int, default=512)
    ap.add_argument("--presamples", type=int, default=0)
    ap.add_argument("--fpga-id", default="100t")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-program", action="store_true")
    ap.add_argument("--skip-functional-check", action="store_true")
    args = ap.parse_args()

    import chipwhisperer as cw

    d = np.load(args.images)
    imgs = d["images"]
    labels = d["labels"]
    if imgs.ndim == 4 and imgs.shape[-1] == 1:
        imgs = imgs[..., 0]
    if imgs.ndim != 3 or imgs.shape[1:] != (LINE, LINE):
        raise ValueError(f"expected images shaped (N,28,28), got {imgs.shape}")
    kernel_bits = load_kernel_bits(args.kernels, args.n_kernels, args.probe_kernels)
    packed = [pack_kernel_le(k) for k in kernel_bits]

    os.makedirs(args.out, exist_ok=True)
    spc = 4
    useful_samples = N_WINDOWS * args.dwell * spc
    nsamp = useful_samples + args.margin_samples + args.presamples
    if nsamp > 24573:
        raise ValueError(f"requested {nsamp} samples; CW-Lite max is 24573")

    scope = cw.scope()
    scope.default_setup()
    scope.clock.adc_src = "extclk_x4"
    scope.adc.samples = nsamp
    scope.adc.offset = 0
    scope.adc.presamples = args.presamples
    scope.adc.basic_mode = "rising_edge"
    scope.trigger.triggers = "tio4"
    scope.gain.db = args.gain

    bsfile = None if args.no_program else os.path.abspath(args.bitstream)
    target = cw.target(scope, cw.targets.CW305, bsfile=bsfile,
                       fpga_id=args.fpga_id, force=not args.no_program)
    target.pll.pll_enable_set(True)
    target.pll.pll_outenable_set(True, 1)
    target.pll.pll_outfreq_set(args.freq, 1)
    scope.clock.reset_adc()
    time.sleep(0.2)

    first_img = binarize_image(imgs[args.start_index], args.threshold)
    if not args.skip_functional_check:
        run_functional_check(target, first_img, kernel_bits[0])

    manifest = {
        "bitstream": os.path.abspath(args.bitstream),
        "bitstream_sha256": sha256_file(args.bitstream) if os.path.exists(args.bitstream) else None,
        "images": os.path.abspath(args.images),
        "images_sha256": sha256_file(args.images) if os.path.exists(args.images) else None,
        "kernels": os.path.abspath(args.kernels),
        "kernels_sha256": sha256_file(args.kernels) if os.path.exists(args.kernels) else None,
        "probe_kernels": args.probe_kernels,
        "trace_source": "cw_lite_analog_sync_x4",
        "design": "cw305_leakage_top",
        "line": LINE,
        "out_side": OUT_SIDE,
        "n_windows": N_WINDOWS,
        "n_kernels": args.n_kernels,
        "avg": args.avg,
        "threshold": args.threshold,
        "fpga_freq_hz": args.freq,
        "adc_src": "extclk_x4",
        "samples_per_cycle": spc,
        "dwell": args.dwell,
        "samples_per_window": args.dwell * spc,
        "samples": nsamp,
        "useful_samples": useful_samples,
        "presamples": args.presamples,
        "gain_db": args.gain,
        "status": "incomplete",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(args.out, "capture_manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=2)

    print(f"capture: images={args.n_images} kernels={args.n_kernels} avg={args.avg} "
          f"samples={nsamp} adc_locked={scope.clock.adc_locked}")

    stop = min(args.start_index + args.n_images, len(imgs))
    for n in range(args.start_index, stop):
        outp = os.path.join(args.out, f"img{n:04d}.npz")
        if os.path.exists(outp) and not args.force:
            continue
        img_bits = binarize_image(imgs[n], args.threshold)
        target.fpga_write(REG_IMAGE, img_bits.reshape(-1).astype(np.uint8).tolist())
        repeats = np.zeros((args.n_kernels, args.avg, nsamp), dtype=np.float32)
        for kid in range(args.n_kernels):
            target.fpga_write(REG_KERNEL, packed[kid])
            for rep in range(args.avg):
                scope.arm()
                target.fpga_write(REG_GO, [1])
                if scope.capture():
                    raise RuntimeError("capture timeout; check trigger, clock, and shunt path")
                repeats[kid, rep] = np.asarray(scope.get_last_trace(), dtype=np.float32)
                wait_idle(target)
        traces = repeats.mean(axis=1)
        np.savez(
            outp,
            repeats=repeats,
            traces=traces,
            image=img_bits.astype(np.uint8),
            label=int(labels[n]),
            kernel_bits=kernel_bits,
            probe_kernels=str(args.probe_kernels),
            kernel_ids=np.arange(args.n_kernels, dtype=np.int16),
            samples_per_cycle=spc,
            dwell=int(args.dwell),
            samples_per_window=int(args.dwell * spc),
            presamples=int(args.presamples),
            trace_source="cw_lite_analog_sync_x4",
        )
        print(f"captured img{n:04d} label={int(labels[n])}", flush=True)

    manifest["status"] = "complete"
    manifest["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(args.out, "capture_manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=2)
    scope.dis()
    target.dis()


if __name__ == "__main__":
    main()
