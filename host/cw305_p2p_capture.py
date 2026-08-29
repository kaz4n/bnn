#!/usr/bin/env python3
"""Bulk single-shot capture for the Power2Picture-style generative attack.

Power2Picture (Huegle et al., FCCM 2023) trains a generative CNN on power
traces, so it needs many distinct input images and no probe kernels: one
trace per image, captured while the accelerator runs its real trained kernel.

This script therefore captures 1 kernel x 1 repeat per image and stores traces
as int16 shards, which keeps thousands of images on disk at ~44 kB each.
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
TRACE_SCALE = 32767.0


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def binarize_image(img, threshold):
    return (np.asarray(img, dtype=np.uint8) > threshold).astype(np.uint8)


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
            out[idx] = np.int8(2 * int(np.count_nonzero(patch == kern_bits)) - 9)
            idx += 1
    return out


def wait_idle(target, timeout_s=1.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if (target.fpga_read(REG_STATUS, 1)[0] & 1) == 0:
            return
        time.sleep(0.001)
    raise TimeoutError("FPGA did not go idle")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bitstream", default="../build/cw305_leakage_d8_l63.bit")
    ap.add_argument("--images", default="mnist_test.npz")
    ap.add_argument("--kernels", default="../training/artifacts/model_3x3/layer1_kernels.npy")
    ap.add_argument("--kidx", type=int, default=0, help="index of the trained kernel to run")
    ap.add_argument("--out", default="traces_p2p")
    ap.add_argument("--start-index", type=int, default=1000)
    ap.add_argument("--n-images", type=int, default=8500)
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--avg", type=int, default=1)
    ap.add_argument("--threshold", type=int, default=127)
    ap.add_argument("--freq", type=float, default=5e6)
    ap.add_argument("--dwell", type=int, default=8)
    ap.add_argument("--gain", type=float, default=40)
    ap.add_argument("--margin-samples", type=int, default=512)
    ap.add_argument("--fpga-id", default="100t")
    ap.add_argument("--no-program", action="store_true")
    args = ap.parse_args()

    import chipwhisperer as cw

    d = np.load(args.images)
    imgs = d["images"]
    labels = d["labels"]
    if imgs.ndim == 4 and imgs.shape[-1] == 1:
        imgs = imgs[..., 0]
    kern = np.load(args.kernels)
    kernel_bits = (kern[args.kidx].reshape(9) > 0).astype(np.uint8)

    os.makedirs(args.out, exist_ok=True)
    spc = 4
    useful = N_WINDOWS * args.dwell * spc
    nsamp = useful + args.margin_samples
    if nsamp > 24573:
        raise ValueError(f"requested {nsamp} samples; CW-Lite max is 24573")

    scope = cw.scope()
    scope.default_setup()
    scope.clock.adc_src = "extclk_x4"
    scope.adc.samples = nsamp
    scope.adc.offset = 0
    scope.adc.presamples = 0
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
    target.fpga_write(REG_KERNEL, pack_kernel_le(kernel_bits))

    # functional check on the first image so the shards are provably silicon data
    first = binarize_image(imgs[args.start_index], args.threshold)
    target.fpga_write(REG_IMAGE, first.reshape(-1).astype(np.uint8).tolist())
    target.fpga_write(REG_GO, [1])
    wait_idle(target)
    hw = np.frombuffer(bytes(target.fpga_read(REG_OUTPUT, N_WINDOWS)), dtype=np.int8)
    if not np.array_equal(hw, golden_valid_conv(first, kernel_bits)):
        raise RuntimeError("functional check failed before bulk capture")
    print("PASS: 676 outputs bit-exact", flush=True)

    manifest = {
        "bitstream": os.path.abspath(args.bitstream),
        "bitstream_sha256": sha256_file(args.bitstream),
        "images": os.path.abspath(args.images),
        "images_sha256": sha256_file(args.images),
        "kernels": os.path.abspath(args.kernels),
        "kernel_index": args.kidx,
        "kernel_bits": kernel_bits.tolist(),
        "probe_kernels": "file",
        "trace_source": "cw_lite_analog_sync_x4",
        "design": "cw305_leakage_top",
        "purpose": "power2picture_generative",
        "line": LINE,
        "out_side": OUT_SIDE,
        "n_windows": N_WINDOWS,
        "n_kernels": 1,
        "avg": args.avg,
        "threshold": args.threshold,
        "fpga_freq_hz": args.freq,
        "adc_src": "extclk_x4",
        "samples_per_cycle": spc,
        "dwell": args.dwell,
        "samples_per_window": args.dwell * spc,
        "samples": nsamp,
        "useful_samples": useful,
        "gain_db": args.gain,
        "trace_dtype": "int16",
        "trace_scale": TRACE_SCALE,
        "start_index": args.start_index,
        "n_images": args.n_images,
        "shard_size": args.shard_size,
        "status": "incomplete",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    mpath = os.path.join(args.out, "capture_manifest.json")
    with open(mpath, "w") as fp:
        json.dump(manifest, fp, indent=2)

    print(f"capture: images={args.n_images} kidx={args.kidx} avg={args.avg} "
          f"samples={nsamp} adc_locked={scope.clock.adc_locked}", flush=True)

    stop = min(args.start_index + args.n_images, len(imgs))
    t0 = time.time()
    shard_traces, shard_imgs, shard_labels, shard_idx = [], [], [], []
    shard_id = 0
    for n in range(args.start_index, stop):
        img_bits = binarize_image(imgs[n], args.threshold)
        target.fpga_write(REG_IMAGE, img_bits.reshape(-1).astype(np.uint8).tolist())
        acc = np.zeros(nsamp, dtype=np.float32)
        for _ in range(args.avg):
            scope.arm()
            target.fpga_write(REG_GO, [1])
            if scope.capture():
                raise RuntimeError("capture timeout; check trigger, clock, shunt path")
            acc += np.asarray(scope.get_last_trace(), dtype=np.float32)
            wait_idle(target)
        acc /= args.avg
        shard_traces.append(np.clip(acc * TRACE_SCALE, -32768, 32767).astype(np.int16))
        shard_imgs.append(img_bits)
        shard_labels.append(int(labels[n]))
        shard_idx.append(n)
        if len(shard_traces) == args.shard_size or n == stop - 1:
            path = os.path.join(args.out, f"shard{shard_id:03d}.npz")
            np.savez_compressed(
                path,
                traces=np.stack(shard_traces),
                images=np.stack(shard_imgs).astype(np.uint8),
                labels=np.asarray(shard_labels, dtype=np.int16),
                indices=np.asarray(shard_idx, dtype=np.int32),
            )
            done = n - args.start_index + 1
            rate = done / max(time.time() - t0, 1e-6)
            print(f"shard{shard_id:03d} images={len(shard_traces)} total={done} "
                  f"rate={rate:.1f} img/s", flush=True)
            shard_traces, shard_imgs, shard_labels, shard_idx = [], [], [], []
            shard_id += 1

    manifest["status"] = "complete"
    manifest["shards"] = shard_id
    manifest["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(mpath, "w") as fp:
        json.dump(manifest, fp, indent=2)
    scope.dis()
    target.dis()


if __name__ == "__main__":
    main()
