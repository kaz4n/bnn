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
CTRL_CHUNK = 32
AES_SIGNATURE = (2, 5, 0x2E)


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


def golden_valid_conv_grey(img_grey, kern_bits):
    """Grey reference: kernel bit 1 adds the pixel, 0 subtracts it (+/- 9*255)."""
    signs = np.where(np.asarray(kern_bits, dtype=np.int16) != 0, 1, -1).astype(np.int16)
    src = img_grey.astype(np.int16)
    out = np.zeros(N_WINDOWS, dtype=np.int16)
    idx = 0
    for y in range(OUT_SIDE):
        for x in range(OUT_SIDE):
            out[idx] = np.int16(int(np.dot(src[y:y + 3, x:x + 3].reshape(-1), signs)))
            idx += 1
    return out


def prepare_image(img, threshold, grey):
    """What goes on the wire: the raw pixel in grey mode, else 0/1."""
    if grey:
        return img.astype(np.uint8)
    return binarize_image(img, threshold)


def golden_valid_conv(img_bits, kern_bits):
    out = np.zeros(N_WINDOWS, dtype=np.int8)
    idx = 0
    for y in range(OUT_SIDE):
        for x in range(OUT_SIDE):
            patch = img_bits[y:y + 3, x:x + 3].reshape(-1)
            out[idx] = np.int8(2 * int(np.count_nonzero(patch == kern_bits)) - 9)
            idx += 1
    return out


def hardware_identity(cw, scope, target):
    """Best-effort live hardware identity for provenance in capture manifests."""
    return {
        "hardware_run": True,
        "chipwhisperer_version": str(getattr(cw, "__version__", "unknown")),
        "scope_serial": str(getattr(scope, "sn", "")),
        "target_serial": str(getattr(target, "sn", "")),
    }


def fpga_base_addr(target, page_addr):
    return int(page_addr) << int(getattr(target, "bytecount_size", 7))


def fpga_write_region(target, page_addr, data, chunk=CTRL_CHUNK):
    if not hasattr(target, "_naeusb") or target._naeusb is None:
        target.fpga_write(page_addr, data)
        return
    base = fpga_base_addr(target, page_addr)
    payload = [int(x) & 0xFF for x in data]
    for off in range(0, len(payload), chunk):
        target._naeusb.cmdWriteMem(base + off, payload[off:off + chunk])


def fpga_read_region(target, page_addr, length, chunk=CTRL_CHUNK):
    if not hasattr(target, "_naeusb") or target._naeusb is None:
        return target.fpga_read(page_addr, length)
    base = fpga_base_addr(target, page_addr)
    out = bytearray()
    for off in range(0, int(length), chunk):
        out.extend(target._naeusb.cmdReadMem(base + off, min(chunk, int(length) - off)))
    return out


def verify_custom_design_loaded(target):
    sig = tuple(int(fpga_read_region(target, page, 1)[0]) for page in (2, 3, 4))
    result = {
        "fpga_done": bool(target.fpga.isFPGAProgrammed()),
        "stock_aes_signature_pages_2_3_4": list(sig),
        "verified_not_stock_aes": sig != AES_SIGNATURE,
    }
    if sig == AES_SIGNATURE:
        raise RuntimeError(
            "CW305 is still responding as the stock AES design "
            "(pages 2/3/4 read 0x02/0x05/0x2e). The custom leakage bitstream "
            "was not the active FPGA image. Check S1 mode switches: USB "
            "programming requires M0=1, M1=1, M2=1, then power-cycle or press "
            "USB RST/SW3 before retrying."
        )
    if not result["fpga_done"]:
        raise RuntimeError("FPGA DONE did not read high after programming")
    return result


def wait_idle(target, timeout_s=1.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if (fpga_read_region(target, REG_STATUS, 1)[0] & 1) == 0:
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
    ap.add_argument("--grey", action="store_true",
                    help="target the grey-pixel bitstream: send raw 0..255 pixels "
                         "and read 16-bit outputs.")
    ap.add_argument("--freq", type=float, default=5e6)
    ap.add_argument("--clock-source", choices=["cw_lite", "target_pll"],
                    default="cw_lite",
                    help="cw_lite drives FPGA through HS2; target_pll expects target clockout")
    ap.add_argument("--dwell", type=int, default=8)
    ap.add_argument("--gain", type=float, default=40)
    ap.add_argument("--margin-samples", type=int, default=512)
    ap.add_argument("--fpga-id", default="100t")
    ap.add_argument("--no-program", action="store_true")
    ap.add_argument("--shuffle-seed", type=int, default=None,
                    help="randomize capture order to reduce time/temperature confounding")
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
    if args.clock_source == "cw_lite":
        scope.clock.adc_src = "clkgen_x4"
        scope.clock.clkgen_freq = args.freq
    else:
        scope.clock.adc_src = "extclk_x4"
    scope.adc.samples = nsamp
    scope.adc.offset = 0
    scope.adc.presamples = 0
    scope.adc.basic_mode = "rising_edge"
    scope.trigger.triggers = "tio4"
    scope.gain.db = args.gain

    bsfile = None if args.no_program else os.path.abspath(args.bitstream)
    target = cw.target(None, cw.targets.CW305, bsfile=bsfile,
                       fpga_id=args.fpga_id, force=not args.no_program,
                       slurp=False)
    target.clkusbautooff = False
    target.pll.pll_enable_set(True)
    if args.clock_source == "cw_lite":
        target.pll.pll_outenable_set(False, 0)
        target.pll.pll_outenable_set(False, 1)
    else:
        target.pll.pll_outenable_set(False, 0)
        target.pll.pll_outenable_set(True, 1)
        target.pll.pll_outfreq_set(args.freq, 1)
    scope.clock.reset_adc()
    time.sleep(0.2)
    design_check = verify_custom_design_loaded(target)
    fpga_write_region(target, REG_KERNEL, pack_kernel_le(kernel_bits))

    # functional check on the first image so the shards are provably silicon data
    first = prepare_image(imgs[args.start_index], args.threshold, args.grey)
    fpga_write_region(target, REG_IMAGE, first.reshape(-1).astype(np.uint8).tolist())
    fpga_write_region(target, REG_GO, [1])
    wait_idle(target)
    if args.grey:
        hw = np.frombuffer(
            bytes(fpga_read_region(target, REG_OUTPUT, N_WINDOWS * 2)), dtype='<i2')
        golden = golden_valid_conv_grey(first, kernel_bits)
    else:
        hw = np.frombuffer(
            bytes(fpga_read_region(target, REG_OUTPUT, N_WINDOWS)), dtype=np.int8)
        golden = golden_valid_conv(first, kernel_bits)
    functional_check = {
        "passed": bool(np.array_equal(hw, golden)),
        "n_outputs": int(N_WINDOWS),
        "hw_min": int(hw.min()),
        "hw_max": int(hw.max()),
        "golden_min": int(golden.min()),
        "golden_max": int(golden.max()),
        "mismatches": int(np.count_nonzero(hw != golden)),
        "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if not functional_check["passed"]:
        raise RuntimeError("functional check failed before bulk capture")
    print("PASS: 676 outputs bit-exact", flush=True)

    stop = min(args.start_index + args.n_images, len(imgs))
    capture_order = np.arange(args.start_index, stop, dtype=np.int32)
    if args.shuffle_seed is not None:
        rng = np.random.default_rng(args.shuffle_seed)
        rng.shuffle(capture_order)

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
        "input_mode": "grey" if args.grey else "binary",
        "fpga_freq_hz": args.freq,
        "clock_source": args.clock_source,
        "adc_src": scope.clock.adc_src,
        "samples_per_cycle": spc,
        "dwell": args.dwell,
        "samples_per_window": args.dwell * spc,
        "samples": nsamp,
        "useful_samples": useful,
        "gain_db": args.gain,
        "adc_locked_initial": bool(scope.clock.adc_locked),
        "capture_mode": "live_cw305_chipwhisperer",
        "design_check": design_check,
        "functional_check": functional_check,
        "capture_order": capture_order.astype(int).tolist(),
        "shuffle_seed": args.shuffle_seed,
        "trace_dtype": "int16",
        "trace_scale": TRACE_SCALE,
        "start_index": args.start_index,
        "n_images": args.n_images,
        "shard_size": args.shard_size,
        "status": "incomplete",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest.update(hardware_identity(cw, scope, target))
    mpath = os.path.join(args.out, "capture_manifest.json")
    with open(mpath, "w") as fp:
        json.dump(manifest, fp, indent=2)

    print(f"capture: images={args.n_images} kidx={args.kidx} avg={args.avg} "
          f"samples={nsamp} adc_locked={scope.clock.adc_locked}", flush=True)

    t0 = time.time()
    shard_traces, shard_imgs, shard_labels, shard_idx = [], [], [], []
    shard_id = 0
    for order_pos, n in enumerate(capture_order):
        n = int(n)
        img_bits = prepare_image(imgs[n], args.threshold, args.grey)
        fpga_write_region(target, REG_IMAGE, img_bits.reshape(-1).astype(np.uint8).tolist())
        acc = np.zeros(nsamp, dtype=np.float32)
        for _ in range(args.avg):
            scope.arm()
            fpga_write_region(target, REG_GO, [1])
            if scope.capture():
                raise RuntimeError("capture timeout; check trigger, clock, shunt path")
            acc += np.asarray(scope.get_last_trace(), dtype=np.float32)
            wait_idle(target)
        acc /= args.avg
        shard_traces.append(np.clip(acc * TRACE_SCALE, -32768, 32767).astype(np.int16))
        shard_imgs.append(img_bits)
        shard_labels.append(int(labels[n]))
        shard_idx.append(n)
        if len(shard_traces) == args.shard_size or order_pos == len(capture_order) - 1:
            path = os.path.join(args.out, f"shard{shard_id:03d}.npz")
            np.savez_compressed(
                path,
                traces=np.stack(shard_traces),
                images=np.stack(shard_imgs).astype(np.uint8),
                labels=np.asarray(shard_labels, dtype=np.int16),
                indices=np.asarray(shard_idx, dtype=np.int32),
            )
            done = order_pos + 1
            rate = done / max(time.time() - t0, 1e-6)
            print(f"shard{shard_id:03d} images={len(shard_traces)} total={done} "
                  f"rate={rate:.1f} img/s", flush=True)
            shard_traces, shard_imgs, shard_labels, shard_idx = [], [], [], []
            shard_id += 1

    manifest["status"] = "complete"
    manifest["shards"] = shard_id
    manifest["adc_locked_final"] = bool(scope.clock.adc_locked)
    manifest["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(mpath, "w") as fp:
        json.dump(manifest, fp, indent=2)
    scope.dis()
    target.dis()


if __name__ == "__main__":
    main()
