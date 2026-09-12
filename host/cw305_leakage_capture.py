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


def golden_valid_conv_grey(img_grey, kern_bits):
    """Grey-input reference: a kernel bit of 1 adds the pixel, 0 subtracts it.

    This is what cw305_leakage_grey_top computes, and it is the layer the paper
    attacks (real 0..255 pixels against binary weights). Range is +/- 9*255."""
    signs = np.where(np.asarray(kern_bits, dtype=np.int16) != 0, 1, -1).astype(np.int16)
    src = img_grey.astype(np.int16)
    out = np.zeros(N_WINDOWS, dtype=np.int16)
    idx = 0
    for y in range(OUT_SIDE):
        for x in range(OUT_SIDE):
            patch = src[y:y + 3, x:x + 3].reshape(-1)
            out[idx] = np.int16(int(np.dot(patch, signs)))
            idx += 1
    return out


def read_outputs(target, grey):
    """Read the 676 convolution outputs. Grey outputs are 16-bit little-endian."""
    if grey:
        raw = bytes(fpga_read_region(target, REG_OUTPUT, N_WINDOWS * 2))
        return np.frombuffer(raw, dtype="<i2")
    raw = bytes(fpga_read_region(target, REG_OUTPUT, N_WINDOWS))
    return np.frombuffer(raw, dtype=np.int8)


def prepare_image(img, threshold, grey):
    """What actually goes on the wire: the raw pixel in grey mode, else 0/1."""
    if grey:
        return img.astype(np.uint8)
    return binarize_image(img, threshold)


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
    """Write a possibly cross-page CW305 region through small control transfers.

    ChipWhisperer automatically uses the bulk endpoint for large fpga_write()
    calls.  Keeping each transfer below the control/bulk threshold makes the
    functional check easier to trust on older CW305 firmware.
    """
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
    """Fail fast if the CW305 is still serving the stock AES register map."""
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
        time.sleep(0.002)
    raise TimeoutError("FPGA did not go idle")


def run_functional_check(target, img_bits, kern_bits, grey=False):
    fpga_write_region(target, REG_IMAGE, img_bits.reshape(-1).astype(np.uint8).tolist())
    fpga_write_region(target, REG_KERNEL, pack_kernel_le(kern_bits))
    fpga_write_region(target, REG_GO, [1])
    wait_idle(target)
    hw = read_outputs(target, grey)
    golden = (golden_valid_conv_grey(img_bits, kern_bits) if grey
              else golden_valid_conv(img_bits, kern_bits))
    result = {
        "passed": bool(np.array_equal(hw, golden)),
        "input_mode": "grey" if grey else "binary",
        "n_outputs": int(N_WINDOWS),
        "hw_min": int(hw.min()),
        "hw_max": int(hw.max()),
        "golden_min": int(golden.min()),
        "golden_max": int(golden.max()),
        "mismatches": int(np.count_nonzero(hw != golden)),
        "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if not np.array_equal(hw, golden):
        bad = int(np.nonzero(hw != golden)[0][0])
        raise RuntimeError(
            f"functional check failed at out[{bad}]: hw={int(hw[bad])} "
            f"golden={int(golden[bad])}"
        )
    return result


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
    ap.add_argument("--grey", action="store_true",
                    help="target the grey-pixel bitstream: send raw 0..255 pixels "
                         "and read 16-bit outputs. Needs cw305_leakage_grey_top.")
    ap.add_argument("--freq", type=float, default=5e6)
    ap.add_argument("--clock-source", choices=["cw_lite", "target_pll"],
                    default="cw_lite",
                    help="cw_lite drives FPGA through HS2; target_pll expects target clockout")
    ap.add_argument("--dwell", type=int, default=8)
    ap.add_argument("--gain", type=float, default=40)
    ap.add_argument("--margin-samples", type=int, default=512)
    ap.add_argument("--presamples", type=int, default=0)
    ap.add_argument("--fpga-id", default="100t")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-program", action="store_true")
    ap.add_argument("--skip-functional-check", action="store_true")
    ap.add_argument("--shuffle-seed", type=int, default=None,
                    help="randomize capture order to reduce time/temperature confounding")
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
    if args.clock_source == "cw_lite":
        scope.clock.adc_src = "clkgen_x4"
        scope.clock.clkgen_freq = args.freq
    else:
        scope.clock.adc_src = "extclk_x4"
    scope.adc.samples = nsamp
    scope.adc.offset = 0
    scope.adc.presamples = args.presamples
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

    first_img = prepare_image(imgs[args.start_index], args.threshold, args.grey)
    functional_check = {"skipped": True}
    if not args.skip_functional_check:
        functional_check = run_functional_check(target, first_img, kernel_bits[0],
                                                grey=args.grey)

    stop = min(args.start_index + args.n_images, len(imgs))
    capture_order = np.arange(args.start_index, stop, dtype=np.int32)
    if args.shuffle_seed is not None:
        rng = np.random.default_rng(args.shuffle_seed)
        rng.shuffle(capture_order)

    manifest = {
        "bitstream": os.path.abspath(args.bitstream),
        "bitstream_sha256": sha256_file(args.bitstream) if os.path.exists(args.bitstream) else None,
        "images": os.path.abspath(args.images),
        "images_sha256": sha256_file(args.images) if os.path.exists(args.images) else None,
        "kernels": os.path.abspath(args.kernels),
        "kernels_sha256": sha256_file(args.kernels) if os.path.exists(args.kernels) else None,
        "probe_kernels": args.probe_kernels,
        "trace_source": "cw_lite_analog_sync_x4",
        "design": "cw305_leakage_grey_top" if args.grey else "cw305_leakage_top",
        "input_mode": "grey" if args.grey else "binary",
        "line": LINE,
        "out_side": OUT_SIDE,
        "n_windows": N_WINDOWS,
        "n_kernels": args.n_kernels,
        "avg": args.avg,
        "threshold": args.threshold,
        "fpga_freq_hz": args.freq,
        "clock_source": args.clock_source,
        "adc_src": scope.clock.adc_src,
        "samples_per_cycle": spc,
        "dwell": args.dwell,
        "samples_per_window": args.dwell * spc,
        "samples": nsamp,
        "useful_samples": useful_samples,
        "presamples": args.presamples,
        "gain_db": args.gain,
        "adc_locked_initial": bool(scope.clock.adc_locked),
        "capture_mode": "live_cw305_chipwhisperer",
        "design_check": design_check,
        "functional_check": functional_check,
        "capture_order": capture_order.astype(int).tolist(),
        "shuffle_seed": args.shuffle_seed,
        "status": "incomplete",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest.update(hardware_identity(cw, scope, target))
    with open(os.path.join(args.out, "capture_manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=2)

    print(f"capture: images={args.n_images} kernels={args.n_kernels} avg={args.avg} "
          f"samples={nsamp} adc_locked={scope.clock.adc_locked}")

    for order_pos, n in enumerate(capture_order):
        n = int(n)
        outp = os.path.join(args.out, f"img{n:04d}.npz")
        if os.path.exists(outp) and not args.force:
            continue
        img_bits = prepare_image(imgs[n], args.threshold, args.grey)
        fpga_write_region(target, REG_IMAGE, img_bits.reshape(-1).astype(np.uint8).tolist())
        repeats = np.zeros((args.n_kernels, args.avg, nsamp), dtype=np.float32)
        for kid in range(args.n_kernels):
            fpga_write_region(target, REG_KERNEL, packed[kid])
            for rep in range(args.avg):
                scope.arm()
                fpga_write_region(target, REG_GO, [1])
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
            capture_order_pos=int(order_pos),
            capture_utc=np.asarray(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        )
        print(f"captured img{n:04d} label={int(labels[n])}", flush=True)

    manifest["status"] = "complete"
    manifest["adc_locked_final"] = bool(scope.clock.adc_locked)
    manifest["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(args.out, "capture_manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=2)
    scope.dis()
    target.dis()


if __name__ == "__main__":
    main()
