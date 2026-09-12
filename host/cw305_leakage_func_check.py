#!/usr/bin/env python3
"""Functional check for cw305_leakage_top.

Programs the leakage-first 3x3 binary-conv bitstream, writes one binary MNIST
image and one 3x3 binary kernel, then reads back 26x26 signed scores. This proves
the FPGA register map and datapath before any power capture.
"""
import argparse
import os
import time

import numpy as np

LINE = 28
OUT_SIDE = 26
N_OUTPUT = OUT_SIDE * OUT_SIDE
REG_IMAGE = 0
REG_OUTPUT = 16
REG_KERNEL = 32
REG_GO = 33
REG_STATUS = 34
CTRL_CHUNK = 32
AES_SIGNATURE = (2, 5, 0x2E)


def binarize_image(img, threshold):
    return (np.asarray(img, dtype=np.uint8) > threshold).astype(np.uint8)


def golden_valid_conv_grey(img_grey, kern_bits):
    """Grey-input reference: kernel bit 1 adds the pixel, 0 subtracts it.

    Matches cw305_leakage_grey_top, the layer the paper attacks (real 0..255
    pixels against binary weights). Range is +/- 9*255."""
    signs = np.where(np.asarray(kern_bits, dtype=np.int16) != 0, 1, -1).astype(np.int16)
    src = img_grey.astype(np.int16)
    out = np.zeros(N_OUTPUT, dtype=np.int16)
    idx = 0
    for y in range(OUT_SIDE):
        for x in range(OUT_SIDE):
            patch = src[y:y + 3, x:x + 3].reshape(-1)
            out[idx] = np.int16(int(np.dot(patch, signs)))
            idx += 1
    return out


def prepare_image(img, threshold, grey):
    """What goes on the wire: the raw pixel in grey mode, else 0/1."""
    if grey:
        return img.astype(np.uint8)
    return binarize_image(img, threshold)


def kernel_bits_from_file(path, kidx):
    kernels = np.load(path)
    if kernels.ndim != 3 or kernels.shape[1:] != (3, 3):
        raise ValueError(f"expected kernels shaped (N,3,3), got {kernels.shape}")
    return (kernels[kidx].reshape(-1) > 0).astype(np.uint8)


def onehot_kernel(kidx):
    bits = np.zeros(9, dtype=np.uint8)
    bits[kidx % 9] = 1
    return bits


def pack_kernel_le(bits9):
    value = 0
    for i, bit in enumerate(np.asarray(bits9, dtype=np.uint8).reshape(9)):
        value |= (int(bit) & 1) << i
    return [value & 0xFF, (value >> 8) & 0x01]


def golden_valid_conv(img_bits, kern_bits):
    out = np.zeros(N_OUTPUT, dtype=np.int8)
    idx = 0
    for y in range(OUT_SIDE):
        for x in range(OUT_SIDE):
            patch = img_bits[y:y + 3, x:x + 3].reshape(-1)
            matches = int(np.count_nonzero(patch == kern_bits))
            out[idx] = np.int8(2 * matches - 9)
            idx += 1
    return out


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
    if sig == AES_SIGNATURE:
        raise RuntimeError(
            "CW305 is still responding as the stock AES design "
            "(pages 2/3/4 read 0x02/0x05/0x2e). The custom leakage bitstream "
            "is not active. Check S1 mode switches: USB programming requires "
            "M0=1, M1=1, M2=1, then power-cycle or press USB RST/SW3."
        )
    if not bool(target.fpga.isFPGAProgrammed()):
        raise RuntimeError("FPGA DONE did not read high after programming")
    return sig


def wait_idle(target, timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status = fpga_read_region(target, REG_STATUS, 1)[0]
        if (status & 1) == 0:
            return
        time.sleep(0.002)
    raise TimeoutError("FPGA did not go idle")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bitstream", default="../build/cw305_leakage_d8_l63.bit")
    ap.add_argument("--images", default="mnist_test.npz")
    ap.add_argument("--kernels", default="../training/artifacts/model_3x3/layer1_kernels.npy")
    ap.add_argument("--probe-kernels", choices=["onehot", "file"], default="onehot")
    ap.add_argument("--img", type=int, default=0)
    ap.add_argument("--kidx", type=int, default=0)
    ap.add_argument("--threshold", type=int, default=127)
    ap.add_argument("--freq", type=float, default=5e6)
    ap.add_argument("--clock-source", choices=["cw_lite", "target_pll"],
                    default="cw_lite")
    ap.add_argument("--fpga-id", default="100t")
    ap.add_argument("--no-program", action="store_true")
    ap.add_argument("--ones", action="store_true")
    ap.add_argument("--grey", action="store_true",
                    help="check the grey-pixel bitstream: send raw 0..255 pixels "
                         "and read 16-bit outputs.")
    args = ap.parse_args()

    import chipwhisperer as cw

    if args.ones:
        img_bits = np.ones((LINE, LINE), dtype=np.uint8)
        kern_bits = np.ones(9, dtype=np.uint8)
    else:
        d = np.load(args.images)
        imgs = d["images"]
        if imgs.ndim == 4 and imgs.shape[-1] == 1:
            imgs = imgs[..., 0]
        if imgs.ndim != 3 or imgs.shape[1:] != (LINE, LINE):
            raise ValueError(f"expected images shaped (N,28,28), got {imgs.shape}")
        img_bits = prepare_image(imgs[args.img], args.threshold, args.grey)
        if args.probe_kernels == "onehot":
            kern_bits = onehot_kernel(args.kidx)
        else:
            kern_bits = kernel_bits_from_file(args.kernels, args.kidx)

    scope = None
    if args.clock_source == "cw_lite":
        scope = cw.scope()
        scope.default_setup()
        scope.clock.adc_src = "clkgen_x4"
        scope.clock.clkgen_freq = args.freq
        scope.clock.reset_adc()

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
    time.sleep(0.2)
    design_sig = verify_custom_design_loaded(target)

    fpga_write_region(target, REG_IMAGE, img_bits.reshape(-1).astype(np.uint8).tolist())
    fpga_write_region(target, REG_KERNEL, pack_kernel_le(kern_bits))
    target.fpga_read(REG_KERNEL, 2)
    fpga_write_region(target, REG_GO, [1])
    wait_idle(target, 1.0)

    if args.grey:
        raw = bytes(fpga_read_region(target, REG_OUTPUT, N_OUTPUT * 2))
        hw = np.frombuffer(raw, dtype="<i2")
        golden = golden_valid_conv_grey(img_bits, kern_bits)
    else:
        raw = bytes(fpga_read_region(target, REG_OUTPUT, N_OUTPUT))
        hw = np.frombuffer(raw, dtype=np.int8)
        golden = golden_valid_conv(img_bits, kern_bits)
    mism = np.nonzero(hw != golden)[0]

    print(f"bitstream={bsfile or '(already programmed)'}")
    print(f"design_signature_pages_2_3_4={list(design_sig)}")
    print(f"freq={args.freq:g} image_bits={int(img_bits.sum())} "
          f"kernel_bits={''.join(str(int(x)) for x in kern_bits)}")
    print(f"hw range={int(hw.min())}..{int(hw.max())} "
          f"golden range={int(golden.min())}..{int(golden.max())}")
    if len(mism):
        print(f"FAIL: {len(mism)}/{N_OUTPUT} mismatches")
        for i in mism[:8]:
            print(f"  out[{i}] y={i // OUT_SIDE} x={i % OUT_SIDE}: "
                  f"hw={int(hw[i])} golden={int(golden[i])}")
        target.dis()
        raise SystemExit(1)

    print(f"PASS: {N_OUTPUT} outputs bit-exact")
    target.dis()
    if scope is not None:
        scope.dis()


if __name__ == "__main__":
    main()
