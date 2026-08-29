#!/usr/bin/env python3
"""
P1/P3 bridge -- capture layer-1 conv power traces from CW305 with ChipWhisperer-Lite.

STATUS: HARDWARE-GATED. Requires the CW305 (programmed with the P1 bitstream) + a
CW-Lite/CW1173. Until you have the bench, use the P3 trace SIMULATOR
(../attack/trace_sim.py) which produces traces in the SAME .npz format, so the whole
attack pipeline (P3) runs end-to-end now and swaps to real traces later.

Synchronous capture (SETUP_PLAN.md C3): the BNN runs on tio_clkout, fed to CW-Lite
HS-In; CW-Lite samples phase-locked (adc_src='extclk_x4'), giving clean per-cycle power
directly -- the paper's async §5 (DC restore, curve fit) is NOT needed.

Output .npz (consumed by ../attack): traces float32 [n_kernels, n_samples],
kernel_ids, image (28x28 uint8), samples_per_cycle, n_cycles, label.
"""
import argparse, os, time
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bitstream", required=True)
    ap.add_argument("--kernels", required=True, help="P2 layer1_kernels.npy (64,K,K)")
    ap.add_argument("--ksize", type=int, choices=[3, 5], required=True)
    ap.add_argument("--n-kernels", type=int, default=9, help="paper uses 9 for template")
    ap.add_argument("--n-images", type=int, default=500)
    ap.add_argument("--avg", type=int, default=50, help="trace averages (boost SNR)")
    ap.add_argument("--freq", type=float, default=25e6, help="target clock Hz")
    ap.add_argument("--images", default="mnist_test.npz",
                    help="npz with arrays images(uint8 N,28,28) + labels; make with dump_mnist.py")
    ap.add_argument("--gain", type=float, default=30, help="CW-Lite LNA gain dB (max 55)")
    ap.add_argument("--out", default="traces")
    args = ap.parse_args()

    import chipwhisperer as cw          # only needed on the bench
    os.makedirs(args.out, exist_ok=True)

    scope = cw.scope()
    scope.clock.clkgen_freq = args.freq
    scope.default_setup()
    scope.clock.adc_src = "extclk_x4"   # synchronous to CW305 tio_clkout (4 samples/cycle)
    scope.clock.clkgen_freq = args.freq
    EW = 28 + 2 * (args.ksize // 2)          # extended raster width (line buffer scan)
    NCONV = EW * EW                          # 900 (3x3), 1024 (5x5)
    scope.adc.samples = 4 * NCONV + 300      # 4 samples/cycle over the whole conv + margin
    # The conv scans EW*EW extended-raster cycles; output pixel (oy,ox) is emitted at
    # extended cycle (oy+K-1)*EW+(ox+K-1). The Verilog trigger (cw305_reg_bnn.v) asserts
    # on the FIRST output write, so trace cycle 0 == extended cycle (K-1)*EW+(K-1). The
    # attack front-end uses ew+cycle0_offset to remap trace cycles back to 28x28 outputs;
    # without them it would misread extended-raster cycles as output pixels.
    CYCLE0_OFFSET = (args.ksize - 1) * EW + (args.ksize - 1)   # e.g. 62 (3x3), 132 (5x5)
    SAMPLE_RATE_HZ = 4.0 * args.freq         # adc_src=extclk_x4: 4 samples per DUT clock
    scope.trigger.triggers = "tio4"
    scope.gain.db = args.gain

    target = cw.target(scope, cw.targets.CW305, bsfile=args.bitstream,
                       fpga_id="100t", force=True)
    target.pll.pll_enable_set(True)
    target.pll.pll_outenable_set(True, 0)
    target.pll.pll_outenable_set(True, 1)
    target.pll.pll_outfreq_set(args.freq, 1)
    scope.clock.reset_adc(); time.sleep(0.3)

    # register (python fpga_write addr; byte addr = addr<<7) -- see cw305_reg_bnn.v
    REG_IMAGE  = 0     # 784 bytes  (bytes 0..783)
    REG_OUTPUT = 16    # 1568 bytes (bytes 2048..)  int16 LE feature map (functional check)
    REG_KERNEL = 32    # BPK bytes  (bytes 4096..)
    REG_GO     = 33    # write [1] -> run conv
    REG_STATUS = 34    # read bit0 = busy
    BPK = (args.ksize*args.ksize + 7) // 8
    kernels = np.load(args.kernels)[:args.n_kernels]

    def pack(k):
        bits = (k.flatten() > 0).astype(np.uint8)
        out = bytearray(); byte = nb = 0
        for b in bits:
            byte = (byte << 1) | int(b); nb += 1
            if nb == 8: out.append(byte); byte = nb = 0
        if nb: out.append(byte << (8 - nb))
        return bytes(out)

    d = np.load(args.images)                              # no TensorFlow needed in cwenv
    xte, yte = d["images"], d["labels"]

    for n in range(args.n_images):
        outp = os.path.join(args.out, f"img{n:04d}.npz")
        if os.path.exists(outp):                  # resumable: skip already-captured
            continue
        img = xte[n].astype(np.uint8)
        target.fpga_write(REG_IMAGE, list(img.flatten()))
        traces = []
        for kid in range(args.n_kernels):
            target.fpga_write(REG_KERNEL, list(pack(kernels[kid])))
            acc = None
            for _ in range(args.avg):
                scope.arm()
                target.fpga_write(REG_GO, [1])
                if scope.capture():            # True == timeout
                    raise RuntimeError("capture timeout -- check trigger wiring")
                w = scope.get_last_trace()
                acc = w if acc is None else acc + w
            traces.append(acc / args.avg)
        traces = np.asarray(traces, np.float32)
        np.savez(os.path.join(args.out, f"img{n:04d}.npz"),
                 traces=traces, kernel_ids=np.arange(args.n_kernels),
                 image=img, label=int(yte[n]),
                 samples_per_cycle=4, n_cycles=NCONV,
                 ew=EW, cycle0_offset=int(CYCLE0_OFFSET),
                 capture_cycles="extended", trace_source="cw305_sync",
                 sample_rate_hz=float(SAMPLE_RATE_HZ))
        if n % 25 == 0:
            print(f"captured {n+1}/{args.n_images}")
    scope.dis(); target.dis()


if __name__ == "__main__":
    main()
