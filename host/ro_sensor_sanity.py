#!/usr/bin/env python3
"""
BLOCKING sanity check before trusting the RO/TDC sensor channel.

Confirms the sensor register returns a REAL per-cycle droop vector, not an echo of the
int16 conv output feature map. Runs 2 different images x 1 kernel, reads candidate sensor
regs (16 and 48), and reports for each:
  - nonzero / variance         (is it alive?)
  - changes between images?     (is it data-dependent?)
  - corr with GOLDEN conv out   (is it just the output map? -> corr ~1 == NOT a sensor)

Verdict: the sensor reg = the one that is alive, image-dependent, and NOT ~equal to golden.
Run in cwenv:  C:\\Users\\narut\\ChipWhisperer\\cwenv\\Scripts\\python.exe ro_sensor_sanity.py
"""
import argparse, time
import numpy as np

LINE = 28; IMG = LINE * LINE
REG_IMAGE, REG_OUTPUT, REG_KERNEL, REG_GO, REG_STATUS = 0, 16, 32, 33, 34
REG_SENSOR_CTRL = 35


def pack_kernel(k):
    bits = (k.flatten() > 0).astype(np.uint8); out = bytearray(); byte = nb = 0
    for b in bits:
        byte = (byte << 1) | int(b); nb += 1
        if nb == 8: out.append(byte); byte = nb = 0
    if nb: out.append(byte << (8 - nb))
    return list(out)


def golden_conv(img, kern, K):
    p = K // 2; out = np.zeros(IMG, np.int32)
    for y in range(LINE):
        for x in range(LINE):
            acc = 0
            for i in range(K):
                for j in range(K):
                    yy, xx = y + i - p, x + j - p
                    pv = int(img[yy, xx]) if (0 <= yy < LINE and 0 <= xx < LINE) else 0
                    acc += pv if kern[i, j] == 1 else -pv
            out[y * LINE + x] = acc
    return out


def wait_idle(target, t=2.0):
    dl = time.time() + t
    while time.time() < dl:
        if (target.fpga_read(REG_STATUS, 1)[0] & 1) == 0:
            return
        time.sleep(0.001)


def read_u16(target, reg):
    raw = bytes(target.fpga_read(reg, 2 * IMG))
    return np.frombuffer(raw, dtype="<u2").astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bitstream", default="../build/cw305_bnn_ro3.bit")
    ap.add_argument("--kernels", default="../training/artifacts/model_3x3/layer1_kernels.npy")
    ap.add_argument("--ksize", type=int, default=3)
    ap.add_argument("--images", default="mnist_test.npz")
    ap.add_argument("--freq", type=float, default=10e6)
    ap.add_argument("--regs", type=int, nargs="+", default=[16, 48])
    ap.add_argument("--run-wait", type=float, default=0.05)
    args = ap.parse_args()

    import chipwhisperer as cw
    xte = np.load(args.images)["images"]
    kern = np.load(args.kernels)[0].astype(np.int8)
    packed = pack_kernel(kern)

    target = cw.target(None, cw.targets.CW305, bsfile=args.bitstream, fpga_id="100t", force=True)
    try:
        target.pll.pll_enable_set(True)
        target.pll.pll_outenable_set(True, 0); target.pll.pll_outenable_set(True, 1)
        target.pll.pll_outfreq_set(args.freq, 1)
        try:
            target.fpga_write(REG_SENSOR_CTRL, [1]); time.sleep(0.1)
            print("sensor ctrl (reg35) enabled")
        except Exception as e:
            print(f"sensor ctrl write failed (may be unused): {e}")

        caps = {}
        for imn in (0, 5):
            img = xte[imn].astype(np.uint8)
            target.fpga_write(REG_IMAGE, img.flatten().tolist())
            target.fpga_write(REG_KERNEL, packed)
            target.fpga_write(REG_GO, [1]); time.sleep(args.run_wait); wait_idle(target)
            caps[imn] = {r: read_u16(target, r) for r in args.regs}
        gold = {imn: golden_conv(xte[imn], kern, args.ksize) for imn in (0, 5)}
        # golden as int16 wrap -> unsigned view for corr against raw reg read
        gold_u = {imn: (gold[imn].astype(np.int16).astype(np.uint16)).astype(np.float64) for imn in (0, 5)}

        print(f"\n{'reg':>4} {'nonzero':>8} {'std':>10} {'d(img0,img5)':>13} {'corr_golden':>12}  verdict")
        for r in args.regs:
            v0, v5 = caps[0][r], caps[5][r]
            nz = int(np.count_nonzero(v0)); std = float(v0.std())
            dimg = float(np.mean(np.abs(v0 - v5)))
            # corr vs golden: try both signed-int16 and raw-unsigned interpretations
            def safe_corr(a, b):
                if a.std() < 1e-9 or b.std() < 1e-9: return 0.0
                return float(np.corrcoef(a, b)[0, 1])
            cg = max(abs(safe_corr(v0, gold_u[0])),
                     abs(safe_corr(v0.astype(np.int16).astype(np.float64),
                                   gold[0].astype(np.float64))))
            if nz == 0:
                verdict = "DEAD (all zero)"
            elif cg > 0.98:
                verdict = "ECHOES CONV OUTPUT (not a sensor)"
            elif dimg < 1e-6:
                verdict = "constant across images (not data-dep)"
            else:
                verdict = "*** LIVE SENSOR CANDIDATE ***"
            print(f"{r:>4} {nz:>8} {std:>10.2f} {dimg:>13.3f} {cg:>12.3f}  {verdict}")
        # dump raw for offline look
        np.savez("ro_sanity_dump.npz",
                 **{f"reg{r}_img{imn}": caps[imn][r] for r in args.regs for imn in (0, 5)},
                 gold0=gold[0], gold5=gold[5])
        print("\nsaved ro_sanity_dump.npz")
    finally:
        target.dis()


if __name__ == "__main__":
    main()
