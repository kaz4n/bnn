#!/usr/bin/env python3
"""Program one bitstream onto a freshly power-cycled CW305 and capture, in one pass.

Why this exists. This board will not reprogram over an already-configured FPGA: the
call reports success, DONE stays high, and the old design keeps running. So every
bitstream change costs a power cycle, and a power cycle spent on a failed attempt is
gone. Two were wasted on 2026-09-06 by programming with
`cw.target(bsfile=..., force=True)` and only discovering afterwards that the design
was dead.

This does the whole sequence while the FPGA is blank, using the programming call that
demonstrably worked earlier the same day -- a bare `cw.target()` with no bitstream,
then an explicit `fpga.FPGAProgram()` -- and refuses to continue unless the design
proves itself alive first.

Usage:
    python program_and_capture.py --bitstream ../build/<x>.bit --out traces_x [capture args]
"""
import argparse
import importlib.util
import os
import subprocess
import sys
import time

import numpy as np


def load_capture_module():
    argv, sys.argv = sys.argv, ["x"]
    spec = importlib.util.spec_from_file_location("cap", "cw305_p2p_capture.py")
    cap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cap)
    sys.argv = argv
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bitstream", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", default="fashion_test.npz")
    ap.add_argument("--grey", action="store_true", default=True)
    ap.add_argument("--kidx", type=int, default=0)
    ap.add_argument("--start-index", type=int, default=2000)
    ap.add_argument("--n-images", type=int, default=2000)
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--freq", type=float, default=5e6)
    ap.add_argument("--gain", type=float, default=40)
    ap.add_argument("--dwell", type=int, default=8)
    ap.add_argument("--avg", type=int, default=1)
    ap.add_argument("--shuffle-seed", type=int, default=8302028)
    ap.add_argument("--expect-binary", action="store_true",
                    help="probe expects the binary design (bit 0 only) rather than grey")
    args = ap.parse_args()

    import chipwhisperer as cw
    cap = load_capture_module()
    bs = os.path.abspath(args.bitstream)
    if not os.path.exists(bs):
        sys.exit(f"no such bitstream: {bs}")

    tgt = cw.target(None, cw.targets.CW305, fpga_id=None, force=False, slurp=False)

    # The FPGA must be blank. If it is already configured this call cannot replace
    # the design, and continuing would capture whatever is already loaded.
    if tgt.fpga.isFPGAProgrammed():
        sys.exit("FPGA is already configured. Power-cycle the board (cutting power, "
                 "not USB RST) and run this again while it is blank.")

    with open(bs, "rb") as f:
        status = tgt.fpga.FPGAProgram(f, exceptOnDoneFailure=False, prog_speed=20e6)
    time.sleep(0.6)
    print(f"programmed {os.path.basename(bs)}  status={status}  "
          f"DONE={tgt.fpga.isFPGAProgrammed()}", flush=True)

    # Prove the register interface is alive before spending a capture on it.
    cap.fpga_write_region(tgt, cap.REG_IMAGE, list(range(16)))
    rb = list(bytes(cap.fpga_read_region(tgt, cap.REG_IMAGE, 16)))
    expect = [i % 2 for i in range(16)] if args.expect_binary else list(range(16))
    print(f"byte-ramp readback: {rb[:8]}", flush=True)
    if rb != expect:
        tgt.dis()
        sys.exit(f"register interface dead (expected {expect[:8]}). Not capturing. "
                 "Power-cycle and retry; if it fails again the bitstream is bad.")
    print("register interface alive", flush=True)
    tgt.dis()

    cmd = [sys.executable, "cw305_p2p_capture.py",
           "--bitstream", bs, "--no-program",
           "--images", args.images, "--kidx", str(args.kidx),
           "--out", args.out,
           "--start-index", str(args.start_index), "--n-images", str(args.n_images),
           "--shard-size", str(args.shard_size), "--avg", str(args.avg),
           "--dwell", str(args.dwell), "--freq", str(args.freq),
           "--gain", str(args.gain), "--shuffle-seed", str(args.shuffle_seed)]
    if args.grey:
        cmd.append("--grey")
    print("capturing...", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(f"capture failed (exit {rc})")

    # Amplitude is the only thing that distinguishes the amplifier variants, since
    # they share a register map and the byte ramp cannot tell them apart.
    import glob
    sh = sorted(glob.glob(os.path.join(args.out, "shard*.npz")))
    if sh:
        z = np.load(sh[0])
        k = "traces" if "traces" in z.files else z.files[0]
        t = np.asarray(z[k], dtype=np.float32)
        print(f"MEAN ABS TRACE = {float(np.mean(np.abs(t))):.2f}  "
              f"(must differ from the other amplifier widths)", flush=True)


if __name__ == "__main__":
    main()
