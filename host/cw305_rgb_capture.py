#!/usr/bin/env python3
"""Capture RGB power traces from the CW305 rgb_linebuf_core with a ChipWhisperer-Lite.

Companion to host/cw305_p2p_capture.py, for the three-channel design. Same conventions:
control-transfer register access, synchronous x4 ADC capture, a functional check against
a golden model before any trace is trusted, and a manifest that records exactly what was
measured.

THREE THINGS THIS DOES THAT THE GREY SCRIPT DOES NOT
----------------------------------------------------
1. Positive design identification. The grey script proves the board is NOT the stock AES
   image. That is a weaker test than it looks: any previously-loaded custom bitstream
   also passes it, which is how the 2026-09-06 amplifier sweep captured the same
   bitstream four times while believing it had reprogrammed between points
   (attack/hardtests_20260830/AMP_SWEEP_VOID.md). This design answers "RGBLBUF1" on its
   signature page, so the check is positive: this exact design, or refuse to capture.

2. Dataflow mode is set by register, not by bitstream. All three channel dataflows are
   measured from one programmed image, so no run in a mode sweep can silently reuse the
   previous condition.

3. Captures are SEGMENTED. A full 32x32 scan at DWELL=8 is 32,768 ADC samples and
   98,304 in serial mode, against a CW-Lite limit near 24,573. The core always scans the
   whole image so the line buffer stays coherent and frames only the trigger, so a full
   scan is reassembled from a few captures. Geometry and DWELL are preserved.

4. The functional check runs per mode. All three modes must return bit-identical,
   golden-matching outputs -- that invariant is asserted in simulation
   (cw305/tb_cw305_reg_linebuf_rgb.sv) and re-asserted here on silicon, because a
   cross-mode power comparison is meaningless if the modes compute different things.

Programming note: reprogramming the CW305 over an already-configured FPGA silently fails
on this board. Power-cycle it (cutting power; USB RST is not enough) and program while
blank. This script verifies the signature after programming and refuses to continue if it
does not match, so a failed reprogram becomes a loud error rather than a quiet wrong
result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np

IMG_SIDE = 32
PLANE = IMG_SIDE * IMG_SIDE            # 1024 pixel positions
N_POS = PLANE                          # line buffer streams every position
IMG_BYTES = PLANE * 4                  # 4096 bytes, RGBx per pixel
OUT_SIDE = IMG_SIDE - 2                # 30
N_WINDOWS = OUT_SIDE * OUT_SIDE        # 900 valid outputs

# Pages are byte-address >> 7 and must match cw305_reg_linebuf_rgb.sv.
REG_IMAGE = 0        # byte 0      .. 4095  (4 bytes per pixel, 4th ignored)
REG_GO = 33
REG_STATUS = 34
REG_SIG = 35
REG_OUTPUT = 64      # byte 8192   .. 9991
REG_CTRL = 128       # byte 16384  .. 16392

# Control-page byte offsets within REG_CTRL.
CTRL_KERNEL = 0      # 4 bytes
CTRL_MODE = 4
CTRL_SEG_START = 5   # 2 bytes, little-endian
CTRL_SEG_LEN = 7     # 2 bytes, little-endian

# ChipWhisperer-Lite ADC FIFO depth. A full 32x32 scan at DWELL=8 is 32,768 samples
# (98,304 in serial mode), so captures are segmented; see plan_segments().
MAX_SAMPLES = 24400

DESIGN_SIGNATURE = b"RGBLBUF1"
TRACE_SCALE = 32767.0
CTRL_CHUNK = 32
MODES = {"serial": 0, "parallel": 1, "summed": 2}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(bytes(b)).hexdigest()


# ---------------------------------------------------------------- register access


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


def verify_design_signature(target):
    """Positive identification: refuse to capture unless THIS design is live."""
    sig = bytes(fpga_read_region(target, REG_SIG, 8))
    done = bool(target.fpga.isFPGAProgrammed())
    if sig != DESIGN_SIGNATURE:
        raise RuntimeError(
            f"CW305 signature page reads {sig!r}, expected {DESIGN_SIGNATURE!r}. "
            "The RGB leakage bitstream is not the active FPGA image. Reprogramming "
            "over a configured FPGA silently fails on this board: power-cycle the "
            "CW305 (cut power; USB RST is not enough) and program it while blank. "
            "Do not capture until this check passes -- a stale bitstream produces "
            "plausible traces of the wrong design."
        )
    if not done:
        raise RuntimeError("FPGA DONE did not read high after programming")
    return {"fpga_done": done, "signature": sig.decode(), "verified_rgb_design": True}


def wait_idle(target, timeout_s=5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if (fpga_read_region(target, REG_STATUS, 1)[0] & 1) == 0:
            return
        time.sleep(0.001)
    raise TimeoutError("FPGA did not go idle")


# ---------------------------------------------------------------- model + payloads


def pack_kernel_le(bits27):
    """27 kernel bits -> 4 little-endian bytes, matching the register file decode."""
    v = 0
    for i, b in enumerate(bits27):
        if int(b):
            v |= (1 << i)
    return [v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0x07]


def image_to_rgbx(img_chw):
    """(3, IMG_SIDE, IMG_SIDE) uint8 -> IMG_BYTES of RGBx, four bytes per pixel.

    Four bytes rather than three so the register file's position decode is a bit slice
    (byteaddr[11:2]) instead of a divide by three. The fourth byte is padding and is
    ignored by the hardware.
    """
    a = np.asarray(img_chw, dtype=np.uint8)
    if a.shape != (3, IMG_SIDE, IMG_SIDE):
        raise ValueError(f"expected (3,{IMG_SIDE},{IMG_SIDE}), got {a.shape}")
    flat = a.reshape(3, PLANE).T                       # (PLANE, 3) position-major
    out = np.zeros((PLANE, 4), dtype=np.uint8)
    out[:, :3] = flat
    return out.reshape(-1).tolist()


def plan_segments(mode_name, dwell, samples_per_cycle=4, max_samples=MAX_SAMPLES):
    """Split a full scan into trigger-framed segments that fit the ADC FIFO.

    The core always scans the whole image so the line buffer stays coherent; only the
    trigger is framed. Shrinking the image or collapsing DWELL would have been the
    alternative, and DWELL is where the measured settling structure lives, so segmenting
    is the cheaper trade -- it costs acquisition time and nothing else.
    """
    per_pos = dwell * samples_per_cycle * (3 if mode_name == "serial" else 1)
    seg_len = max(1, max_samples // per_pos)
    starts = list(range(0, N_POS, seg_len))
    return [(int(st), int(min(seg_len, N_POS - st))) for st in starts], per_pos


def golden_valid_conv_rgb(img_chw, bits27):
    """Independent reference for the functional check. Identical in all three modes."""
    a = np.asarray(img_chw, dtype=np.int32)
    k = np.where(np.asarray(bits27, dtype=np.int32) > 0, 1, -1).reshape(3, 3, 3)
    win = np.lib.stride_tricks.sliding_window_view(a, (3, 3), axis=(1, 2))
    win = np.transpose(win, (1, 2, 0, 3, 4)).reshape(N_WINDOWS, 3, 9)
    return (win * k.reshape(3, 9)[None, :, :]).sum(axis=(1, 2)).astype(np.int32)


def read_outputs(target):
    raw = fpga_read_region(target, REG_OUTPUT, N_WINDOWS * 2)
    return np.frombuffer(bytes(raw), dtype="<i2").astype(np.int32)


def write_control(target, kbytes, mode_val, seg_start, seg_len):
    payload = list(kbytes) + [mode_val,
                              seg_start & 0xFF, (seg_start >> 8) & 0xFF,
                              seg_len & 0xFF, (seg_len >> 8) & 0xFF]
    fpga_write_region(target, REG_CTRL, payload)


def run_once(target, scope, rgbx, kbytes, mode_val, seg_start=0, seg_len=0):
    """Load inputs, set mode and segment, trigger, capture one trace, read outputs."""
    fpga_write_region(target, REG_IMAGE, rgbx)
    write_control(target, kbytes, mode_val, seg_start, seg_len)
    wait_idle(target)
    scope.arm()
    fpga_write_region(target, REG_GO, [1])
    scope.capture()
    wait_idle(target)
    return scope.get_last_trace(), read_outputs(target)


def capture_segmented(target, scope, rgbx, kbytes, mode_val, segments, per_pos):
    """Capture every segment of one image and concatenate into a full-scan trace.

    Each capture returns a full ADC buffer, but the LAST segment frames fewer positions
    than the buffer holds (1024 positions do not divide evenly into 762-position
    segments). Everything past the framed region is post-scan idle, so each segment is
    TRIMMED to seg_len * per_pos samples before concatenation. Without the trim the stored
    trace carries ~16k samples of idle and no longer maps position -> sample offset, which
    silently misaligns every downstream featurization.
    """
    parts = []
    outs = None
    for seg_start, seg_len in segments:
        tr, o = run_once(target, scope, rgbx, kbytes, mode_val, seg_start, seg_len)
        want = seg_len * per_pos
        a = np.asarray(tr, dtype=np.float64)
        if len(a) < want:
            raise RuntimeError(
                f"segment at {seg_start} returned {len(a)} samples, need {want}. "
                "Raise scope.adc.samples or shorten the segment.")
        parts.append(a[:want])
        outs = o                       # identical every segment; framing gates the trigger only
    return np.concatenate(parts), outs


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bitstream", required=True)
    ap.add_argument("--images", required=True,
                    help=".npz with an 'images' array of (N,3,32,32) uint8")
    ap.add_argument("--kernels", required=True, help=".npy of (K,3,3,3) in {-1,+1}")
    ap.add_argument("--kernel-index", type=int, default=0)
    ap.add_argument("--modes", nargs="*", default=list(MODES), choices=list(MODES))
    ap.add_argument("--n-images", type=int, default=2000)
    ap.add_argument("--avg", type=int, default=1)
    ap.add_argument("--freq", type=float, default=5e6)
    ap.add_argument("--gain", type=int, default=40)
    ap.add_argument("--dwell", type=int, default=8)
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--out", required=True, help="output capture directory")
    args = ap.parse_args()

    import chipwhisperer as cw

    _d = np.load(args.images)
    images = _d["images"].astype(np.uint8)[:args.n_images]
    groups = _d["group"][:args.n_images] if "group" in _d else None
    kernels = np.load(args.kernels)
    kern = kernels[args.kernel_index]
    if kern.shape != (3, 3, 3):
        raise SystemExit(f"kernel must be (3,3,3), got {kern.shape}")
    bits27 = (kern.reshape(-1) > 0).astype(np.int32)
    kbytes = pack_kernel_le(bits27)

    samples_per_cycle = 4
    plans = {m: plan_segments(m, args.dwell, samples_per_cycle) for m in args.modes}
    nsamp = max(seg[1] * plans[m][1] for m in args.modes for seg in plans[m][0])

    scope = cw.scope()
    scope.default_setup()
    scope.clock.adc_src = "clkgen_x4"
    scope.clock.clkgen_freq = args.freq
    scope.adc.samples = min(nsamp, MAX_SAMPLES)
    scope.adc.offset = 0
    scope.adc.presamples = 0
    scope.adc.basic_mode = "rising_edge"
    scope.trigger.triggers = "tio4"
    scope.gain.db = args.gain

    target = cw.target(None, cw.targets.CW305, bsfile=args.bitstream,
                       force=True, fpga_id="100t")
    target.clkusbautooff = False
    target.pll.pll_enable_set(True)
    target.pll.pll_outenable_set(False, 0)
    target.pll.pll_outenable_set(True, 1)
    target.pll.pll_outfreq_set(args.freq, 1)
    scope.clock.reset_adc()
    time.sleep(0.5)

    # Refuse to proceed unless THIS design answers. See verify_design_signature.
    identity = verify_design_signature(target)
    print(f"design verified: {identity['signature']}")

    os.makedirs(args.out, exist_ok=True)
    manifest = {
        "kind": "cw305_rgb_leakage_capture",
        "source": "hardware",
        "capture_mode": "live_cw305_chipwhisperer",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bitstream": os.path.abspath(args.bitstream),
        "bitstream_sha256": sha256_file(args.bitstream),
        "design_signature": identity["signature"],
        "design_check": identity,
        "images": os.path.abspath(args.images),
        "images_sha256": sha256_file(args.images),
        "kernels": os.path.abspath(args.kernels),
        "kernel_index": int(args.kernel_index),
        "kernel_bits": bits27.tolist(),
        "input_mode": "rgb8",
        "img_side": IMG_SIDE, "out_side": OUT_SIDE, "n_windows": N_WINDOWS,
        "n_positions": N_POS, "pixel_format": "rgbx_4byte_position_major",
        "trace_source": "cw_lite_analog_sync_x4",
        "samples_per_cycle": samples_per_cycle,
        "dwell": args.dwell,
        "fpga_freq_hz": args.freq,
        "clock_source": "cw_lite",
        "adc_src": "clkgen_x4",
        "gain_db": float(args.gain),
        "avg": int(args.avg),
        "n_images": int(len(images)),
        "groups": (None if groups is None else
                   {g: int((groups == g).sum()) for g in sorted(set(groups.tolist()))}),
        "modes": {},
        "trace_scale": TRACE_SCALE,
        "adc_locked_initial": bool(scope.clock.adc_locked),
    }

    for mode_name in args.modes:
        mode_val = MODES[mode_name]
        segments, per_pos = plans[mode_name]
        mdir = os.path.join(args.out, mode_name)
        os.makedirs(mdir, exist_ok=True)
        print(f"\nmode {mode_name} ({mode_val}): {len(segments)} segment(s), "
              f"{per_pos} samples/position, {N_POS * per_pos} samples/full scan")

        # Functional check on the first image BEFORE any trace is kept. Uses a NATURAL
        # image, not index 0: the first entries are solid-colour diagnostics whose every
        # window is identical, so they would pass a golden comparison while telling you
        # nothing about whether the line buffer is stepping correctly.
        check_i = int(np.argmax(groups == "natural")) if groups is not None else 0
        _, outs = run_once(target, scope, image_to_rgbx(images[check_i]), kbytes, mode_val)
        want = golden_valid_conv_rgb(images[check_i], bits27)
        mism = int(np.count_nonzero(outs != want))
        fc = {"passed": mism == 0, "n_outputs": int(outs.size),
              "hw_min": int(outs.min()), "hw_max": int(outs.max()),
              "golden_min": int(want.min()), "golden_max": int(want.max()),
              "mismatches": mism,
              "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if mism:
            raise RuntimeError(
                f"functional check FAILED in mode {mode_name}: {mism} mismatches. "
                "All three modes must compute identical, golden-matching outputs; a "
                "cross-mode power comparison is meaningless otherwise. Not capturing.")
        print(f"  functional check passed ({outs.size} outputs, 0 mismatches)")

        n_shard, shard_i, buf_t, buf_im, buf_idx = 0, 0, [], [], []
        t0 = time.time()
        for i, img in enumerate(images):
            rgbx = image_to_rgbx(img)
            acc = None
            for _ in range(args.avg):
                tr, _o = capture_segmented(target, scope, rgbx, kbytes, mode_val, segments, per_pos)
                acc = tr if acc is None else acc + tr
            acc /= float(args.avg)
            buf_t.append((acc * TRACE_SCALE).astype(np.int16))
            buf_im.append(img)
            buf_idx.append(i)
            if len(buf_t) >= args.shard_size or i == len(images) - 1:
                np.savez_compressed(
                    os.path.join(mdir, f"shard{shard_i:03d}.npz"),
                    traces=np.stack(buf_t), images=np.stack(buf_im),
                    indices=np.asarray(buf_idx, dtype=np.int32))
                n_shard += len(buf_t)
                shard_i += 1
                buf_t, buf_im, buf_idx = [], [], []
                print(f"  {n_shard}/{len(images)}  {time.time()-t0:.0f}s", end="\r")

        manifest["modes"][mode_name] = {
            "mode_value": mode_val,
            "samples_per_position": per_pos,
            "segments": [{"start": a, "len": b} for a, b in segments],
            "samples_per_full_scan": N_POS * per_pos,
            "functional_check": fc,
            "n_traces": n_shard,
            "dir": mode_name,
        }
        print(f"\n  wrote {n_shard} traces to {mdir}")

    with open(os.path.join(args.out, "capture_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"\nmanifest: {os.path.join(args.out, 'capture_manifest.json')}")


if __name__ == "__main__":
    main()
