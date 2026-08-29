#!/usr/bin/env python3
"""
Paper-faithful EXTERNAL ANALOG power capture (CW-Lite scope), high oversampling.

The paper samples at 2.5 GHz (thousands of samples/cycle) and runs the Section-5 analog
cleanup (low-pass, DC restore, Pearson align, RC curve fit, trailing subtraction). We
emulate that on CW-Lite by clocking the BNN slowly (~5 MHz) and sampling the ADC
asynchronously at ~105 MHz -> ~20 samples per conv cycle. The stored raw multi-sample
trace is what attack/run_on_hardware.py --frontend paper consumes.

The conv is an extended-raster line buffer (EW*EW cycles, EW=28+2*(K//2)); `ew` is stored
so the extractor maps the per-cycle power back to the 28x28 output grid.

    python cw305_analog_capture.py --bitstream ..\\build\\cw305_bnn_3.bit \\
        --kernels ..\\training\\artifacts\\model_3x3\\layer1_kernels.npy \\
        --ksize 3 --n-images 300 --n-kernels 9 --avg 20 --out traces_analog
"""
import argparse, json, os, time
import numpy as np

LINE = 28


def pack(k):
    bits = (k.flatten() > 0).astype(np.uint8)
    out = bytearray(); b = n = 0
    for v in bits:
        b = (b << 1) | int(v); n += 1
        if n == 8: out.append(b); b = n = 0
    if n: out.append(b << (8 - n))
    return list(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bitstream", required=True)
    ap.add_argument("--kernels", required=True)
    ap.add_argument("--ksize", type=int, default=3)
    ap.add_argument("--n-images", type=int, default=300)
    ap.add_argument("--n-kernels", type=int, default=9)
    ap.add_argument("--avg", type=int, default=20, help="trace averages (SNR)")
    ap.add_argument("--fpga-freq", type=float, default=5e6, help="slow BNN clock")
    ap.add_argument("--adc-freq", type=float, default=105e6, help="CW-Lite ADC async rate")
    ap.add_argument("--clock-mode", choices=["async", "sync-x4"], default="async",
                    help="async = ADC clockgen at --adc-freq; sync-x4 = ADC locked to CW305 clock x4")
    ap.add_argument("--capture-cycles", choices=["output", "extended"], default="extended",
                    help="output captures the complete physical first-to-last output span; "
                         "extended retains EW*EW clocks. Neither mode assumes 784 "
                         "contiguous clocks because the current raster has row gaps")
    ap.add_argument("--margin-samples", type=int, default=64,
                    help="extra samples after the useful window")
    ap.add_argument("--presamples", type=int, default=0,
                    help="samples to keep before trigger")
    ap.add_argument("--adc-phase", type=int, default=None,
                    help="optional CW-Lite ADC phase setting, if supported")
    ap.add_argument("--gain", type=float, default=45)
    ap.add_argument("--pll-slew", choices=["+0nS", "+1nS", "+2nS", "+3nS"], default=None,
                    help="optional CW305 PLL output slew setting")
    ap.add_argument("--start-method", choices=["reg", "usb-trigger"], default="reg",
                    help="reg writes the start register; usb-trigger can run with USB clock disabled")
    ap.add_argument("--usb-auto-off", action="store_true",
                    help="disable CW305 USB clock during each capture to reduce measurement noise")
    ap.add_argument("--usb-sleep-ms", type=float, default=1.0,
                    help="time to keep USB clock disabled after usb-trigger start")
    ap.add_argument("--images", default="mnist_test.npz")
    ap.add_argument("--out", default="traces_analog")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="recapture files that already exist")
    ap.add_argument("--discard-repeats", action="store_true",
                    help="save only a pointwise mean (not recommended, especially in async mode)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import chipwhisperer as cw
    EW = LINE + 2 * (args.ksize // 2)
    NCONV = EW * EW
    cycle0_offset = (args.ksize - 1) * EW + (args.ksize - 1)
    # Outputs are not contiguous in the current padded raster. From first output to last
    # output the inclusive physical span is 838 clocks for K=3 and 892 for K=5.
    output_span_cycles = (LINE - 1) * (EW + 1) + 1
    useful_cycles = output_span_cycles if args.capture_cycles == "output" else NCONV
    sample_rate = float(args.adc_freq)
    spc = int(round(args.adc_freq / args.fpga_freq))          # ~samples per cycle

    scope = cw.scope(); scope.default_setup()
    if args.clock_mode == "sync-x4":
        scope.clock.adc_src = "extclk_x4"                      # phase-locked to CW305 clock
        sample_rate = float(args.fpga_freq * 4.0)
        spc = 4
    else:
        scope.clock.adc_src = "clkgen_x1"                      # ADC async to target
        scope.clock.clkgen_freq = args.adc_freq
    nsamp = min(24000, int(spc * useful_cycles + args.margin_samples))
    scope.adc.samples = nsamp
    scope.adc.offset = 0
    scope.adc.presamples = args.presamples
    scope.trigger.triggers = "tio4"
    scope.gain.db = args.gain
    scope.adc.basic_mode = "rising_edge"
    if args.adc_phase is not None and hasattr(scope.clock, "adc_phase"):
        scope.clock.adc_phase = args.adc_phase

    target = cw.target(scope, cw.targets.CW305, bsfile=args.bitstream,
                       fpga_id="100t", force=True)
    target.pll.pll_enable_set(True)
    target.pll.pll_outenable_set(True, 1)
    target.pll.pll_outfreq_set(args.fpga_freq, 1)             # FPGA slow, ADC independent
    if args.pll_slew is not None:
        target.pll.pll_outslew_set(args.pll_slew, 1)
    if args.usb_auto_off:
        if args.start_method != "usb-trigger":
            raise ValueError("--usb-auto-off requires --start-method usb-trigger")
        target.clkusbautooff = False                           # manual: custom BNN start path
    scope.clock.reset_adc()

    d = np.load(args.images); xte, yte = d["images"], d["labels"]
    if xte.ndim == 4 and xte.shape[-1] == 1:
        xte = xte[..., 0]
    if xte.ndim != 3 or xte.shape[1:] != (LINE, LINE):
        raise ValueError(
            f"current CW305 BNN bitstream expects images shaped (N,{LINE},{LINE}); "
            f"got {xte.shape}. CIFAR-10 maps/parameters need separate 32x32x3 or "
            "multi-channel gateware."
        )
    kernels_all = np.load(args.kernels)
    if kernels_all.ndim != 3 or kernels_all.shape[1:] != (args.ksize, args.ksize):
        raise ValueError(
            f"current CW305 BNN bitstream expects kernels shaped (N,{args.ksize},{args.ksize}); "
            f"got {kernels_all.shape}. CIFAR-10 conv parameters such as 3x3x3x128 "
            "do not match this 1-channel target."
        )
    kernels = kernels_all[:args.n_kernels]
    packed_kernels = [pack(kernels[kid]) for kid in range(args.n_kernels)]
    manifest = {
        "bitstream": os.path.abspath(args.bitstream),
        "kernels": os.path.abspath(args.kernels),
        "images": os.path.abspath(args.images),
        "ksize": args.ksize,
        "n_kernels": args.n_kernels,
        "avg": args.avg,
        "fpga_freq_hz": args.fpga_freq,
        "clock_mode": args.clock_mode,
        "capture_cycles": args.capture_cycles,
        "sample_rate_hz": sample_rate,
        "samples_per_cycle_nominal": spc,
        "samples": nsamp,
        "useful_cycles": useful_cycles,
        "presamples": args.presamples,
        "adc_phase": args.adc_phase,
        "cycle0_offset": cycle0_offset,
        "trigger_source": "first_output_write",
        "start_method": args.start_method,
        "usb_auto_off": bool(args.usb_auto_off),
        "usb_sleep_ms": args.usb_sleep_ms,
        "pll_slew": args.pll_slew,
        "trace_source": "analog",
        "raw_repeats_saved": not args.discard_repeats,
        "status": "incomplete",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(args.out, "capture_manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=2)
    print(f"[analog] EW={EW} cycles={useful_cycles} spc~{spc} samples={nsamp} "
          f"avg={args.avg} clock={args.clock_mode} start={args.start_method}")

    for n in range(args.start_index, args.n_images):
        outp = os.path.join(args.out, f"img{n:04d}.npz")
        if os.path.exists(outp) and not args.force:
            continue
        img = xte[n].astype(np.uint8)
        target.fpga_write(0, img.flatten().tolist())
        traces = np.zeros((args.n_kernels, nsamp), np.float32)
        repeats = None if args.discard_repeats else np.zeros(
            (args.n_kernels, args.avg, nsamp), np.float32)
        for kid in range(args.n_kernels):
            target.fpga_write(32, packed_kernels[kid])
            acc = None
            for rep in range(args.avg):
                scope.arm()
                if args.start_method == "usb-trigger":
                    if args.usb_auto_off:
                        target.usb_clk_setenabled(False)
                    target.usb_trigger_toggle()
                    if args.usb_auto_off:
                        time.sleep(args.usb_sleep_ms / 1000.0)
                        target.usb_clk_setenabled(True)
                else:
                    target.fpga_write(33, [1])
                if scope.capture():
                    if args.usb_auto_off:
                        target.usb_clk_setenabled(True)
                    raise RuntimeError("capture timeout -- check trigger / clock")
                w = scope.get_last_trace()
                if repeats is not None:
                    repeats[kid, rep] = np.asarray(w, dtype=np.float32)
                acc = w if acc is None else acc + w
            traces[kid] = acc / args.avg
        payload = dict(traces=traces, kernel_ids=np.arange(args.n_kernels),
                       image=img, label=int(yte[n]), samples_per_cycle=spc,
                       n_cycles=useful_cycles, ew=EW, trace_source="analog",
                       cycle0_offset=int(cycle0_offset),
                       trigger_source="first_output_write",
                       clock_mode=str(args.clock_mode),
                       capture_cycles=str(args.capture_cycles),
                       sample_rate_hz=float(sample_rate),
                       fpga_freq_hz=float(args.fpga_freq),
                       avg=int(args.avg), gain_db=float(args.gain),
                       raw_repeats_saved=bool(repeats is not None),
                       repetitions_prealigned=False,
                       start_method=str(args.start_method),
                       usb_auto_off=bool(args.usb_auto_off),
                       presamples=int(args.presamples),
                       adc_phase=-999999 if args.adc_phase is None else int(args.adc_phase),
                       pll_slew="" if args.pll_slew is None else str(args.pll_slew))
        if repeats is not None:
            payload["repeats"] = repeats
        np.savez(outp, **payload)
        if n % 20 == 0:
            print(f"captured {n+1}/{args.n_images}", flush=True)
    manifest["status"] = "complete"
    manifest["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(args.out, "capture_manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=2)
    scope.dis(); target.dis()


if __name__ == "__main__":
    main()
