#!/usr/bin/env python3
"""Diagnose the CW305 clock path in one shot.

The CW305 has two independent connections to the PC/CW-Lite:
  * its own USB cable, which carries register reads and writes, and
  * the 20-pin ribbon to the CW-Lite, which carries the clock (HS2) and trigger.

They fail independently, and the failure looks confusing: register access keeps
working over USB while the convolution engine never runs, because the design is
built with CLOCK_SETTINGS[2:0] = 101, meaning the crypto clock comes from the
20-pin input and nothing else.

This prints the three facts that separate the cases:
  go_count      increments  -> USB register path is alive
  heartbeat     increments  -> the FPGA crypto clock is running
  extclk        non-zero    -> the FPGA clock is coming back to the CW-Lite
"""
import time

import chipwhisperer as cw

AES_SIGNATURE = (2, 5, 0x2E)


def main():
    scope = cw.scope()
    scope.default_setup()
    scope.io.target_pwr = True
    time.sleep(0.3)
    scope.clock.clkgen_freq = 5e6
    scope.clock.adc_src = "clkgen_x4"
    scope.io.hs2 = "clkgen"
    time.sleep(0.4)
    scope.clock.reset_adc()
    time.sleep(0.8)

    scope.clock.freq_ctr_src = "clkgen"
    time.sleep(0.3)
    gen = int(scope.clock.freq_ctr)
    scope.clock.freq_ctr_src = "extclk"
    time.sleep(0.4)
    ext = int(scope.clock.freq_ctr)

    target = cw.target(None, cw.targets.CW305, bsfile=None, force=False)
    sig = tuple(int(target.fpga_read(p, 1)[0]) for p in (2, 3, 4))

    a = bytes(target.fpga_read(34, 9))
    target.fpga_write(33, [1])
    time.sleep(0.4)
    b = bytes(target.fpga_read(34, 9))

    usb_ok = b[1] != a[1]
    crypto_ok = b[2] != a[2] or b[3] != a[3]

    print(f"design pages 2/3/4 : {[hex(x) for x in sig]}"
          f"  {'STOCK AES' if sig == AES_SIGNATURE else 'custom design ok'}")
    print(f"clkgen (CW-Lite)   : {gen} Hz   {'ok' if gen > 0 else 'NOT GENERATING'}")
    print(f"extclk (from FPGA) : {ext} Hz   {'ok' if ext > 0 else 'NO CLOCK COMING BACK'}")
    print(f"go_count           : {a[1]} -> {b[1]}   "
          f"{'USB register path ok' if usb_ok else 'USB REGISTER PATH DEAD'}")
    print(f"heartbeat/start    : {a[2]}/{a[3]} -> {b[2]}/{b[3]}   "
          f"{'crypto clock running' if crypto_ok else 'CRYPTO CLOCK DEAD'}")

    print()
    if sig == AES_SIGNATURE:
        print("VERDICT: FPGA is serving the stock AES image. Reprogram it "
              "(S1 mode switches M0=M1=M2=1, then power-cycle or USB RST/SW3).")
    elif usb_ok and not crypto_ok and gen > 0:
        print("VERDICT: USB works but no clock reaches the FPGA. The 20-pin ribbon "
              "between the CW-Lite and the CW305 is the only path that carries it. "
              "Reseat both ends of that cable, then run this again.")
    elif crypto_ok:
        print("VERDICT: clock path is healthy. Capture should work.")
    else:
        print("VERDICT: unexpected combination; see the lines above.")

    target.dis()
    scope.dis()


if __name__ == "__main__":
    main()
