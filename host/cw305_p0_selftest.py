#!/usr/bin/env python3
"""
P0 bench self-test -- run in the ChipWhisperer Windows env (cwenv), NOT the project env.
No Jupyter needed.

    cd C:\\Users\\narut\\ChipWhisperer
    call cwenv\\Scripts\\activate.bat
    python C:\\Users\\narut\\OneDrive\\Desktop\\Project\\bnn\\host\\cw305_p0_selftest.py

Proves the whole capture path with the STOCK CW305 AES bitstream (ships with
ChipWhisperer): scope connects, CW305 programs, PLL locks, trigger fires, a real power
trace is captured. If this works, the bench is good and only the BNN bitstream (P1) is
left. If it fails, the exact failing step is printed -- send me that line.
"""
import sys
import numpy as np

try:
    import chipwhisperer as cw
except Exception as e:
    print("FAIL [import]:", e); sys.exit(1)

print("[1] connecting scope (CW-Lite) ...")
scope = cw.scope()
print("    OK:", scope._getNAEUSB().snum if hasattr(scope, "_getNAEUSB") else scope)

print("[2] connecting CW305 + programming stock AES bitstream ...")
# fpga_id '100t' for CW305-A100; use '35t' if you have the A35 board.
target = cw.target(scope, cw.targets.CW305, fpga_id="100t", force=True)
print("    OK: CW305 programmed")

print("[3] clock + ADC setup (synchronous capture) ...")
scope.gain.db = 25
scope.adc.samples = 5000
scope.adc.offset = 0
scope.clock.adc_src = "extclk_x4"          # sample 4x, locked to CW305 clock
target.pll.pll_enable_set(True)
target.pll.pll_outenable_set(False, 0)
target.pll.pll_outenable_set(True, 1)      # PLL1 -> FPGA
target.pll.pll_outenable_set(False, 2)
target.pll.pll_outfreq_set(10E6, 1)        # 10 MHz target clock
scope.clock.reset_adc()
import time; time.sleep(0.5)
print("    ADC locked:", scope.clock.adc_locked)
scope.trigger.triggers = "tio4"
scope.adc.basic_mode = "rising_edge"

print("[4] capturing one AES power trace ...")
key = bytearray(range(16)); text = bytearray(range(16))
target.fpga_write(target.REG_CRYPT_KEY, key[::-1])
target.fpga_write(target.REG_CRYPT_TEXTIN, text[::-1])
scope.arm()
target.fpga_write(target.REG_CRYPT_GO, [1])
if scope.capture():
    print("FAIL [capture]: trigger timeout -- check 20-pin ribbon (TIO4 trigger) "
          "and that the AES bitstream asserts trigger"); sys.exit(1)
wave = scope.get_last_trace()
cout = target.fpga_read(target.REG_CRYPT_CIPHEROUT, 16)

print(f"    captured {len(wave)} samples, range {wave.min():.4f}..{wave.max():.4f}")
print(f"    AES ciphertext: {bytes(cout).hex()}")
np.save("p0_trace.npy", wave)
print("\nPASS: bench works. Saved p0_trace.npy. Next: build the BNN bitstream (P1).")
scope.dis(); target.dis()
