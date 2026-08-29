# P0 bench self-test — RESULT: PASS ✅

Proof the ChipWhisperer-Lite + CW305 capture chain works end-to-end, using the stock
NewAE AES bitstream (before building our own BNN bitstream).

- **Date:** 2026-06-23
- **Script:** [`cw305_p0_selftest.py`](cw305_p0_selftest.py)
- **Run env:** native Windows `cwenv` (`C:\Users\narut\ChipWhisperer\cwenv`)
- **Hardware:** CW-Lite `2b3e:ace2` + CW305-A100 `2b3e:c305`

## Console output

```
(ChipWhisperer NAEUSB ERROR|File naeusb.py:828) Your firmware (0.51.0) is outdated - latest is 0.54.0
[1] connecting scope (CW-Lite) ...
    OK: 50203220415447303030313238323035
[2] connecting CW305 + programming stock AES bitstream ...
    OK: CW305 programmed
[3] clock + ADC setup (synchronous capture) ...
    ADC locked: True
[4] capturing one AES power trace ...
    captured 5000 samples, range -0.0674..0.0215
    AES ciphertext: 5aea53c65894c3f145f06e41b50b940a

PASS: bench works. Saved p0_trace.npy. Next: build the BNN bitstream (P1).
```

## What each line proves

| Step | Proves |
|---|---|
| `[1] OK <serial>` | CW-Lite enumerated + USB/WinUSB communication |
| `[2] CW305 programmed` | CW305 USB link + FPGA bitstream programming |
| `[3] ADC locked: True` | CW305 PLL clock reaches CW-Lite → **synchronous** capture (the basis for per-cycle power, replaces paper §5) |
| `[4] 5000 samples, range -0.067..0.021` | trigger (TIO4) fired + real power measured via X4→MEASURE |
| `AES ciphertext: 5aea…940a` | full register read/write + the FPGA actually computed |

## Captured trace

![CW305 AES power trace](p0_trace.png)

5000 samples, synchronous (4× clock), AC-coupled. min −0.0674 V, max 0.0215 V, std 0.0083 V.
Periodic structure = AES round activity. Raw data: `p0_trace.npy`.

## Conclusion

Capture path verified: USB · FPGA programming · synchronous clock · trigger · power
measurement · register I/O all work. The only remaining piece for the real attack is the
**BNN bitstream (P1)** — same capture chain, our conv design instead of stock AES.

> Note: firmware 0.51 vs 0.54 = warning only, harmless. Stuck-handle gotcha: killing a
> Jupyter kernel that held the scope leaves a stuck WinUSB handle → replug CW-Lite to clear.
