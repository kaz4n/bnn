"""Diagnose uploads to this experiment only, with fixed public RGB bytes."""
import json
import chipwhisperer as cw

target = cw.target(None, cw.targets.CW305, fpga_id="100t", slurp=False)
try:
    target.bytecount_size = 16
    signature = bytes(target.fpga_read(0, 8))
    if signature != b"RGBBAY01":
        raise RuntimeError("Expected the already-programmed instrumented RGB design")
    target.fpga_write(1, [2])
    target.fpga_write(2, [5])
    target.fpga_write(1, [1])
    print("initial", bytes(target.fpga_read(3,4)).hex(), flush=True)
    for length in (3, 6, 45, 96, 192):
        target.fpga_write(5, bytes([11,67,199]) * (length // 3))
        print(json.dumps({"write_length":length,
                          "status": bytes(target.fpga_read(3,4)).hex(),
                          "samples_first8": bytes(target.fpga_read(6,8)).hex()}), flush=True)
finally:
    target.dis()
