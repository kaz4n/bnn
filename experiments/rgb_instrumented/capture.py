"""Capture intentional RGB sampling on CW305; physical acquisition is default.

This does not acquire power, EM or timing leakage. The FPGA deliberately
selects one RGB component per pixel, stores it, and exposes it for readback.
There is no automatic fallback to generated observations on a hardware error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from .data import cases, expected_mosaic


class HardwareCapture:
    def __init__(self, bitstream: Path):
        import chipwhisperer as cw
        if not bitstream.is_file():
            raise FileNotFoundError(bitstream)
        self.target = None
        self.bitstream_sha256 = hashlib.sha256(bitstream.read_bytes()).hexdigest()
        self.target = cw.target(None, cw.targets.CW305, bsfile=str(bitstream.resolve()),
                                force=True, fpga_id="100t", slurp=False)
        try:
            self.target.bytecount_size = 16
            if not self.target.fpga.isFPGAProgrammed():
                raise RuntimeError("FPGA programming did not complete")
            signature = bytes(self.target.fpga_read(0, 8))
            if signature != b"RGBBAY01":
                raise RuntimeError(f"Wrong FPGA protocol signature: {signature!r}")
            self.metadata = {
                "board": "CW305", "fpga_part": "xc7a100tftg256-2",
                "bitstream_sha256": self.bitstream_sha256,
                "protocol_signature": signature.decode("ascii"),
                "chipwhisperer_version": cw.__version__,
                "usb_transport": "control_transfers_write45_read32",
                "acquisition_kind": "intentional_digital_RGGB_sampling_in_FPGA",
                "external_adc_used": False,
                "capture_clock": "CW305 USB clock; single clock domain",
            }
        except Exception:
            self.close()
            raise

    def capture(self, rgb):
        target = self.target
        width = rgb.shape[0]
        if rgb.shape != (width, width, 3) or width not in (32,64,128) or rgb.dtype != np.uint8:
            raise ValueError("Expected supported square RGB8 image")
        begin = time.perf_counter()
        target.fpga_write(1, [2])
        target.fpga_write(2, [width.bit_length() - 1])
        target.fpga_write(1, [1])
        initial = bytes(target.fpga_read(3, 4))
        if len(initial) != 4 or initial != bytes([1,0,0,0]):
            raise RuntimeError(f"FPGA did not begin an empty frame: {initial.hex()}")
        raw = rgb.tobytes(order="C")
        # Chunks may reset the USB byte offset; RTL maintains RGB byte phase.
        # On this board/SDK combination, bulk writes completed without reaching
        # the FPGA; the saved transport probe shows control writes work. Keep
        # writes below the SDK's control/bulk threshold and record this choice.
        for offset in range(0, len(raw), 45):
            target.fpga_write(5, raw[offset:offset+45])
        deadline = time.monotonic() + 5
        while True:
            status = bytes(target.fpga_read(3, 4))
            if len(status) != 4:
                raise RuntimeError("Short hardware status record")
            if status[0] & 4:
                raise RuntimeError(f"FPGA reported protocol error: {status.hex()}")
            if status[0] & 2:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"FPGA frame incomplete: {status.hex()}")
            time.sleep(.01)
        count = status[1] | (status[2] << 8)
        if status[0] != 2 or count != width * width or status[3] != 0:
            raise RuntimeError(f"Invalid completed frame: {status.hex()}")
        payload = b"".join(bytes(target._naeusb.cmdReadMem((6 << 16) + offset,
                                                          min(32, count - offset)))
                           for offset in range(0, count, 32))
        if len(payload) != count:
            raise RuntimeError(f"Short capture: {len(payload)} of {count}")
        mosaic = np.frombuffer(payload, dtype=np.uint8).reshape(width, width).copy()
        return mosaic, {
            **self.metadata, "status_hex": status.hex(), "captured_sample_count": count,
            "acquisition_seconds": time.perf_counter() - begin,
            "acquisition_time_includes": "host USB upload, FPGA sampling, status and readback",
        }

    def close(self):
        if self.target is not None:
            self.target.dis()
            self.target = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("fpga", "software_model"), default="fpga")
    parser.add_argument("--bitstream", type=Path,
                        default=Path(__file__).parent / "build/cw305_rgb_instrumented.bit")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", choices=(32,64,128), default=[32,64,128])
    parser.add_argument("--smoke", action="store_true", help="Capture only diagnostic gradient")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if root.exists():
        raise FileExistsError(f"Use a new run directory to preserve evidence: {root}")
    # Acquire/download input images before opening or programming the FPGA.
    inputs = list(cases(args.sizes, smoke=args.smoke))
    root.mkdir(parents=True)
    (root / "captures").mkdir()
    (root / "truth").mkdir()
    manifest = {"source": args.source, "pattern": "RGGB", "status": "running",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "python_version": platform.python_version(), "frames": [],
                "scope": "intentional FPGA imaging measurements; not physical side-channel recovery"}
    manifest_path = root / "capture_manifest.json"
    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    save_manifest()
    device = None
    try:
        if args.source == "fpga":
            device = HardwareCapture(args.bitstream)
        for case_id, rgb, info in inputs:
            input_hash = hashlib.sha256(rgb.tobytes(order="C")).hexdigest()
            Image.fromarray(rgb).save(root / "truth" / f"{case_id}.png")
            expected = expected_mosaic(rgb)
            if device is not None:
                mosaic, acquisition = device.capture(rgb)
            else:
                mosaic = expected.copy()
                acquisition = {"acquisition_kind": "software_model_only",
                               "captured_sample_count": mosaic.size, "acquisition_seconds": 0.0}
            mismatches = int(np.count_nonzero(mosaic != expected))
            provenance = {**info, **acquisition, "source": args.source, "pattern": "RGGB",
                          "case_id": case_id, "width": rgb.shape[0], "height": rgb.shape[1],
                          "input_sha256": input_hash,
                          "capture_sha256": hashlib.sha256(mosaic.tobytes()).hexdigest(),
                          "hardware_validation_passed": bool(device is not None and mismatches == 0),
                          "reference_mismatch_count": mismatches,
                          "captured_utc": datetime.now(timezone.utc).isoformat()}
            np.savez_compressed(root / "captures" / f"{case_id}.npz", mosaic=mosaic,
                                provenance_json=json.dumps(provenance, allow_nan=False))
            manifest["frames"].append(provenance)
            save_manifest()
            if mismatches:
                raise RuntimeError(f"{case_id}: {mismatches} hardware/reference mismatches; saved for diagnosis")
            print(json.dumps({"case": case_id, "source": args.source, "samples": mosaic.size,
                              "reference_mismatches": mismatches}), flush=True)
        manifest["status"] = "complete"
        manifest["total_samples"] = sum(x["captured_sample_count"] for x in manifest["frames"])
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if device is not None:
            device.close()
        save_manifest()


if __name__ == "__main__":
    main()
