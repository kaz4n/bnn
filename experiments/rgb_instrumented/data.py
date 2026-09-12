"""Small public RGB images and explicit diagnostic patterns for this benchmark."""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ASSETS = Path(__file__).resolve().parent / "assets"
BASE = "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.18.3/skimage/data/"
NATURAL = ("astronaut", "coffee", "chelsea", "rocket")


def prepare_sources() -> dict[str, Path]:
    """Download only four public example files, using a pinned upstream version."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name in NATURAL:
        path = ASSETS / (f"{name}.jpg" if name == "rocket" else f"{name}.png")
        if not path.exists():
            with urllib.request.urlopen(BASE + path.name, timeout=30) as response:
                payload = response.read()
            temporary = path.with_suffix(".download")
            temporary.write_bytes(payload)
            with Image.open(temporary) as image:
                image.verify()
            temporary.replace(path)
        with Image.open(path) as image:
            image.verify()
        sources[name] = path
    return sources


def patterns(size: int):
    yy, xx = np.indices((size, size))
    ramp = np.linspace(0, 255, size).round().astype(np.uint8)
    gradient = np.stack((np.broadcast_to(ramp, (size, size)),
                         np.broadcast_to(ramp[:, None], (size, size)),
                         np.rint((xx + yy) * 255 / (2 * (size - 1))).astype(np.uint8)), axis=-1)
    palette = np.array([[255,0,0],[0,255,0],[0,0,255],[255,255,0],
                        [0,255,255],[255,0,255],[255,255,255],[0,0,0]], dtype=np.uint8)
    bars = palette[np.minimum(xx * 8 // size, 7)]
    # Deliberately exceeds Bayer color sampling bandwidth: a failure control.
    nyquist = np.stack(((xx % 2) * 255, (yy % 2) * 255,
                        ((xx + yy) % 2) * 255), axis=-1).astype(np.uint8)
    return {"gradient": gradient, "colorbars": bars, "nyquist": nyquist}


def cases(sizes=(32, 64, 128), smoke=False):
    sources = {} if smoke else prepare_sources()
    for size in sizes:
        if size not in (32, 64, 128):
            raise ValueError("FPGA supports square 32, 64 or 128 only")
        for name, path in sources.items():
            with Image.open(path) as image:
                resized = ImageOps.fit(image.convert("RGB"), (size, size),
                                       method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
                rgb = np.array(resized)
            yield f"natural_{name}_{size}", rgb, {
                "image_source_url": BASE + path.name,
                "source_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "resize": "center square crop and Pillow Lanczos to target size",
                "color_space": "source encoded RGB8; no linear-light conversion",
            }
        for name, rgb in patterns(size).items():
            if smoke and name != "gradient":
                continue
            yield f"control_{name}_{size}", rgb, {
                "image_source_url": f"procedural:{name}:v1",
                "color_space": "encoded RGB8 numeric test pattern",
            }


def expected_mosaic(rgb):
    """Reference for validation/model-only runs; never substituted for FPGA data."""
    result = rgb[..., 1].copy()
    result[0::2, 0::2] = rgb[0::2, 0::2, 0]
    result[1::2, 1::2] = rgb[1::2, 1::2, 2]
    return result
