"""Evaluate reconstruction of explicit RGGB exports without using truth as input.

Run from the repository root with::

    python -m experiments.rgb_instrumented.evaluate --run-dir RUN_DIR

Expected pairs are ``captures/ID.npz`` and ``truth/ID.png``. Each NPZ contains
uint8 ``mosaic`` and scalar UTF-8 ``provenance_json``. Provenance must specify
``source`` (fpga or software_model), ``pattern=RGGB``, and ``input_sha256`` of
the RGB8 truth's C-order pixel bytes. FPGA provenance additionally requires
``hardware_validation_passed=true``. IDs start with natural_ or control_.

Truth is used only for validation, scoring, and display. ``reconstruct`` sees
only a copied, normalized mosaic and fixed method parameters. Metrics describe
floating reconstructions before PNG rounding and use a fixed [0, 1] range.
Failures are written to metrics.json/results.md and cause a nonzero CLI exit.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if __package__:
    from .reconstruction import METHODS, reconstruct
else:
    from reconstruction import METHODS, reconstruct


METHOD_PARAMETERS = {
    "nearest": {},
    "bilinear": {},
    "smooth_ridge": {"ridge": 0.05, "rtol": 1e-7, "atol": 1e-10, "maxiter": 1000},
    "tv": {"weight": 0.05, "rtol": 1e-5, "atol": 1e-8, "maxiter": 2000, "check_every": 10},
}

METRIC_DEFINITIONS = {
    "rgb_mse": "Mean squared error across all H*W*3 normalized RGB components, before PNG rounding.",
    "rgb_psnr_db": "10*log10(1/rgb_mse), with fixed peak=1; null and is_infinite=true for zero MSE.",
    "channel_mae_rgb": "Mean absolute normalized error separately in R, G, B order.",
    "missing_mse": "Mean squared error over the 2*H*W unmeasured RGB components only.",
    "measurement_max_abs_error": "Maximum absolute normalized difference at the H*W measured components.",
    "pooled_psnr": "PSNR computed from total squared error divided by total component count.",
    "mean_per_image_psnr": "Arithmetic mean of per-image PSNR; infinite if any image has zero MSE.",
    "runtime_seconds": "Wall-clock reconstruction call only; excludes input/output, validation, and metrics.",
}


def _masks(shape):
    mask = np.zeros((*shape, 3), dtype=bool)
    mask[::2, ::2, 0] = True
    mask[1::2, 1::2, 2] = True
    mask[..., 1] = ~(mask[..., 0] | mask[..., 2])
    return mask


def _psnr(mse):
    return (None, True) if mse == 0 else (float(-10 * np.log10(mse)), False)


def calculate_metrics(output, truth, mosaic):
    """Score a reconstruction; the reference is never passed to reconstruct."""
    if output.shape != (*mosaic.shape, 3) or not np.issubdtype(output.dtype, np.floating):
        raise ValueError("reconstruction must be floating RGB with the capture's dimensions")
    if not np.all(np.isfinite(output)) or np.any(output < 0) or np.any(output > 1):
        raise ValueError("reconstruction contains nonfinite or out-of-range values")
    mask = _masks(mosaic.shape)
    measured = np.broadcast_to(mosaic[..., None], output.shape)[mask]
    measurement_error = float(np.max(np.abs(output[mask] - measured)))
    if measurement_error != 0:
        raise ValueError(f"reconstruction changed measured samples (maximum error {measurement_error})")
    difference = output - truth
    squared = difference * difference
    rgb_sum, missing_sum = float(squared.sum()), float(squared[~mask].sum())
    rgb_count, missing_count = squared.size, int(np.count_nonzero(~mask))
    rgb_mse, missing_mse = rgb_sum / rgb_count, missing_sum / missing_count
    rgb_psnr, rgb_infinite = _psnr(rgb_mse)
    missing_psnr, missing_infinite = _psnr(missing_mse)
    return {
        "rgb_mse": rgb_mse,
        "rgb_psnr_db": rgb_psnr,
        "rgb_psnr_is_infinite": rgb_infinite,
        "channel_mae_rgb": np.mean(np.abs(difference), axis=(0, 1)).tolist(),
        "missing_mse": missing_mse,
        "missing_psnr_db": missing_psnr,
        "missing_psnr_is_infinite": missing_infinite,
        "measurement_max_abs_error": measurement_error,
        "rgb_squared_error_sum": rgb_sum,
        "rgb_component_count": rgb_count,
        "missing_squared_error_sum": missing_sum,
        "missing_component_count": missing_count,
    }


def _reject_json_constant(value):
    raise ValueError(f"nonfinite JSON value: {value}")


def _load_capture(path):
    with np.load(path, allow_pickle=False) as capture:
        if not {"mosaic", "provenance_json"}.issubset(capture.files):
            raise ValueError("capture needs mosaic and provenance_json arrays")
        raw_provenance = capture["provenance_json"]
        if raw_provenance.shape != () or raw_provenance.dtype.kind not in "US":
            raise ValueError("provenance_json must be a scalar string")
        raw_provenance = raw_provenance.item()
        if isinstance(raw_provenance, bytes):
            raw_provenance = raw_provenance.decode("utf-8")
        provenance = json.loads(raw_provenance, parse_constant=_reject_json_constant)
        if not isinstance(provenance, dict):
            raise ValueError("provenance_json must contain an object")
        return capture["mosaic"].copy(), provenance


def _validate_case(mosaic, provenance, truth_path, case_id):
    if mosaic.dtype != np.uint8 or mosaic.ndim != 2 or min(mosaic.shape) < 2:
        raise ValueError("capture mosaic must be uint8 (H, W) with both dimensions >= 2")
    if provenance.get("source") not in ("fpga", "software_model"):
        raise ValueError("provenance source must explicitly be fpga or software_model")
    if provenance.get("pattern") != "RGGB":
        raise ValueError("provenance pattern must be RGGB")
    if provenance["source"] == "fpga" and provenance.get("hardware_validation_passed") is not True:
        raise ValueError("FPGA capture lacks hardware_validation_passed=true")
    if not case_id.startswith(("natural_", "control_")):
        raise ValueError("case ID must start with natural_ or control_ for separate summaries")
    with Image.open(truth_path) as image:
        if image.format != "PNG" or image.mode != "RGB":
            raise ValueError("truth must be an RGB8 PNG without implicit mode conversion")
        truth = np.array(image, dtype=np.uint8)
    if truth.shape != (*mosaic.shape, 3):
        raise ValueError(f"truth dimensions {truth.shape} do not match capture {mosaic.shape}")
    for name, actual in (("height", mosaic.shape[0]), ("width", mosaic.shape[1])):
        if name in provenance and provenance[name] != actual:
            raise ValueError(f"provenance {name} does not match capture dimensions")
    digest = hashlib.sha256(truth.tobytes(order="C")).hexdigest()
    if provenance.get("input_sha256") != digest:
        raise ValueError("input_sha256 does not match raw RGB8 truth pixel bytes")
    expected = truth[..., 1].copy()
    expected[::2, ::2] = truth[::2, ::2, 0]
    expected[1::2, 1::2] = truth[1::2, 1::2, 2]
    if not np.array_equal(mosaic, expected):
        raise ValueError("capture samples do not match the declared RGGB truth samples")
    return truth


def aggregate_results(results):
    """Keep sources, resolutions, and natural/control categories separate."""
    groups = defaultdict(list)
    for result in results:
        key = (result["source"], result["category"], result["resolution"], result["method"])
        groups[key].append(result)
    summaries = []
    for (source, category, resolution, method), records in sorted(groups.items()):
        successful = [record for record in records if record["status"] == "ok"]
        summary = {
            "source": source, "category": category, "resolution": resolution,
            "method": method, "successful_images": len(successful),
            "failed_images": len(records) - len(successful),
        }
        for prefix in ("rgb", "missing"):
            if not successful:
                summary.update({f"pooled_{prefix}_mse": None, f"pooled_{prefix}_psnr_db": None,
                                f"pooled_{prefix}_psnr_is_infinite": False,
                                f"mean_per_image_{prefix}_psnr_db": None,
                                f"mean_per_image_{prefix}_psnr_is_infinite": False})
                continue
            squared_sum = sum(record["metrics"][f"{prefix}_squared_error_sum"] for record in successful)
            count = sum(record["metrics"][f"{prefix}_component_count"] for record in successful)
            mse = squared_sum / count
            psnr, infinite = _psnr(mse)
            mean_infinite = any(record["metrics"][f"{prefix}_psnr_is_infinite"] for record in successful)
            mean_psnr = None if mean_infinite else float(np.mean([record["metrics"][f"{prefix}_psnr_db"] for record in successful]))
            summary.update({f"pooled_{prefix}_mse": mse, f"pooled_{prefix}_psnr_db": psnr,
                            f"pooled_{prefix}_psnr_is_infinite": infinite,
                            f"mean_per_image_{prefix}_psnr_db": mean_psnr,
                            f"mean_per_image_{prefix}_psnr_is_infinite": mean_infinite})
        summary["mean_runtime_seconds"] = float(np.mean([record["runtime_seconds"] for record in successful])) if successful else None
        summaries.append(summary)
    return summaries


def _score(value, infinite=False):
    if infinite:
        return "inf"
    return "n/a" if value is None else f"{value:.3f}"


def _markdown(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _write_report(run_dir, report):
    source_label = ", ".join(report["sources"]) or "unavailable (validation failed)"
    lines = [
        "# Instrumented RGGB reconstruction evaluation", "",
        f"**Recorded acquisition sources: {source_label}. Status: {report['status']}.**", "",
        "`software_model` denotes software-generated samples. `fpga` denotes records with the required hardware-validation flag; full supplied provenance is retained in metrics.json.", "",
        "Inputs are explicit uint8 Bayer exports. Reconstruction receives only normalized mosaic samples and the fixed parameters listed below. Truth is used for validation, metrics, and display.", "",
        "Scores use floating RGB in [0,1] before exported PNG rounding. Missing-channel scores exclude measured samples. Pooled PSNR is computed from pooled squared error; mean per-image PSNR is a separate arithmetic mean. Perfect reconstruction has infinite PSNR (JSON null with an explicit is_infinite flag).", "",
        "Natural images and synthetic controls are summarized separately. These scenes are a small evaluation set and do not establish statistical generalization. Runtime covers reconstruction only.", "",
        f"Cases: {len(report['cases'])}; successful reconstructions: {sum(r['status'] == 'ok' for r in report['results'])}; recorded failures: {len(report['failures'])}.", "",
        "## Fixed method parameters", "", "```json", json.dumps(METHOD_PARAMETERS, indent=2, allow_nan=False), "```", "",
        "`smooth_ridge` is graph-Laplacian inpainting with a ridge toward bilinear RGB. `tv` minimizes quadratic distance to bilinear RGB plus spatial isotropic TV. Both retain hard measured constraints. Neither parameter set is tuned using these evaluation images.", "",
        "## Summaries by source, scene category, and resolution", "",
        "| Source | Category | Resolution | Method | Passed / failed | Pooled RGB MSE | Pooled RGB PSNR dB | Mean image RGB PSNR dB | Pooled missing PSNR dB | Mean seconds |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["summaries"]:
        mse = "n/a" if row["pooled_rgb_mse"] is None else f"{row['pooled_rgb_mse']:.7g}"
        values = [row["source"], row["category"], row["resolution"], row["method"], f"{row['successful_images']} / {row['failed_images']}", mse,
                  _score(row["pooled_rgb_psnr_db"], row["pooled_rgb_psnr_is_infinite"]),
                  _score(row["mean_per_image_rgb_psnr_db"], row["mean_per_image_rgb_psnr_is_infinite"]),
                  _score(row["pooled_missing_psnr_db"], row["pooled_missing_psnr_is_infinite"]), _score(row["mean_runtime_seconds"])]
        lines.append("| " + " | ".join(map(_markdown, values)) + " |")
    lines += ["", "## Per-image results", "", "| Case | Source | Method | RGB PSNR dB | Missing PSNR dB | R/G/B MAE | Measured max error | Seconds | Status |", "|---|---|---|---:|---:|---|---:|---:|---|"]
    for row in report["results"]:
        if row["status"] == "ok":
            metric = row["metrics"]
            fields = [_score(metric["rgb_psnr_db"], metric["rgb_psnr_is_infinite"]), _score(metric["missing_psnr_db"], metric["missing_psnr_is_infinite"]), "/".join(f"{v:.5f}" for v in metric["channel_mae_rgb"]), f"{metric['measurement_max_abs_error']:.3g}"]
        else:
            fields = ["n/a"] * 4
        values = [row["id"], row["source"], row["method"], *fields, _score(row["runtime_seconds"]), row["status"]]
        lines.append("| " + " | ".join(map(_markdown, values)) + " |")
    if report["failures"]:
        lines += ["", "## Failures", ""]
        for error in report["failures"]:
            lines.append(f"- {_markdown(error['id'])}, source={_markdown(error.get('source'))}, {_markdown(error['stage'])}: {_markdown(error['error'])}")
    lines += ["", "## Comparison", "", "The montage includes every successfully validated case at the largest image area present, with original RGB, grayscale Bayer samples, and each method. Captions report RGB PSNR; failed methods remain visibly marked.", ""]
    lines += ["![Reconstruction comparison](comparison.png)" if (run_dir / "comparison.png").is_file() else "The comparison image could not be generated; see the recorded failure.", ""]
    (run_dir / "results.md").write_text("\n".join(lines), encoding="utf-8")


def _font(size):
    for name in ("DejaVuSans.ttf", "C:/Windows/Fonts/segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _write_montage(run_dir, report):
    valid = [case for case in report["cases"] if case["status"] == "ok"]
    largest_area = max((case["height"] * case["width"] for case in valid), default=0)
    selected = [case for case in valid if case["height"] * case["width"] == largest_area]
    report["montage_case_ids"] = [case["id"] for case in selected]
    tile, gap, margin, top, row_height = 192, 16, 24, 72, 276
    columns = len(METHODS) + 2
    canvas = Image.new("RGB", (2 * margin + columns * tile + (columns - 1) * gap, top + max(len(selected), 1) * row_height), "#f5f5f2")
    draw, font, small = ImageDraw.Draw(canvas), _font(20), _font(14)
    sources = ", ".join(report["sources"]) or "unavailable"
    draw.text((margin, 12), f"Instrumented RGGB reconstruction | source: {sources}", fill="#181818", font=font)
    draw.text((margin, 42), "Largest-resolution cases | RGB PSNR before PNG rounding | natural images and controls reported separately", fill="#454545", font=small)
    results = {(row["id"], row["method"]): row for row in report["results"]}
    for row_index, case in enumerate(selected):
        case_id = case["id"]
        y = top + row_index * row_height
        draw.text((margin, y), f"{case_id} | {case['category']} | {case['resolution']} | source={case['source']}", fill="#181818", font=small)
        with Image.open(run_dir / "truth" / f"{case_id}.png") as truth_image:
            panels = [("Original RGB", truth_image.copy(), "reference")]
        mosaic, _ = _load_capture(run_dir / "captures" / f"{case_id}.npz")
        panels.append(("RGGB samples", Image.fromarray(mosaic).convert("RGB"), "grayscale sample values"))
        for method in METHODS:
            result = results[(case_id, method)]
            if result["status"] == "ok":
                with Image.open(run_dir / result["output_png"]) as output_image:
                    panel = output_image.copy()
                metric = result["metrics"]
                caption = "PSNR " + _score(metric["rgb_psnr_db"], metric["rgb_psnr_is_infinite"]) + " dB"
            else:
                panel, caption = Image.new("RGB", (tile, tile), "#e4dddd"), "FAILED - see results.md"
            panels.append((method, panel, caption))
        for column, (title, panel, caption) in enumerate(panels):
            x = margin + column * (tile + gap)
            draw.text((x, y + 23), title, fill="#181818", font=small)
            panel.thumbnail((tile, tile), resample=Image.Resampling.NEAREST)
            if panel.width < tile and panel.height < tile:
                scale = min(tile / panel.width, tile / panel.height)
                panel = panel.resize((round(panel.width * scale), round(panel.height * scale)), Image.Resampling.NEAREST)
            canvas.paste(panel, (x + (tile-panel.width)//2, y + 46 + (tile-panel.height)//2))
            draw.text((x, y + 242), caption, fill="#454545", font=small)
    if not selected:
        draw.text((margin, top + 20), "No valid capture/truth pairs. See failures in results.md.", fill="#801818", font=font)
    canvas.save(run_dir / "comparison.png")


def evaluate_run(run_dir):
    """Write evaluation artifacts and return the complete finite-JSON report."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise ValueError(f"run directory does not exist: {run_dir}")
    captures = {path.stem: path for path in sorted((run_dir / "captures").glob("*.npz"))}
    truths = {path.stem: path for path in sorted((run_dir / "truth").glob("*.png"))}
    report = {"schema_version": 1, "run_dir": str(run_dir), "status": "ok", "sources": [],
              "method_parameters": METHOD_PARAMETERS, "metric_definitions": METRIC_DEFINITIONS,
              "cases": [], "results": [], "summaries": [], "failures": []}
    for case_id in sorted(set(captures) | set(truths)):
        case = {"id": case_id, "source": None, "provenance": None, "status": "failed"}
        report["cases"].append(case)
        try:
            # Invalid inputs on a rerun must not leave earlier successful images.
            for method in METHODS:
                (run_dir / "outputs" / method / f"{case_id}.png").unlink(missing_ok=True)
            if case_id not in captures:
                raise ValueError("missing capture NPZ")
            mosaic_u8, provenance = _load_capture(captures[case_id])
            case.update(source=provenance.get("source"), provenance=provenance)
            if case_id not in truths:
                raise ValueError("missing truth PNG")
            truth_u8 = _validate_case(mosaic_u8, provenance, truths[case_id], case_id)
            case.update(status="ok", height=mosaic_u8.shape[0], width=mosaic_u8.shape[1],
                        resolution=f"{mosaic_u8.shape[1]}x{mosaic_u8.shape[0]}",
                        category="natural" if case_id.startswith("natural_") else "control")
        except Exception as error:
            case["error"] = f"{type(error).__name__}: {error}"
            report["failures"].append({"id": case_id, "source": case["source"], "stage": "input_validation", "error": case["error"]})
            continue
        mosaic = mosaic_u8.astype(np.float64) / 255.0
        truth = truth_u8.astype(np.float64) / 255.0
        for method in METHODS:
            output_path = run_dir / "outputs" / method / f"{case_id}.png"
            result = {"id": case_id, "source": case["source"], "provenance": provenance,
                      "category": case["category"], "resolution": case["resolution"], "method": method,
                      "parameters": METHOD_PARAMETERS[method], "status": "failed", "runtime_seconds": None}
            report["results"].append(result)
            started = None
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                started = time.perf_counter()
                output = reconstruct(mosaic.copy(), method, **METHOD_PARAMETERS[method])
                result["runtime_seconds"] = float(time.perf_counter() - started)
                metrics = calculate_metrics(output, truth, mosaic)
                Image.fromarray(np.rint(output * 255).astype(np.uint8)).save(output_path)
                result.update(status="ok", metrics=metrics, output_png=output_path.relative_to(run_dir).as_posix())
            except Exception as error:
                if started is not None and result["runtime_seconds"] is None:
                    result["runtime_seconds"] = float(time.perf_counter() - started)
                result["error"] = f"{type(error).__name__}: {error}"
                report["failures"].append({"id": case_id, "source": case["source"], "method": method,
                                           "stage": "reconstruction_or_scoring", "error": result["error"]})
    if not report["cases"]:
        report["failures"].append({"id": "<run>", "source": None, "stage": "input_validation", "error": "No capture NPZ or truth PNG files found"})
    report["sources"] = sorted({case["source"] for case in report["cases"] if isinstance(case["source"], str)})
    report["summaries"] = aggregate_results(report["results"])
    try:
        (run_dir / "comparison.png").unlink(missing_ok=True)
        _write_montage(run_dir, report)
    except Exception as error:
        report["failures"].append({"id": "<run>", "source": None, "stage": "montage", "error": f"{type(error).__name__}: {error}"})
    report["status"] = "failed" if report["failures"] else "ok"
    _write_report(run_dir, report)
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = evaluate_run(args.run_dir)
    except (OSError, ValueError) as error:
        print(f"Evaluation failed: {error}")
        return 1
    print(f"Evaluation {report['status']}: {len(report['cases'])} cases, "
          f"{sum(result['status'] == 'ok' for result in report['results'])} successful reconstructions, "
          f"{len(report['failures'])} failures")
    for filename in ("metrics.json", "results.md", "comparison.png"):
        print(str(Path(report["run_dir"]) / filename))
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
