"""Evaluation provenance, scoring, failure visibility, and truth isolation."""

import hashlib
import json

import numpy as np
from PIL import Image
import pytest

from experiments.rgb_instrumented import evaluate as ev


def write_case(run_dir, case_id="natural_fixture_4", *, source="software_model", rgb=None):
    if rgb is None:
        rgb = np.random.default_rng(17).integers(0, 256, (4, 6, 3), dtype=np.uint8)
    (run_dir / "captures").mkdir(exist_ok=True)
    (run_dir / "truth").mkdir(exist_ok=True)
    Image.fromarray(rgb).save(run_dir / "truth" / f"{case_id}.png")
    mosaic = rgb[..., 1].copy()
    mosaic[::2, ::2] = rgb[::2, ::2, 0]
    mosaic[1::2, 1::2] = rgb[1::2, 1::2, 2]
    provenance = {"source": source, "pattern": "RGGB", "input_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(), "hardware_validation_passed": source == "fpga", "image_source_url": "synthetic:test-fixture"}
    np.savez(run_dir / "captures" / f"{case_id}.npz", mosaic=mosaic, provenance_json=json.dumps(provenance))
    return mosaic, provenance


def rewrite_capture(run_dir, case_id, mosaic, provenance):
    np.savez(run_dir / "captures" / f"{case_id}.npz", mosaic=mosaic, provenance_json=json.dumps(provenance))


def test_full_evaluation_keeps_truth_out_of_reconstruction(tmp_path, monkeypatch):
    mosaic, provenance = write_case(tmp_path)
    original_reconstruct = ev.reconstruct
    calls = []
    def checked_reconstruct(input_mosaic, method, **kwargs):
        np.testing.assert_array_equal(input_mosaic, mosaic.astype(float) / 255)
        assert kwargs == ev.METHOD_PARAMETERS[method]
        calls.append(method)
        return original_reconstruct(input_mosaic, method, **kwargs)
    monkeypatch.setattr(ev, "reconstruct", checked_reconstruct)
    assert ev.main(["--run-dir", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "metrics.json").read_text(), parse_constant=lambda value: pytest.fail(value))
    assert calls == list(ev.METHODS)
    assert report["status"] == "ok"
    assert len(report["results"]) == 4
    for result in report["results"]:
        assert result["provenance"] == provenance
        assert result["source"] == "software_model"
        assert result["metrics"]["measurement_max_abs_error"] == 0
        assert (tmp_path / result["output_png"]).is_file()
    with Image.open(tmp_path / "comparison.png") as montage:
        assert montage.width > 1000
    assert "software_model" in (tmp_path / "results.md").read_text()


def test_perfect_psnr_json_and_control_separation(tmp_path):
    rgb = np.full((4, 4, 3), [51, 102, 153], dtype=np.uint8)
    write_case(tmp_path, "control_constant_4", rgb=rgb)
    write_case(tmp_path, "natural_fixture_4")
    report = ev.evaluate_run(tmp_path)
    assert report["status"] == "ok"
    assert len(report["summaries"]) == 8
    perfect = next(row for row in report["results"] if row["category"] == "control" and row["method"] == "nearest")
    assert perfect["metrics"]["rgb_psnr_db"] is None
    assert perfect["metrics"]["rgb_psnr_is_infinite"] is True
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("problem", ["hash", "pattern", "source", "hardware_flag", "mosaic", "dtype", "missing_truth"])
def test_invalid_case_is_saved_and_other_case_continues(tmp_path, problem):
    mosaic, provenance = write_case(tmp_path, "control_bad_4", source="fpga")
    write_case(tmp_path, "natural_good_4")
    for method in ev.METHODS:
        stale_image = tmp_path / "outputs" / method / "control_bad_4.png"
        stale_image.parent.mkdir(parents=True)
        Image.new("RGB", (4, 4)).save(stale_image)
    if problem == "hash":
        provenance["input_sha256"] = "0" * 64
    elif problem == "pattern":
        provenance["pattern"] = "BGGR"
    elif problem == "source":
        provenance["source"] = "unknown"
    elif problem == "hardware_flag":
        provenance["hardware_validation_passed"] = False
    elif problem == "mosaic":
        mosaic[0, 0] ^= 1
    elif problem == "dtype":
        mosaic = mosaic.astype(float)
    elif problem == "missing_truth":
        (tmp_path / "truth" / "control_bad_4.png").unlink()
    rewrite_capture(tmp_path, "control_bad_4", mosaic, provenance)
    assert ev.main(["--run-dir", str(tmp_path)]) == 1
    report = json.loads((tmp_path / "metrics.json").read_text())
    assert len(report["failures"]) == 1
    assert len(report["results"]) == 4
    assert all(result["id"] == "natural_good_4" for result in report["results"])
    assert report["failures"][0]["stage"] == "input_validation"
    assert "control_bad_4" in (tmp_path / "results.md").read_text()
    for method in ev.METHODS:
        assert not (tmp_path / "outputs" / method / "control_bad_4.png").exists()


def test_method_failure_visible_with_nonzero_exit(tmp_path, monkeypatch):
    write_case(tmp_path)
    real_reconstruct = ev.reconstruct
    def failed_method(mosaic, method, **kwargs):
        if method == "tv":
            raise RuntimeError("test iteration budget exhausted")
        return real_reconstruct(mosaic, method, **kwargs)
    monkeypatch.setattr(ev, "reconstruct", failed_method)
    assert ev.main(["--run-dir", str(tmp_path)]) == 1
    report = json.loads((tmp_path / "metrics.json").read_text())
    assert len(report["failures"]) == 1
    assert sum(row["status"] == "ok" for row in report["results"]) == 3
    failed_summary = next(row for row in report["summaries"] if row["method"] == "tv")
    assert failed_summary["failed_images"] == 1
    assert failed_summary["pooled_rgb_psnr_db"] is None


def test_pooled_and_mean_psnr_are_distinct():
    rows = []
    for error, count in ((0.01, 12), (0.09, 48)):
        metric = {}
        for prefix in ("rgb", "missing"):
            metric.update({f"{prefix}_squared_error_sum": error * count, f"{prefix}_component_count": count,
                           f"{prefix}_psnr_db": float(-10 * np.log10(error)), f"{prefix}_psnr_is_infinite": False})
        rows.append({"source": "fpga", "category": "natural", "resolution": "test", "method": "nearest", "status": "ok", "runtime_seconds": 0.1, "metrics": metric})
    summary = ev.aggregate_results(rows)[0]
    assert summary["pooled_rgb_mse"] == pytest.approx(0.074)
    assert summary["pooled_rgb_psnr_db"] == pytest.approx(-10 * np.log10(0.074))
    assert summary["mean_per_image_rgb_psnr_db"] == pytest.approx((20 - 10 * np.log10(0.09)) / 2)
    assert summary["pooled_rgb_psnr_db"] != summary["mean_per_image_rgb_psnr_db"]


def test_missing_metrics_exclude_measured_components():
    truth = np.full((2, 2, 3), 0.5)
    mosaic = np.full((2, 2), 0.5)
    result = truth.copy()
    result[0, 0, 1] = 0.8
    metrics = ev.calculate_metrics(result, truth, mosaic)
    assert metrics["rgb_mse"] == pytest.approx(0.09 / 12)
    assert metrics["missing_mse"] == pytest.approx(0.09 / 8)
    assert metrics["channel_mae_rgb"] == pytest.approx([0, 0.075, 0])


def test_empty_run_and_missing_capture_fail_explicitly(tmp_path):
    assert ev.main(["--run-dir", str(tmp_path)]) == 1
    write_case(tmp_path)
    (tmp_path / "captures" / "natural_fixture_4.npz").unlink()
    report = ev.evaluate_run(tmp_path)
    assert report["status"] == "failed"
    assert report["failures"][0]["error"].endswith("missing capture NPZ")
