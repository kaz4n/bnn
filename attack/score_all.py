#!/usr/bin/env python3
"""Uniform scoring for every reconstruction route in this project.

All methods are scored with the same metric set so they can be compared with
each other and with both reference papers:

  * pixel/bit accuracy and foreground F1   (this project)
  * pixel-level distance on the 0..255 scale (Wei et al., Eq. 7)
  * MSSIM, 11x11 window                    (Power2Picture / Wang et al.)
  * recognition accuracy of the recovered image (both papers)

Supported result kinds:
  active   - active_power_template_cw305.py attack output + truth json
  bgdetect - background_detect.py output directory
  p2p      - p2p_generator.py output directory
"""
import argparse
import json
import os

import numpy as np

from p2p_generator import image_metrics
from recognize_numpy import GoldenMLP


def load_active(results_dir, truth_path):
    """Read recovered images.  Works on a finished run and on one still going:
    if attack_summary.json is not written yet, pair recovered_NNNN.npz files with
    the matching truth entry by image index."""
    import glob

    truth = json.load(open(truth_path, encoding="utf-8"))
    by_file = {item["file"]: item for item in truth["items"]}
    summary_path = os.path.join(results_dir, "attack_summary.json")
    if os.path.exists(summary_path):
        pairs = [(item["recovered_file"], item["file"])
                 for item in json.load(open(summary_path, encoding="utf-8"))["images"]]
    else:
        pairs = []
        for path in sorted(glob.glob(os.path.join(results_dir, "recovered_*.npz"))):
            name = os.path.basename(path)
            pairs.append((name, f"img{name[len('recovered_'):-len('.npz')]}.npz"))
    rec, gt, labels = [], [], []
    for rec_name, src_name in pairs:
        t = by_file[src_name]
        rec.append(np.load(os.path.join(results_dir, rec_name))["recovered"])
        gt.append(np.asarray(t["image"], dtype=np.uint8))
        labels.append(int(t["label"]))
    return np.array(rec, dtype=np.float32), np.array(gt, dtype=np.uint8), np.array(labels)


def load_npz_dir(results_dir, name):
    d = np.load(os.path.join(results_dir, name))
    return (d["recovered"].astype(np.float32), d["truth"].astype(np.uint8),
            d["labels"].astype(np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["active", "bgdetect", "p2p"], required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--truth", help="truth json (active only)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.kind == "active":
        if not args.truth:
            raise SystemExit("--truth is required for --kind active")
        rec, gt, labels = load_active(args.results, args.truth)
    elif args.kind == "bgdetect":
        rec, gt, labels = load_npz_dir(args.results, "recovered.npz")
    else:
        rec, gt, labels = load_npz_dir(args.results, "test_recovered.npz")

    m = image_metrics(rec, gt.astype(np.float32))
    clf = GoldenMLP()
    m["recognition_accuracy_original"] = float(np.mean(clf.predict(gt.astype(np.float32)) == labels))
    m["recognition_accuracy_recovered"] = float(
        np.mean(clf.predict((rec >= 0.5).astype(np.float32)) == labels))
    m["n_images"] = int(len(rec))
    payload = {"name": args.name or os.path.basename(args.results.rstrip("/\\")),
               "kind": args.kind, "results": os.path.abspath(args.results),
               "metrics": m}
    out = args.out or os.path.join(args.results, "score_all.json")
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
