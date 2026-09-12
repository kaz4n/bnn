#!/usr/bin/env python3
"""Collect score_all.py outputs into one markdown table."""
import argparse
import glob
import json
import os

ROWS = [
    ("n_images", "N", "{:d}"),
    ("bit_acc", "pixel acc", "{:.4f}"),
    ("all_zero_bit_acc", "all-zero baseline", "{:.4f}"),
    ("f1_chance", "F1 chance ceiling", "{:.4f}"),
    ("f1_excess", "F1 above chance", "{:.4f}"),
    ("foreground_f1", "fg F1", "{:.4f}"),
    ("foreground_iou", "fg IoU", "{:.4f}"),
    ("mssim", "MSSIM", "{:.3f}"),
    ("pixel_level_distance", "pixel dist (0-255)", "{:.2f}"),
    ("recognition_accuracy_recovered", "recognition", "{:.4f}"),
    ("recognition_accuracy_original", "recognition (orig)", "{:.4f}"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True,
                    help="score_all.json paths or globs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths = []
    for pattern in args.files:
        paths.extend(sorted(glob.glob(pattern)) or ([pattern] if os.path.exists(pattern) else []))
    entries = [json.load(open(p, encoding="utf-8")) for p in paths]
    if not entries:
        raise SystemExit("no score files found")

    header = "| metric | " + " | ".join(e["name"] for e in entries) + " |"
    sep = "|---" * (len(entries) + 1) + "|"
    lines = [header, sep]
    for key, label, fmt in ROWS:
        cells = []
        for e in entries:
            v = e["metrics"].get(key)
            cells.append("-" if v is None else fmt.format(int(v) if fmt == "{:d}" else v))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print(table)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fp:
            fp.write(table + "\n")


if __name__ == "__main__":
    main()
