#!/usr/bin/env python3
"""Score saved hardware attack outputs with the golden MNIST MLP.

Run from attack/ in an environment with TensorFlow:
  python score_results.py --results results_hw
"""
import argparse, json, os, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join("..", "training"))
    from train_golden_mlp import build_mlp

    summary_path = os.path.join(args.results, "summary.json")
    with open(summary_path) as fp:
        res = json.load(fp)

    model = build_mlp()
    model.load_weights(os.path.join("..", "training", "artifacts",
                                    "golden_mlp", "weights.weights.h5"))

    def recog(img28):
        x = (img28.astype("float32") / 127.5 - 1.0)[None, ..., None]
        return int(model.predict(x, verbose=0).argmax())

    or_ok = bg_ok = tm_ok = 0
    for row in res["images"]:
        d = np.load(os.path.join(args.results, row["file"]))
        label = int(row["label"])
        ro = recog(d["original"])
        rb = recog(d["background"])
        rt = recog(d["template"])
        row["recog_orig"] = ro
        row["recog_bg"] = rb
        row["recog_tm"] = rt
        or_ok += (ro == label)
        bg_ok += (rb == label)
        tm_ok += (rt == label)

    n = len(res["images"])
    res["summary"]["recog_acc_orig"] = or_ok / n
    res["summary"]["recog_acc_background"] = bg_ok / n
    res["summary"]["recog_acc_template"] = tm_ok / n

    with open(summary_path, "w") as fp:
        json.dump(res, fp, indent=2)
    print(json.dumps(res["summary"], indent=2))


if __name__ == "__main__":
    main()
