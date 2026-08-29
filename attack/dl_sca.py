#!/usr/bin/env python3
"""
Deep-learning profiled side-channel attack (alternative route, same CW305+CW-Lite bench).

Linear correlation / template matching fail here because per-cycle leakage is ~0.1. A
neural network can combine many weakly-leaking features (9 kernels x 4 samples x the K*K
cycles whose window contains a pixel) non-linearly and recover what linear methods cannot.

Structure-aware + data-efficient: for each of the 784 pixels we gather the trace samples of
exactly the convolution cycles whose K*K window contains that pixel (the line-buffer
locality the paper exploits), giving one training example per pixel -> 784 examples/image.

Trains a profiled model on known (image, trace) pairs, then recovers held-out images.
Reports pixel accuracy + recognition accuracy (golden MLP) vs the paper.

    python dl_sca.py --traces-dir ../host/traces_dl --ksize 3 \
        --n-profile 250 --n-eval 50 --mode clf
"""
import argparse, glob, json, os
import numpy as np

LINE, SPC = 28, 4


def build_xy(files, ksize, mode):
    """Return X [n_pixels, n_feat], y [n_pixels], and per-file pixel index for recon."""
    PAD = ksize // 2
    EW = LINE + 2 * PAD
    NCONV = EW * EW
    NK = None
    # offsets of output cycles whose window contains pixel (y,x): oy in [y-P..y+P]
    X, Y, meta = [], [], []
    for fi, f in enumerate(files):
        d = np.load(f)
        tr = d["traces"]                      # [NK, NCONV*SPC]
        NK = tr.shape[0]
        t = tr[:, :NCONV*SPC].reshape(NK, NCONV, SPC)
        # per-trace standardize (kill offset/gain variation across captures)
        t = (t - t.mean(axis=(1, 2), keepdims=True)) / (t.std(axis=(1, 2), keepdims=True) + 1e-9)
        img = d["image"].astype(np.float32)
        for y in range(LINE):
            for x in range(LINE):
                feats = []
                for oy in range(y-PAD, y+PAD+1):
                    for ox in range(x-PAD, x+PAD+1):
                        if 0 <= oy < LINE and 0 <= ox < LINE:
                            er, ec = oy + (ksize-1), ox + (ksize-1)   # output-emit cycle
                            c = er * EW + ec
                            feats.append(t[:, c, :].reshape(-1))      # NK*SPC
                        else:
                            feats.append(np.zeros(NK*SPC, np.float32))
                X.append(np.concatenate(feats))
                if mode == "clf":
                    Y.append(1.0 if img[y, x] > 0 else 0.0)
                else:
                    Y.append(img[y, x] / 255.0)
                meta.append((fi, y, x))
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), meta, NK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-dir", required=True)
    ap.add_argument("--ksize", type=int, default=3)
    ap.add_argument("--n-profile", type=int, default=250)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--mode", choices=["clf", "reg"], default="clf")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default="results_dl")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import tensorflow as tf
    from evaluate import try_load_golden, recog

    files = sorted(glob.glob(os.path.join(args.traces_dir, "*.npz")))
    need = args.n_profile + args.n_eval
    assert len(files) >= need, f"need {need} captures, found {len(files)}"
    prof_f, ev_f = files[:args.n_profile], files[args.n_profile:need]

    print(f"[dl] building profiling set from {len(prof_f)} images ...")
    Xtr, Ytr, _, NK = build_xy(prof_f, args.ksize, args.mode)
    print(f"[dl] X {Xtr.shape}  (per-pixel features = {Xtr.shape[1]})")

    # SCA model: combine the weak features non-linearly
    inp = tf.keras.Input(shape=(Xtr.shape[1],))
    h = tf.keras.layers.BatchNormalization()(inp)
    for u in (256, 128, 64):
        h = tf.keras.layers.Dense(u, activation="relu")(h)
        h = tf.keras.layers.Dropout(0.2)(h)
    if args.mode == "clf":
        out = tf.keras.layers.Dense(1, activation="sigmoid")(h)
        loss, met = "binary_crossentropy", ["accuracy"]
    else:
        out = tf.keras.layers.Dense(1, activation="sigmoid")(h)
        loss, met = "mse", ["mae"]
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss, metrics=met)
    cw = None
    if args.mode == "clf":                         # MNIST ~80% background -> balance classes
        pos = float(Ytr.mean()); cw = {0: 1.0, 1: (1-pos)/max(pos, 1e-3)}
        print(f"[dl] class balance: fg={pos:.3f}, weight={cw}")
    model.fit(Xtr, Ytr, epochs=args.epochs, batch_size=256, validation_split=0.1,
              class_weight=cw, verbose=2)

    # attack held-out images
    print(f"[dl] attacking {len(ev_f)} held-out images ...")
    golden = try_load_golden()
    Xev, Yev, meta, _ = build_xy(ev_f, args.ksize, args.mode)
    pred = model.predict(Xev, batch_size=512, verbose=0).ravel()

    recon = {fi: np.zeros((LINE, LINE), np.float32) for fi in range(len(ev_f))}
    for (fi, y, x), p in zip(meta, pred):
        recon[fi][y, x] = p
    pix_acc, rec_ok, n = [], 0, len(ev_f)
    res = {"mode": args.mode, "ksize": args.ksize, "images": []}
    for fi, f in enumerate(ev_f):
        d = np.load(f); img = d["image"]; label = int(d["label"])
        r = recon[fi]
        if args.mode == "clf":
            marker = (r > 0.5).astype(np.uint8); recov = marker * 255
            pa = float((marker == (img > 0).astype(np.uint8)).mean())
        else:
            recov = (r * 255).astype(np.uint8)
            pa = float(np.abs(recov.astype(int) - img.astype(int)).mean())
        pix_acc.append(pa)
        rt = recog(golden, recov.astype(np.uint8))
        rec_ok += (rt == label)
        np.savez(os.path.join(args.out, os.path.basename(f)),
                 original=img, recovered=recov.astype(np.uint8), label=label)
        res["images"].append({"file": os.path.basename(f), "label": label,
                              "pix": pa, "recog": rt})
    res["summary"] = {"pixel_metric_mean": float(np.mean(pix_acc)),
                      "recog_acc": rec_ok / n if golden else None,
                      "paper_template_recog": 0.898 if args.ksize == 3 else 0.790}
    with open(os.path.join(args.out, "summary.json"), "w") as fp:
        json.dump(res, fp, indent=2)
    print("\n[dl summary]", json.dumps(res["summary"], indent=2))


if __name__ == "__main__":
    main()
