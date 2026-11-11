#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BINARY Training at native resolution: melanoma (mel) vs non-melanoma — HAM10000
===============================================================================
Adaptation of `entrenamiento_binario_mel.py` (previous repository) to train on
HAM10000 images at their native resolution of 600 x 450 px (width x height).

What it does
------------
1. Reads the dataset prepared by `scripts/balance_dataset_native.py`
   (default: data/splits-v2/training|validation/<class>/). The test subset is
   never used here.
2. Streams the images from disk with tf.data (no resizing when they are
   already 600 x 450) and converts labels to binary (1 = mel).
3. Builds an EfficientNet (B0/B4) with a 450 x 600 input and a sigmoid head,
   and trains it in two phases:
   head (frozen backbone) -> fine-tuning (last N layers, low lr).
4. Evaluates on validation: AUC, AP, sensitivity, specificity, precision, F1,
   confusion matrix, threshold sweep.
5. Creates a run folder with EVERYTHING that occurred:
      training_summary.txt       <- complete summary (config, environment, data, epochs,
                                     events, metrics, warnings, files)
      console.log                <- copy of the console output
      epoch_history.csv          <- metrics per epoch
      validation_predictions.csv <- P(mel) per validation image
      metrics.json               <- final and phase metrics
      training_curves.png / stage1_evaluation.png
      <name>_stage1.keras        <- trained model (+ <name>_stage1.json with metadata)

Differences with the previous (224 x 224) version
-------------------------------------------------
- Input is 450 x 600 x 3 (height x width x channels), so lesions are not
  squashed into a square and no detail is lost to downscaling.
- Images are streamed from disk with tf.data instead of being loaded into
  NumPy arrays: at 600 x 450 the full training + validation set would need
  about 7 GB of RAM.
- Image preprocessing (hair removal, color constancy, ...) is no longer
  applied here. It is applied once when the dataset is prepared
  (`balance_dataset_native.py --method ...`), so pass the matching
  --dataset-dir.
- The dataset is no longer organized by this script (the old 80/20 split
  did not match the official 70/15/15 split).
- The default batch size is 16 because each image has ~5x more pixels
  than at 224 x 224. Optional mixed precision (--mixed-precision) reduces
  GPU memory use further.

Examples (from the project root)
--------------------------------
  # default: data/splits-v2, EfficientNetB4, 600 x 450
  python3 scripts/train_binary_mel_native.py

  # quick pipeline test (30 img/class, B0, 1 epoch, no fine-tuning)
  python3 scripts/train_binary_mel_native.py --quick

  # prioritize not missing melanomas (highest threshold with sensitivity >= 85%)
  python3 scripts/train_binary_mel_native.py --thr-mode sensitivity --min-sensitivity 0.85

  # if the GPU runs out of memory
  python3 scripts/train_binary_mel_native.py --batch-size 8 --mixed-precision

For educational use only — not intended for clinical diagnosis.
"""

import argparse
import datetime as dt
import gc
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # No windows: graphs are saved as files
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (classification_report, roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (Input, Dense, Dropout, BatchNormalization, GlobalMaxPooling2D,
                                     RandomFlip, RandomRotation, RandomZoom)
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB4
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "splits-v2"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "binary"

NATIVE_IMAGE_SIZE = (600, 450)  # (width, height)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


# General utilities

def fmt_dur(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class Tee:
    """Duplicates console output to a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return False


def make_event_logger(events, t0):
    """Returns a function that logs (and displays) timestamped events."""
    def event(msg):
        line = f"[{fmt_dur(time.time() - t0)}] {msg}"
        events.append(line)
        print(line, flush=True)
    return event


def count_params(weights):
    return int(sum(int(np.prod(tuple(w.shape))) for w in weights))


def get_lr(model):
    lr = model.optimizer.learning_rate
    for fn in (lambda: float(lr.numpy()),
               lambda: float(tf.keras.backend.get_value(lr)),
               lambda: float(lr)):
        try:
            return fn()
        except Exception:
            continue
    return float("nan")


def parse_thr_mode(s):
    """'f1' | 'sensitivity' | number (fixed threshold)."""
    try:
        return float(s)
    except ValueError:
        if s in ("f1", "sensitivity"):
            return s
        raise argparse.ArgumentTypeError("thr-mode must be 'f1', 'sensitivity', or a number (e.g., 0.5)")


# Dataset: file listing and tf.data pipeline

def count_images(folder, categories):
    return {c: len([f for f in os.listdir(folder / c) if f.lower().endswith(IMAGE_EXTENSIONS)])
            if (folder / c).is_dir() else 0 for c in categories}


def list_split(folder, categories, limit=None):
    """
    List the image files of one subset and check that each one can be opened.

    Only the image header is read (fast). Unreadable files are skipped, and the
    image sizes found are returned so that resizing can be reported.
    """
    paths, labels, files, sizes, failed = [], [], [], {}, 0
    for idx, c in enumerate(categories):
        cdir = folder / c
        if not cdir.is_dir():
            continue
        names = sorted(f for f in os.listdir(cdir) if f.lower().endswith(IMAGE_EXTENSIONS))
        if limit:
            names = names[:limit]
        for f in tqdm(names, desc=f"{folder.name}/{c}", leave=False):
            path = cdir / f
            try:
                with Image.open(path) as img:
                    size = img.size
            except Exception:
                failed += 1
                continue
            sizes[size] = sizes.get(size, 0) + 1
            paths.append(str(path))
            labels.append(idx)
            files.append(f"{c}/{f}")
    return paths, np.asarray(labels, dtype=np.int64), files, sizes, failed


def make_dataset(paths, labels, img_size, batch_size, shuffle=False, seed=42):
    """
    tf.data pipeline that reads and decodes images from disk batch by batch.

    img_size is (width, height). Images that already have that size are used
    as they are; any other image is resized (area interpolation). Pixel values
    stay in [0, 255]: EfficientNet in Keras does its own rescaling.
    """
    width, height = img_size
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)  # RGB, uint8
        same_size = tf.reduce_all(tf.equal(tf.shape(img)[:2], [height, width]))
        img = tf.cond(same_size,
                      lambda: tf.cast(img, tf.float32),
                      lambda: tf.image.resize(img, (height, width), method="area"))
        img.set_shape((height, width, 3))
        return img, label

    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


# Model

def make_augmentation():
    # No color changes: color is clinically relevant information.
    # These layers work on non-square images; rotation fills corners by reflection.
    return Sequential([RandomFlip("horizontal_and_vertical"), RandomRotation(0.2), RandomZoom(0.1)],
                      name="data_augmentation")


def compile_binary(model, lr):
    model.compile(optimizer=Adam(lr), loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])


def build_binary_model(variant, img_size, augment, pretrained, head_lr):
    width, height = img_size
    input_shape = (height, width, 3)
    Base = {"B0": EfficientNetB0, "B4": EfficientNetB4}[variant]
    base = Base(include_top=False, weights=pretrained, input_shape=input_shape)
    base.trainable = False  # Phase 1: learn only the head.

    inputs = Input(shape=input_shape)
    x = make_augmentation()(inputs) if augment else inputs
    x = base(x, training=False)
    x = GlobalMaxPooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation="sigmoid", dtype="float32")(x)  # float32 also under mixed precision

    model = Model(inputs, outputs)
    compile_binary(model, head_lr)
    return model, base


def unfreeze_top_layers(model, base, n_layers, lr):
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False  # retains ImageNet statistics
    compile_binary(model, lr)  # You must recompile after changing `trainable`.


def get_class_weights(y, mode):
    assert mode in ("none", "sqrt", "balanced")
    if mode == "none":
        return None
    classes = np.unique(y)
    cw = compute_class_weight("balanced", classes=classes, y=y)
    if mode == "sqrt":
        cw = np.sqrt(cw)
    return {int(c): float(w) for c, w in zip(classes, cw)}


# Training log (callback that populates the .txt summary)

class PhaseLogger(tf.keras.callbacks.Callback):
    """Store metrics by epoch, including timing, learning rate (LR) used, best epochs, and LR reductions."""

    def __init__(self, phase, rows, event_fn, epochs_planned):
        super().__init__()
        self.phase, self.rows, self.event, self.planned = phase, rows, event_fn, epochs_planned
        self.best_val_loss, self.best_epoch, self.n = np.inf, None, 0
        self.lr_start, self._lr_epoch, self._t = None, None, None

    def on_train_begin(self, logs=None):
        self.lr_start = get_lr(self.model)
        self.event(f"[{self.phase}] start | initial lr={self.lr_start:.2e} | "
                   f"trainable parameters={count_params(self.model.trainable_weights):,} | "
                   f"max epochs={self.planned}")

    def on_epoch_begin(self, epoch, logs=None):
        self._t = time.time()
        self._lr_epoch = get_lr(self.model)  # lr used for training during THIS epoch

    def on_epoch_end(self, epoch, logs=None):
        logs = {k: float(v) for k, v in (logs or {}).items() if k not in ("lr", "learning_rate")}
        self.n += 1
        self.rows.append({"phase": self.phase, "epoch": len(self.rows) + 1, "epoch_phase": epoch + 1,
                          "lr": self._lr_epoch, "seconds": round(time.time() - self._t, 1), **logs})
        v = logs.get("val_loss")
        if v is not None and v < self.best_val_loss:
            self.best_val_loss, self.best_epoch = v, epoch + 1
            self.event(f"[{self.phase}] epoch {epoch + 1}: new best val_loss={v:.4f} "
                       f"(val_auc={logs.get('val_auc', float('nan')):.4f})")
        lr_now = get_lr(self.model)
        if lr_now < self._lr_epoch * 0.999:
            self.event(f"[{self.phase}] ReduceLROnPlateau after epoch {epoch + 1}: "
                       f"lr {self._lr_epoch:.2e} -> {lr_now:.2e}")


def fit_phase(model, name, train_ds, val_ds, epochs, class_weight, rows, event):
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, verbose=1)
    logger = PhaseLogger(name, rows, event, epochs)
    t0 = time.time()
    model.fit(train_ds, validation_data=val_ds, epochs=epochs,
              class_weight=class_weight, callbacks=[es, rlr, logger], verbose=2)
    stopped_early = logger.n < epochs
    if stopped_early:
        event(f"[{name}] EarlyStopping: stopped after {logger.n} of {epochs} epochs (patience=5); "
              f"best epoch={logger.best_epoch} (val_loss={logger.best_val_loss:.4f}), "
              f"weights restored to that epoch")
    else:
        event(f"[{name}] completed {epochs} epochs without triggering EarlyStopping "
              f"(best epoch={logger.best_epoch}, val_loss={logger.best_val_loss:.4f})")
    return dict(phase=name, max_epochs=epochs, epochs_run=logger.n, early_stopping=stopped_early,
                best_epoch=logger.best_epoch, best_val_loss=float(logger.best_val_loss),
                initial_lr=logger.lr_start, final_lr=get_lr(model), seconds=round(time.time() - t0, 1))


# Metrics and charts (stage 1)

def binary_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1); prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    return dict(thr=float(thr), sensitivity=sens, specificity=spec, precision=prec, f1=f1,
                balanced_acc=(sens + spec) / 2, accuracy=(tp + tn) / len(y), tp=tp, fp=fp, fn=fn, tn=tn)


def choose_threshold(y, p, mode="f1", min_sensitivity=0.85):
    if isinstance(mode, (int, float)):
        return float(mode)  # fixed threshold
    ts = np.linspace(0.01, 0.99, 99)
    ms = [binary_metrics(y, p, t) for t in ts]
    if mode == "f1":
        return float(ts[int(np.argmax([m["f1"] for m in ms]))])
    if mode == "sensitivity":
        ok = [t for t, m in zip(ts, ms) if m["sensitivity"] >= min_sensitivity]
        return float(max(ok)) if ok else float(ts[0])  # the highest one that still meets the sensitivity requirement
    raise ValueError("thr_mode: 'f1', 'sensitivity' or a number")


def plot_history(df, path, title):
    panels = [(c, t) for c, t in [("accuracy", "Accuracy"), ("loss", "Loss"), ("auc", "AUC")] if c in df.columns]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4))
    axes = np.atleast_1d(axes)
    boundaries = df.index[df["phase"] != df["phase"].shift()].tolist()[1:]  # phase changes
    for ax, (col, label) in zip(axes, panels):
        ax.plot(df["epoch"], df[col], label=f"train_{col}")
        ax.plot(df["epoch"], df[f"val_{col}"], label=f"val_{col}")
        for b in boundaries:
            ax.axvline(df.loc[b, "epoch"] - 0.5, color="gray", ls="--", lw=1)
        ax.set_xlabel("Epoch"); ax.set_ylabel(label); ax.set_title(f"{label}"); ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_stage1(y, p, thr, path, title):
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    m = binary_metrics(y, p, thr)
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    axes[0].plot(fpr, tpr); axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].scatter([1 - m["specificity"]], [m["sensitivity"]], color="red", zorder=3, label=f"threshold {thr:.2f}")
    axes[0].set(xlabel="1 − specificity", ylabel="sensitivity (recall of mel)",
                title=f"ROC (AUC = {roc_auc_score(y, p):.3f})")
    axes[0].legend(loc="lower right")
    axes[1].plot(rec, prec)
    axes[1].axhline(y.mean(), color="k", ls="--", lw=0.8, label="chance (prevalence)")
    axes[1].scatter([m["sensitivity"]], [m["precision"]], color="red", zorder=3, label=f"threshold {thr:.2f}")
    axes[1].set(xlabel="recall (mel)", ylabel="precision (mel)",
                title=f"Precision-Recall (AP = {average_precision_score(y, p):.3f})")
    axes[1].legend(loc="upper right")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[2],
                xticklabels=["non-mel", "mel"], yticklabels=["non-mel", "mel"])
    axes[2].set(xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix (threshold {thr:.2f})")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# Construction of the .txt summary

def _df_txt(df, **kw):
    sci = {c: (lambda v: f"{v:.2e}") for c in ("lr", "initial_lr", "final_lr") if c in df.columns}
    return df.to_string(index=False, float_format=lambda v: f"{v:.4f}", formatters=sci, **kw)


def build_report(ctx):
    L = []

    def section(title):
        L.extend(["", "=" * 86, title, "=" * 86])

    def kv(k, v):
        L.append(f"{k:<34}: {v}")

    a = ctx.get("args", {})
    L.append("SUMMARY OF BINARY TRAINING — melanoma (MEL) vs. non-melanoma (native resolution)")
    L.append("For educational use only — not intended for clinical diagnosis.")

    # 1. Executive summary
    section("1. EXECUTIVE SUMMARY")
    kv("Run name", a.get("name"))
    kv("Status", ctx.get("status"))
    kv("Start time", ctx.get("start"))
    kv("End time", ctx.get("end"))
    kv("Total duration", fmt_dur(ctx.get("elapsed", 0)))
    phases = ctx.get("phases", [])
    if phases:
        kv("Epochs executed (total)", sum(p["epochs_run"] for p in phases))
    ev = ctx.get("eval")
    if ev:
        kv("AUC (validation)", f"{ev['auc']:.4f}" if ev["auc"] is not None else "n/a")
        kv("Average Precision (validation)", f"{ev['ap']:.4f}" if ev["ap"] is not None else "n/a")
        kv(f"Chosen threshold ({ev['thr_mode']})", f"{ev['thr']:.2f}")
        m = ev["at_thr"]
        kv("Sensitivity (mel recall)", f"{m['sensitivity']:.4f}  ({m['tp']} of {m['tp'] + m['fn']} melanomas detected)")
        kv("Specificity", f"{m['specificity']:.4f}")
        kv("Precision (mel)", f"{m['precision']:.4f}")
        kv("F1 (mel)", f"{m['f1']:.4f}")
        kv("False negatives / false positives", f"{m['fn']} / {m['fp']}")
    if ctx.get("error"):
        L.append("")
        L.append("!! EXECUTION DID NOT FINISH CORRECTLY — see section 9 (error).")

    # 2. Environment
    section("2. ENVIRONMENT")
    for k, v in ctx.get("env", {}).items():
        kv(k, v)
    kv("Command", ctx.get("cmd"))

    # 3. Configuration
    section("3. CONFIGURATION")
    for k, v in a.items():
        kv(k, v)

    # 4. Data
    d = ctx.get("dataset")
    section("4. DATA")
    if d:
        kv("Training folder", d["train_folder"])
        kv("Validation folder", d["val_folder"])
        kv("Classes (alphabetical order)", ", ".join(d["categories"]))
        L.append("")
        L.append("Images per class on disk (before loading):")
        tab = pd.DataFrame({"class": d["categories"],
                            "training": [d["counts_train"][c] for c in d["categories"]],
                            "validation": [d["counts_val"][c] for c in d["categories"]]})
        L.append(tab.to_string(index=False))
        L.append("")
        kv("Images used (train / val)", f"{d['n_train']} / {d['n_val']}")
        kv("Unreadable images skipped", f"train={d['failed_train']}, val={d['failed_val']}")
        kv("Melanomas train", f"{d['mel_train']}  ({100 * d['mel_train'] / max(d['n_train'], 1):.1f} %)")
        kv("Melanomas val", f"{d['mel_val']}  ({100 * d['mel_val'] / max(d['n_val'], 1):.1f} %)")
        kv("Non-melanoma train / val", f"{d['n_train'] - d['mel_train']} / {d['n_val'] - d['mel_val']}")
        kv("Model input (H, W, C)", d["input_shape"])
        kv("Image sizes on disk (W x H: n)", d["sizes"])
        kv("Images resized on the fly", d["n_resized"])
        kv("Loading", "streamed from disk with tf.data (not held in RAM)")
        kv("File check time", fmt_dur(d["load_seconds"]))
        kv("Class weights (mode)", a.get("class_weights"))
        kv("Class weights (values)", ctx.get("class_weights"))
    else:
        L.append("(dataset was not loaded)")

    # 5. Model
    section("5. MODEL")
    mo = ctx.get("model")
    if mo:
        kv("Architecture", f"EfficientNet{a.get('variant')} + GlobalMaxPool + Dense(256) + BN + Dropout(0.5) + Dense(1, sigmoid)")
        kv("Backbone weights", a.get("pretrained"))
        kv("Compute precision", "mixed_float16" if a.get("mixed_precision") else "float32")
        kv("Total parameters", f"{mo['params_total']:,}")
        kv("Trainable in phase 1 (head)", f"{mo['trainable_head']:,}")
        if mo.get("trainable_ft") is not None:
            kv("Trainable in phase 2 (fine-tuning)", f"{mo['trainable_ft']:,}")
        L.append("")
        L.append("model.summary():")
        L.extend(mo["summary"])
    else:
        L.append("(model was not built)")

    # 6. Training
    section("6. TRAINING")
    if phases:
        L.append("Phases:")
        L.append(_df_txt(pd.DataFrame(phases)))
        L.append("")
    hist = ctx.get("history")
    if hist is not None and len(hist):
        L.append("Metrics per epoch (lr = learning rate used in that epoch):")
        L.append(_df_txt(hist))
    L.append("")
    L.append("Event timeline:")
    L.extend(ctx.get("events", []) or ["(no events)"])

    # 7. Evaluation
    section("7. VALIDATION EVALUATION (final model)")
    if ev:
        kv("Threshold", f"{ev['thr']:.2f}  (mode: {ev['thr_mode']}"
            + (f", minimum sensitivity {a.get('min_sensitivity')}" if ev['thr_mode'] == 'sensitivity' else "") + ")")
        kv("val_loss / val_accuracy / val_auc (keras)", ", ".join(f"{k}={v:.4f}" for k, v in ev["keras_eval"].items()))
        L.append("")
        m = ev["at_thr"]
        L.append("Confusion matrix (rows = actual, columns = predicted):")
        L.append(f"{'':>15}{'pred non-mel':>15}{'pred mel':>12}")
        L.append(f"{'actual non-mel':>15}{m['tn']:>15}{m['fp']:>12}")
        L.append(f"{'actual mel':>15}{m['fn']:>15}{m['tp']:>12}")
        L.append("")
        L.append("Metrics at chosen threshold:")
        L.append(_df_txt(pd.DataFrame([m])))
        L.append("")
        L.append("Metrics with threshold 0.5 (reference):")
        L.append(_df_txt(pd.DataFrame([ev["at_05"]])))
        L.append("")
        L.append("Classification report (chosen threshold):")
        L.append(ev["report"])
        L.append("P(mel) distribution in validation:")
        L.append(_df_txt(pd.DataFrame(ev["prob_stats"])))
        L.append("")
        L.append("Threshold sweep (sensitivity / false alarms tradeoff):")
        L.append(_df_txt(pd.DataFrame(ev["sweep"])))
    else:
        L.append("(evaluation was not reached)")

    # 8. Files
    section("8. GENERATED FILES")
    L.extend(ctx.get("files", []) or ["(none)"])

    # 9. Notes / warnings / error
    section("9. AUTOMATIC NOTES AND WARNINGS")
    for w in ctx.get("warnings", []):
        L.append(f"- {w}")
    L.append("- The threshold and early stopping are decided using the SAME validation set used for evaluation: "
             "the metrics are somewhat optimistic.")
    if ctx.get("error"):
        L.extend(["", "ERROR / INTERRUPTION:", ctx["error"]])

    return "\n".join(L) + "\n"


def derive_warnings(ctx):
    w = []
    phases = ctx.get("phases", [])
    d = ctx.get("dataset")
    if d and d["mel_val"] < 50:
        w.append(f"There are only {d['mel_val']} melanomas in validation: metrics are highly unstable.")
    if d and (d["failed_train"] or d["failed_val"]):
        w.append(f"Unreadable images were skipped (train={d['failed_train']}, val={d['failed_val']}).")
    if d and d["n_resized"]:
        w.append(f"{d['n_resized']} images did not have the model input size and were resized on the fly.")
    if len(phases) == 2 and phases[1]["best_val_loss"] >= phases[0]["best_val_loss"]:
        w.append("Fine-tuning did NOT improve the best val_loss from phase 1 "
                 f"({phases[1]['best_val_loss']:.4f} >= {phases[0]['best_val_loss']:.4f}); "
                 "the final model comes from phase 2.")
    hist = ctx.get("history")
    if hist is not None and len(hist) and {"auc", "val_auc"} <= set(hist.columns):
        last = hist.iloc[-1]
        if last["auc"] - last["val_auc"] > 0.10:
            w.append(f"Possible overfitting: train AUC {last['auc']:.3f} vs val {last['val_auc']:.3f} in the last epoch.")
    ev = ctx.get("eval")
    if ev and ev["at_thr"]["sensitivity"] < 0.5:
        w.append(f"Low sensitivity ({ev['at_thr']['sensitivity']:.2f}): the model misses many melanomas.")
    if ev and ev["auc"] is not None and ev["auc"] < 0.7:
        w.append(f"Low AUC ({ev['auc']:.3f}): the model has poor discrimination.")
    if not w:
        w.append("No relevant automatic warnings.")
    return w


# Main execution

def run_training(args, run_dir, ctx, event):
    files = ctx["files"]
    tf.keras.utils.set_random_seed(args.seed)
    img_size = tuple(args.img_size)  # (width, height)

    # dataset
    dataset_dir = Path(args.dataset_dir)
    train_folder, val_folder = dataset_dir / "training", dataset_dir / "validation"
    for folder in (train_folder, val_folder):
        if not folder.is_dir():
            raise FileNotFoundError(f"{folder} does not exist. Prepare the dataset first with "
                                    f"scripts/balance_dataset_native.py or pass --dataset-dir.")
    categories = sorted(c for c in os.listdir(train_folder) if (train_folder / c).is_dir())
    if "mel" not in categories:
        raise ValueError(f"No 'mel' folder found in {train_folder}. Classes found: {categories}")
    mel_idx = categories.index("mel")
    counts_train, counts_val = count_images(train_folder, categories), count_images(val_folder, categories)
    event(f"Classes: {categories} | train={sum(counts_train.values())} val={sum(counts_val.values())} images on disk")

    t_load = time.time()
    event(f"Checking image files (input {img_size[0]}x{img_size[1]}, limit_per_class={args.limit_per_class}) ...")
    paths_tr, Y_tr, _, sizes_tr, failed_tr = list_split(train_folder, categories, args.limit_per_class)
    paths_va, Y_va, files_va, sizes_va, failed_va = list_split(val_folder, categories, args.limit_per_class)
    Y1_tr, Y1_va = (Y_tr == mel_idx).astype(np.int64), (Y_va == mel_idx).astype(np.int64)  # 1 = melanoma
    sizes = {k: sizes_tr.get(k, 0) + sizes_va.get(k, 0) for k in set(sizes_tr) | set(sizes_va)}
    n_resized = sum(n for k, n in sizes.items() if k != img_size)
    ctx["dataset"] = dict(train_folder=str(train_folder), val_folder=str(val_folder), categories=categories,
                          counts_train=counts_train, counts_val=counts_val,
                          n_train=len(Y1_tr), n_val=len(Y1_va),
                          mel_train=int(Y1_tr.sum()), mel_val=int(Y1_va.sum()),
                          failed_train=failed_tr, failed_val=failed_va,
                          input_shape=(img_size[1], img_size[0], 3),
                          sizes=", ".join(f"{w}x{h}: {n}" for (w, h), n in sorted(sizes.items())),
                          n_resized=n_resized, load_seconds=time.time() - t_load)
    event(f"Data ready: train {len(Y1_tr)} images ({int(Y1_tr.sum())} mel) | val {len(Y1_va)} images "
          f"({int(Y1_va.sum())} mel) | unreadable train={failed_tr} val={failed_va} | resized on the fly={n_resized}")
    if len(Y1_tr) == 0 or len(Y1_va) == 0:
        raise ValueError("No readable images found in training or validation.")

    train_ds = make_dataset(paths_tr, Y1_tr, img_size, args.batch_size, shuffle=True, seed=args.seed)
    val_ds = make_dataset(paths_va, Y1_va, img_size, args.batch_size)

    cw = get_class_weights(Y1_tr, args.class_weights)
    ctx["class_weights"] = {("mel" if k == 1 else "non-mel"): round(v, 3) for k, v in cw.items()} if cw else "none"
    if cw:
        event(f"Class weights ({args.class_weights}): {ctx['class_weights']}")

    # model
    tf.keras.backend.clear_session()
    gc.collect()
    tf.keras.utils.set_random_seed(args.seed)
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        event("Mixed precision enabled (mixed_float16)")
    pretrained = None if args.pretrained == "none" else args.pretrained
    event(f"Building EfficientNet{args.variant} (weights={args.pretrained}, "
          f"input {img_size[1]}x{img_size[0]}x3) ...")
    model, base = build_binary_model(args.variant, img_size, args.augment, pretrained, args.head_lr)
    summary_lines = []
    model.summary(print_fn=lambda s, **kw: summary_lines.append(s))
    ctx["model"] = dict(params_total=model.count_params(), trainable_head=count_params(model.trainable_weights),
                        trainable_ft=None, summary=summary_lines)

    # training
    rows, phases = [], ctx["phases"]
    phases.append(fit_phase(model, "phase1_head", train_ds, val_ds, args.epochs_head, cw, rows, event))
    if args.fine_tune:
        unfreeze_top_layers(model, base, args.ft_layers, args.ft_lr)
        ctx["model"]["trainable_ft"] = count_params(model.trainable_weights)
        event(f"Fine-tuning: unfreezing the last {args.ft_layers} layers of the backbone (BatchNorm frozen), "
              f"lr={args.ft_lr:.0e}")
        phases.append(fit_phase(model, "phase2_finetuning", train_ds, val_ds, args.epochs_ft, cw, rows, event))
    hist = pd.DataFrame(rows)
    pref = ["phase", "epoch", "epoch_phase", "loss", "accuracy", "auc", "val_loss", "val_accuracy", "val_auc", "lr", "seconds"]
    hist = hist[[c for c in pref if c in hist.columns] + [c for c in hist.columns if c not in pref]]
    ctx["history"] = hist
    hist.to_csv(run_dir / "epoch_history.csv", index=False)
    files.append("epoch_history.csv — metrics per epoch")
    plot_history(hist, run_dir / "training_curves.png", f"{args.name} — Stage 1 (mel vs non-mel)")
    files.append("training_curves.png — accuracy / loss / AUC curves (gray line = phase change)")

    # evaluation (val_ds is not shuffled, so predictions follow the order of files_va)
    event("Evaluating on validation ...")
    p_mel = model.predict(val_ds, verbose=0).ravel().astype(np.float64)
    keras_eval = dict(zip(["val_loss", "val_accuracy", "val_auc"], model.evaluate(val_ds, verbose=0)))
    thr = choose_threshold(Y1_va, p_mel, args.thr_mode, args.min_sensitivity)
    m_thr, m_05 = binary_metrics(Y1_va, p_mel, thr), binary_metrics(Y1_va, p_mel, 0.5)
    try:
        auc, ap = float(roc_auc_score(Y1_va, p_mel)), float(average_precision_score(Y1_va, p_mel))
        plot_stage1(Y1_va, p_mel, thr, run_dir / "stage1_evaluation.png", f"{args.name} — Stage 1")
        files.append("stage1_evaluation.png — ROC curve, Precision-Recall curve, and confusion matrix")
    except ValueError as e:  # e.g., validation set without melanomas (with a very low --limit-per-class)
        auc = ap = None
        event(f"WARNING: could not calculate AUC/AP ({e})")
    pred = (p_mel >= thr).astype(int)
    report = classification_report(Y1_va, pred, labels=[0, 1], target_names=["non-mel", "mel"], zero_division=0)
    prob_stats = [dict(group=g, n=int((Y1_va == v).sum()), mean=float(p_mel[Y1_va == v].mean()) if (Y1_va == v).any() else float("nan"),
                       median=float(np.median(p_mel[Y1_va == v])) if (Y1_va == v).any() else float("nan"),
                       p10=float(np.percentile(p_mel[Y1_va == v], 10)) if (Y1_va == v).any() else float("nan"),
                       p90=float(np.percentile(p_mel[Y1_va == v], 90)) if (Y1_va == v).any() else float("nan"))
                  for g, v in [("non-mel (actual)", 0), ("mel (actual)", 1)]]
    sweep = [binary_metrics(Y1_va, p_mel, t) for t in np.round(np.arange(0.05, 0.96, 0.05), 2)]
    ctx["eval"] = dict(auc=auc, ap=ap, thr=thr, thr_mode=str(args.thr_mode), at_thr=m_thr, at_05=m_05,
                       keras_eval={k: float(v) for k, v in keras_eval.items()},
                       report=report, prob_stats=prob_stats, sweep=sweep)
    event(f"Threshold={thr:.2f} ({args.thr_mode}) | AUC={auc if auc is None else round(auc, 3)} | "
          f"sensitivity={m_thr['sensitivity']:.3f} | specificity={m_thr['specificity']:.3f} | "
          f"precision={m_thr['precision']:.3f}")

    # save artifacts
    pd.DataFrame({"file": files_va, "original_class": [categories[i] for i in Y_va],
                  "y_actual_mel": Y1_va, "p_mel": p_mel, "pred_mel": pred}).to_csv(
        run_dir / "validation_predictions.csv", index=False)
    files.append("validation_predictions.csv — P(mel) and prediction per validation image")

    model_path = run_dir / f"{args.name}_stage1.keras"
    model.save(model_path)
    files.append(f"{model_path.name} — final trained model")
    meta = dict(thr=thr, img_width=img_size[0], img_height=img_size[1], to_rgb=True,
                variant=args.variant, dataset_dir=str(dataset_dir), mixed_precision=args.mixed_precision)
    (run_dir / f"{args.name}_stage1.json").write_text(json.dumps(meta, indent=2))
    files.append(f"{args.name}_stage1.json — metadata (threshold and input format)")

    (run_dir / "metrics.json").write_text(json.dumps(
        dict(config={k: str(v) for k, v in vars(args).items()}, phases=phases,
             evaluation={k: v for k, v in ctx["eval"].items() if k not in ("report",)}),
        indent=2, default=float))
    files.append("metrics.json — final metrics, phases, and configuration")


def parse_args():
    p = argparse.ArgumentParser(description="Binary training mel vs non-mel (HAM10000) at native resolution "
                                            "with .txt summary",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR,
                   help="Folder with training/ and validation/ (output of balance_dataset_native.py)")
    g.add_argument("--limit-per-class", type=int, default=None, help="Max. images per class (quick runs)")
    g.add_argument("--img-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=list(NATIVE_IMAGE_SIZE),
                   help="Model input width and height (native HAM10000: 600 450)")
    g = p.add_argument_group("model / training")
    g.add_argument("--variant", choices=["B0", "B4"], default="B4")
    g.add_argument("--pretrained", choices=["imagenet", "none"], default="imagenet")
    g.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--class-weights", choices=["none", "sqrt", "balanced"], default="sqrt")
    g.add_argument("--epochs-head", type=int, default=15)
    g.add_argument("--fine-tune", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--epochs-ft", type=int, default=15)
    g.add_argument("--ft-layers", type=int, default=30)
    g.add_argument("--head-lr", type=float, default=5e-4)
    g.add_argument("--ft-lr", type=float, default=1e-5)
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=False,
                   help="Use mixed_float16 to reduce GPU memory use")
    g.add_argument("--seed", type=int, default=42)
    g = p.add_argument_group("threshold")
    g.add_argument("--thr-mode", type=parse_thr_mode, default="f1", help="'f1', 'sensitivity', or a fixed number (e.g. 0.5)")
    g.add_argument("--min-sensitivity", type=float, default=0.85, help="Used if thr-mode='sensitivity'")
    g = p.add_argument_group("output")
    g.add_argument("--name", default="B1_binary_B4_600x450", help="Run name")
    g.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Folder where <name>_<date>/ is created")
    g.add_argument("--quick", action="store_true", help="Quick test: 30 img/class, B0, 1 epoch, no fine-tuning")
    args = p.parse_args()

    if min(args.img_size) < 32:
        p.error("Image width and height must be at least 32 (EfficientNet minimum).")
    if args.quick:
        args.limit_per_class, args.variant, args.epochs_head, args.fine_tune = 30, "B0", 1, False
        if args.name == p.get_default("name"):
            args.name = "QUICK_TEST_binary_600x450"
    return args


def main():
    args = parse_args()
    started = dt.datetime.now()
    run_dir = Path(args.output_dir) / f"{args.name}_{started:%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    console_log = open(run_dir / "console.log", "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, console_log)

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    try:
        build = tf.sysconfig.get_build_info()
        cuda = f"CUDA {build.get('cuda_version', 'n/a')} / cuDNN {build.get('cudnn_version', 'n/a')}"
    except Exception:
        cuda = "n/a"
    import sklearn
    import PIL
    env = {"Computer": platform.node(), "System": platform.platform(), "Python": platform.python_version(),
           "TensorFlow": tf.__version__, "Keras": getattr(tf.keras, "__version__", "n/a"), "NumPy": np.__version__,
           "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "Pillow": PIL.__version__,
           "GPU(s)": ", ".join(g.name for g in gpus) if gpus else "none (CPU)", "TF Build": cuda}

    t0 = time.time()
    events = []
    event = make_event_logger(events, t0)
    ctx = dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               cmd=" ".join(sys.argv), env=env, start=f"{started:%Y-%m-%d %H:%M:%S}",
               events=events, files=[], phases=[], status="IN PROGRESS", run_dir=str(run_dir))
    ctx["args"]["thr_mode"] = str(args.thr_mode)

    exit_code = 0
    try:
        event(f"Run '{args.name}' — results in {run_dir}")
        run_training(args, run_dir, ctx, event)
        ctx["status"] = "COMPLETED"
    except BaseException as e:  # includes Ctrl+C: partial summary is still written
        ctx["status"] = "INTERRUPTED" if isinstance(e, KeyboardInterrupt) else "ERROR"
        ctx["error"] = traceback.format_exc()
        event(f"{ctx['status']}: {type(e).__name__}: {e}")
        exit_code = 1
    finally:
        ctx["elapsed"] = time.time() - t0
        ctx["end"] = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}"
        ctx["warnings"] = derive_warnings(ctx)
        ctx["files"] += ["training_summary.txt — this file", "console.log — complete console output"]
        report_path = run_dir / "training_summary.txt"
        report_path.write_text(build_report(ctx), encoding="utf-8")
        print(f"\nSummary written to: {report_path}")
        sys.stdout = sys.__stdout__
        console_log.close()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
