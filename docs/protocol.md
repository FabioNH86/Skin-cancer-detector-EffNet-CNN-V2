# Experimental Protocol and Run Log

Log of training runs: conditions, results, and observations. Raw outputs live in
`runs/` (git-ignored, copied from the training machine).

---

## Run B1 — Binary MEL vs non-MEL, EfficientNet-B4, native resolution 600 x 450

- **Run ID:** `B1_binary_B4_600x450`
- **Output folder:** `runs/binary/B1_binary_B4_600x450_20261111_112623/`
- **Date:** 2026-11-11, 11:26:23 – 12:25:28 (duration 00:59:06)
- **Script:** `scripts/train_binary_mel_native.py` (commit `6a608b5`), default arguments
- **Status:** Completed
- **Related decision:** D-001 in `decisions.md`

> **Key condition:** this run was trained at the **native HAM10000 resolution
> (600 x 450 px, input tensor 450 x 600 x 3)**. That is ~1.87x the pixels of
> EfficientNet-B4's design resolution (380 x 380). Overfitting was probably
> observed (see Observations), and exceeding B4's working resolution is the
> leading suspected cause.

### Environment

| Item | Value |
|---|---|
| Machine | kikai-one, Linux 7.0.0-42-generic, 1 GPU |
| Python / TensorFlow / Keras | 3.14.8 / 2.22.0-rc0 / 3.16.0.dev2026100804 |
| NumPy / pandas / scikit-learn / Pillow | 2.5.3 / 3.0.6 / 1.9.1 / 12.3.0 |

### Data

- Dataset: `data/splits-v2` (lesion-level split, original files copied byte for
  byte, **not balanced**). The balanced variant was not used.
- All 8,505 images are 600 x 450; 0 resized on the fly; 0 unreadable.
- Binary task: `mel` = positive; `akiec, bcc, bkl, df, nv, vasc` = negative.

| Split | Total | MEL | non-MEL | MEL % |
|---|---|---|---|---|
| Training | 7,002 | 770 | 6,232 | 11.0 % |
| Validation | 1,503 | 162 | 1,341 | 10.8 % |

- Class weights (sqrt): non-mel 0.75, mel 2.132.
- Test split: **not used** in this run.

### Model

- EfficientNet-B4 (ImageNet weights) -> GlobalMaxPooling2D -> Dense(256) ->
  BatchNorm -> Dropout(0.5) -> Dense(1, sigmoid).
- 18,134,112 total parameters; 459,777 trainable in phase 1; 5,269,621 in phase 2.
- Online augmentation: RandomFlip (horizontal and vertical), RandomRotation(0.2), RandomZoom(0.1).
- float32 (no mixed precision), batch size 16, seed 42.

### Training schedule

| Phase | Trainable | Initial LR | Max epochs | Epochs run | Best epoch | Best val_loss | Time |
|---|---|---|---|---|---|---|---|
| 1 – head | Head only (backbone frozen) | 5e-4 | 15 | 11 | 6 | 0.3130 | 33.9 min |
| 2 – fine-tuning | Last 30 backbone layers (BatchNorm frozen) | 1e-5 | 15 | 7 | 2 | 0.4111 | 24.0 min |

- Callbacks: EarlyStopping(val_loss, patience 5, restore best weights);
  ReduceLROnPlateau(val_loss, factor 0.3, patience 2).
- Threshold mode: `f1` (selected on the validation set) -> 0.50.

### Results (validation, final model)

| Metric | Value |
|---|---|
| ROC AUC | **0.7985** |
| Average Precision | 0.3251 |
| Threshold | 0.50 |
| Sensitivity (MEL recall) | 0.6111 (99 / 162) |
| Specificity | 0.8031 |
| Precision (MEL) | 0.2727 |
| F1 (MEL) | 0.3771 |
| Balanced accuracy | 0.7071 |
| Accuracy | 0.7824 |

Confusion matrix:

| | Pred non-MEL | Pred MEL |
|---|---|---|
| **Actual non-MEL** | 1,077 (TN) | 264 (FP) |
| **Actual MEL** | 63 (FN) | 99 (TP) |

Selected points from the threshold sweep:

| Threshold | Sensitivity | Specificity | Precision | FN | FP |
|---|---|---|---|---|---|
| 0.10 | 0.951 | 0.439 | 0.170 | 8 | 752 |
| 0.20 | 0.889 | 0.541 | 0.190 | 18 | 615 |
| 0.25 | 0.846 | 0.594 | 0.201 | 25 | 545 |
| 0.30 | 0.796 | 0.641 | 0.211 | 33 | 482 |
| 0.50 | 0.611 | 0.803 | 0.273 | 63 | 264 |

Reaching the 0.85 sensitivity target needs a threshold of about 0.25. At that
threshold the model raises ~545 false alarms out of 1,341 negatives.

### Observations

1. **Probable overfitting.** Train AUC rose steadily to 0.927 while validation
   AUC plateaued around 0.79–0.81 (gap ~0.12 at the last epoch). Train loss fell
   from 0.95 to 0.27. Validation loss reached its minimum at epoch 6 (0.3130),
   then oscillated between 0.31 and 0.52 and never improved again.
2. **Resolution above B4's design input.** At 600 x 450 the input has ~1.87x
   the pixels of B4's 380 x 380 design resolution, and the final feature map is
   15 x 19 instead of 12 x 12. This is the leading suspected cause of the
   overfitting (more high-frequency detail to memorize, and a mismatch with the
   pretrained receptive-field scale), but this run alone does not isolate it.
3. **Fine-tuning did not help.** Phase 2 did not improve on phase 1's best
   val_loss (0.4111 vs 0.3130). Validation AUC rose only marginally
   (0.791 -> 0.798) while train AUC kept climbing.
4. **Unstable validation accuracy** (0.71–0.87 across epochs). This suggests
   predictions near the decision boundary are sensitive to small weight changes.
5. **Weak separation.** Median P(mel) is 0.56 for actual melanomas and 0.16 for
   non-melanomas. The upper 10 % of non-melanomas reach P(mel) >= 0.65.

### Limitations

- Early stopping and threshold selection used the **same validation set** that
  is reported, so the metrics are optimistic.
- The training split is imbalanced (11 % MEL). Class weights were the only
  mitigation.
- No test-set evaluation and only a single seed.

### Follow-up runs planned

- **B2 (control):** identical to B1 but at B4's design resolution (380 x 380,
  or a 4:3 equivalent such as 384 x 288). This isolates the effect of input size.
- **B3:** native 600 x 450 with a backbone designed for larger inputs (B5–B7).
- Train on `data/splits-v2-balanced`, use a separate split for threshold
  selection, and report on the held-out test split.
