# Decision Log

Record of design decisions, their rationale, and their observed consequences.
Each entry is dated and references the run(s) that support it.

---

## D-001 — Train EfficientNet-B4 at HAM10000 native resolution (600 x 450)

- **Date:** 2026-11-11
- **Status:** Tested in run B1 — result **not satisfactory**; superseded for the B4 baseline by D-002 (run B2)
- **Related commits:** `afa50d4` (Use of native resolution), `65a8432` (Data balance with native resolution), `6a608b5` (New training binary)
- **Related run:** `runs/binary/B1_binary_B4_600x450_20261111_112623` (see `protocol.md`, run B1)

### Context

The V1 pipeline resized all images to 224 x 224, which discards fine-grained
dermoscopic detail and distorts the 4:3 aspect ratio. The goal of V2 was to
keep the original HAM10000 images untouched (600 x 450 px, W x H, copied byte
for byte, no JPEG re-encoding) and feed them to the network at that size.

### Decision

Feed images to EfficientNet-B4 at **600 x 450 (input tensor 450 x 600 x 3)**
instead of resizing to a fixed square input.

### Important caveat: this exceeds EfficientNet-B4's design resolution

EfficientNet-B4 was designed and pretrained (via compound scaling) for an input
of **380 x 380 px**. The native HAM10000 input used here is well above that:

| | Input (H x W) | Pixels | Final feature map (stride 32) |
|---|---|---|---|
| EfficientNet-B4 design resolution | 380 x 380 | 144,400 | 12 x 12 = 144 positions |
| This run (native HAM10000) | 450 x 600 | 270,000 | 15 x 19 = 285 positions |
| Ratio | | **~1.87x** | **~2x** |

Implications:

- The network's depth and width were scaled for 380 px; at ~1.87x the pixels,
  the receptive fields cover a relatively smaller fraction of the lesion than
  during ImageNet pretraining (a train/test resolution discrepancy with respect
  to the pretrained weights).
- The larger input adds more fine-grained, high-frequency detail (texture, hair,
  JPEG artifacts, ruler marks) that the model can memorize. With only 770
  melanoma training images, this makes overfitting more likely.
- `GlobalMaxPooling2D` over a ~2x larger feature map picks the single strongest
  activation out of 285 positions, which may amplify spurious local cues.
- Cost: ~3 min per epoch with batch size 16 and float32 (~59 min total run).

### Outcome (run B1)

- Validation AUC **0.7985**, AP 0.3251. At threshold 0.50: sensitivity 0.611
  (99/162 melanomas), specificity 0.803, precision 0.273, F1 0.377.
- **Probable overfitting:** train AUC climbed to 0.927 while validation AUC
  plateaued at ~0.80 (gap ~0.12 in the last epoch). Validation loss reached its
  minimum at epoch 6 (0.3130) and never improved after that. It oscillated
  between 0.31 and 0.52 while training loss kept falling.
- Fine-tuning (last 30 backbone layers, lr 1e-5) did not reduce the best
  validation loss: phase 2's best was 0.4111, above phase 1's 0.3130. It only
  raised validation AUC slightly (0.791 -> 0.798). The saved model comes from
  phase 2.

### Interpretation

Exceeding B4's 380 px design resolution is the **leading hypothesis** for the
overfitting. It is not yet proven, because other factors in this run may also
contribute:

1. The training split was **not balanced** (`data/splits-v2`, 11 % melanoma;
   only sqrt class weights were applied).
2. Early stopping and the decision threshold were selected on the **same
   validation set** used for reporting, so the metrics are optimistic.
3. Only light online augmentation was used (flip, rotation 0.2, zoom 0.1).

### Next steps

- Run a **controlled comparison** that changes only the input size: B4 at its
  design resolution (380 x 380, or a 4:3 equivalent such as 384 x 288), all
  other settings identical to B1.
- Optionally test a larger variant whose design resolution is closer to
  600 x 450 (B5: 456, B6: 528, B7: 600) as an alternative to downscaling.
- Repeat on `data/splits-v2-balanced` and report on the held-out test split.

---

## D-002 — Train EfficientNet-B4 at its design resolution (380 x 380)

- **Date:** 2026-11-11
- **Status:** Provisionally adopted as the B4 baseline; native resolution with
  a larger backbone (B5–B7) still to be evaluated
- **Related run:** `runs/binary/B2_binary_B4_380x380_20261111_144536` (see `protocol.md`, run B2)

### Context

Run B1 (D-001) trained B4 at the native 600 x 450, about 1.87x the pixels of
B4's 380 x 380 design resolution, and showed probable overfitting. B2 is the
controlled comparison proposed in D-001.

### Decision

Train B4 at **380 x 380 (input tensor 380 x 380 x 3)**. Every other setting is
identical to B1: same script (`scripts/train_binary_mel_native.py`), same data
(`data/splits-v2`), seed, hyperparameters, augmentation, and callbacks. The
only change is `--img-size 380 380`.

Condition to note: the 600 x 450 (4:3) images are resized on the fly to a
square 380 x 380 with area interpolation. This **does not preserve the aspect
ratio**, so lesions are slightly squashed horizontally (all 8,505 images were
resized).

### Outcome (B2 vs B1)

| Metric (validation) | B1 — 600 x 450 | B2 — 380 x 380 | Change |
|---|---|---|---|
| ROC AUC | 0.7985 | **0.8175** | +0.019 |
| Average Precision | 0.3251 | **0.3434** | +0.018 |
| Best val_loss (phase 1 / phase 2) | 0.3130 / 0.4111 | **0.3071 / 0.3788** | lower |
| Train AUC − val AUC (last epoch) | 0.927 − 0.805 = **0.12** | 0.896 − 0.830 = **0.07** | gap ~halved |
| Sensitivity / specificity at chosen threshold | 0.611 / 0.803 (thr 0.50) | 0.599 / 0.813 (thr 0.46) | similar |
| Sensitivity / specificity at thr 0.25 | 0.846 / 0.594 | 0.889 / 0.575 | better sensitivity |
| Total training time | 59:06 | **25:52** | ~2.3x faster |

- The script's automatic "possible overfitting" warning fired in B1 but
  **not in B2**.
- Validation curves were more stable (val_accuracy 0.77–0.88 after epoch 2,
  vs 0.71–0.87 in B1).
- As in B1, fine-tuning did not improve phase 1's best val_loss
  (0.3788 vs 0.3071), although validation AUC kept rising in phase 2
  (0.808 -> 0.830).

### Interpretation

B2 supports the hypothesis from D-001. Training B4 above its design resolution
contributed to the overfitting, and returning to 380 px reduced the generalization
gap while slightly improving ranking quality, at less than half the compute.

Caveats:

- The AUC improvement (+0.019) comes from a single seed and only 162
  validation melanomas. It is probably within run-to-run variance, so it is
  not yet a significant difference.
- The non-overfitting part of the problem remains: the imbalanced training split,
  threshold and early stopping selected on the reported validation set, and
  fine-tuning that does not lower val_loss.
- Square resizing distorts the aspect ratio. A 4:3 input near B4's pixel budget
  (e.g. 440 x 330 or 384 x 288) has not been tested.

### Next steps

- Repeat B1 and B2 with 2–3 seeds to check that the gap and AUC difference hold.
- Test B5–B7 at a resolution closer to native (B3 in `protocol.md`).
- Move to `data/splits-v2-balanced`, use a separate split for threshold
  selection, and report on the held-out test split.
