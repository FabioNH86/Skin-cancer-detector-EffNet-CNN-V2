# Decision Log

Record of design decisions, their rationale, and their observed consequences.
Each entry is dated and references the run(s) that support it.

---

## D-001 — Train EfficientNet-B4 at HAM10000 native resolution (600 x 450)

- **Date:** 2026-11-11
- **Status:** Tested in run B1 — result **not satisfactory**; under review (see D-002)
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

## D-002 — Pending: choose input resolution for the binary MEL model

- **Date:** 2026-11-11
- **Status:** Open
- **Question:** Should the binary model be trained at B4's design resolution
  (380 px) or at native resolution with a larger backbone (B5–B7)?
- **Decide after:** the controlled comparison described in D-001.
