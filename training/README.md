# Training

Three notebooks covering model training and edge deployment preparation.
All notebooks load from `processing/mel/` or `processing/fft/` and save
checkpoints to `training/models/`.

## Inputs (from preprocessing)

| File | Shape | Notes |
|------|-------|-------|
| `{rep}/train.npz` | X: (24801, 32, 32) float32 | 4 balanced classes |
| `{rep}/test.npz`  | X: (6047, 32, 32) float32  | same class proportions |
| `{rep}/stats.npz` | mean/std: (32, 32) float32 | for inference normalization |

`{rep}` is either `mel` or `fft`. Run each notebook on both representations
to compare which works better.

---

## Notebook 1 — `autoencoder.ipynb`

**Goal:** detect any anomaly (normal vs. broken) without requiring labelled
anomaly examples at training time. Train only on normal segments; flag inputs
whose reconstruction error exceeds a threshold.

### Architecture
Conv autoencoder operating on the 32×32 input as a 2D image:

```
Encoder: Conv2d(1→16) → Conv2d(16→32) → flatten → Dense(bottleneck=64)
Decoder: Dense(64→...) → reshape → ConvTranspose2d(32→16) → ConvTranspose2d(16→1)
```

Bottleneck size (64) is a hyperparameter to tune — smaller forces more
compression but may lose discriminating features.

### Training procedure
1. Filter train set to normal segments only (~6 773 samples)
2. Train to minimise MSE reconstruction loss
3. Compute reconstruction error on the full test set (normal + broken)
4. Choose threshold: e.g. mean + 2×std of normal reconstruction errors
5. Evaluate binary classification metrics at that threshold

### Outputs saved to `models/`
- `autoencoder_mel.keras` / `autoencoder_fft.keras` — full model
- `autoencoder_mel_threshold.npy` — scalar threshold value

### Key metrics to report
- AUROC (threshold-independent)
- Precision/recall/F1 at chosen threshold
- Reconstruction error distributions plotted for normal vs. broken (visual sanity check)

---

## Notebook 2 — `classifier.ipynb`

**Goal:** 4-class classification: normal / 1-broken / 2-broken / 3-4-broken.
Uses all labelled segments.

### Architecture
Small CNN suited to 32×32 input and eventual quantisation to 8-bit for Arduino:

```
Conv2d(1→16, 3×3) → BN → ReLU → MaxPool(2×2)
Conv2d(16→32, 3×3) → BN → ReLU → MaxPool(2×2)
Flatten → Dense(64) → Dropout(0.3) → Dense(4, softmax)
```

Total parameters target: <50k to leave room for quantisation overhead in 256 KB RAM.

### Training procedure
1. Use full train set with `y_class` labels
2. Cross-entropy loss, Adam optimiser, early stopping on val loss
3. 10% of train set held out as validation (split by recording, same leakage rules)
4. Train on both `mel` and `fft` representations; pick the better one for deployment

### Outputs saved to `models/`
- `classifier_mel.keras` / `classifier_fft.keras`

### Key metrics to report
- Per-class accuracy and confusion matrix
- Macro F1 (penalises poor performance on any single class equally)

---

## Notebook 3 — `quantise.ipynb`

**Goal:** compress the best classifier (and optionally autoencoder) to INT8
for deployment on the Arduino Nano 33 BLE Sense (256 KB RAM, Cortex-M4 @ 64 MHz).

### Steps

1. **Post-training quantisation (PTQ)**
   Convert the saved `.keras` model to TensorFlow Lite with full INT8 quantisation
   using a representative dataset (100–200 normal train segments as calibration data).
   Target: model fits in ~100 KB leaving headroom for activations and input buffer.

2. **Evaluate quantised model**
   Run the `.tflite` model on the test set using the TFLite interpreter.
   Compare accuracy vs. the float32 baseline — acceptable degradation: <2% macro F1.

3. **Quantisation-aware training (QAT) — if PTQ degrades too much**
   Re-train with `tf.keras.quantization` fake-quant nodes inserted, then re-export
   to TFLite INT8. QAT typically recovers 1–2% accuracy lost in PTQ.

4. **Measure model size and estimated RAM**
   - `.tflite` file size = flash usage
   - Peak activation memory = largest intermediate tensor (can be estimated with
     TFLite profiler or `tflite-micro` arena size tool)
   - Target: flash <150 KB, peak RAM <100 KB (leaves ~156 KB for stack + OS)

### Outputs saved to `models/`
- `classifier_ptq.tflite` — post-training quantised model
- `classifier_qat.tflite` — QAT model (if PTQ insufficient)

---

## Folder structure

```
training/
├── README.md
├── autoencoder.ipynb
├── classifier.ipynb
├── quantise.ipynb
├── models/
│   ├── autoencoder_mel.keras
│   ├── autoencoder_fft.keras
│   ├── autoencoder_mel_threshold.npy
│   ├── classifier_mel.keras
│   ├── classifier_fft.keras
│   ├── classifier_ptq.tflite
│   └── classifier_qat.tflite        (if needed)
└── plots/
    ├── autoencoder_mel_loss.png
    ├── autoencoder_mel_error_dist.png
    ├── autoencoder_mel_pr_curve.png
    ├── autoencoder_fft_loss.png
    ├── autoencoder_fft_error_dist.png
    ├── autoencoder_fft_pr_curve.png
    ├── classifier_mel_confusion.png
    ├── classifier_mel_pr_curve.png
    ├── classifier_fft_confusion.png
    └── classifier_fft_pr_curve.png
```

## Dependencies

```
tensorflow>=2.15      # Keras 3, TFLite converter, QAT support
numpy
matplotlib
scikit-learn          # metrics
```

Install into the existing `.venv`:
```bash
uv pip install tensorflow
```
