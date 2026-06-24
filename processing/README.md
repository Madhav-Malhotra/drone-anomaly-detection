# Preprocessing Pipeline

Converts raw UAV flight recordings into fixed-size log-mel spectrogram arrays
ready for autoencoder or classifier training.

## Input

```
dataset/Dataset/
├── normal/          <mission_type>/<recording_id>/ch1.wav
├── 1_broken/        <config>/<mission_type>/<recording_id>/ch1.wav  (not used — see note)
├── 2_broken/        <config>/<mission_type>/<recording_id>/ch1.wav
└── 3_4_broken/      <3_broken|4_broken>/<config>/<mission_type>/<recording_id>/ch1.wav
```

- 293 WAV files total, 16 kHz mono PCM, variable duration (38–109 s per file)
- Class counts: normal 78, 1_broken 72, 2_broken 71, 3_4_broken 72

> **Note on directory depth:** `normal` is one level shallower than broken classes
> (no config subdirectory). The loader must handle both depths.

## Steps

### 1. File discovery and label assignment

Walk `dataset/Dataset/` and assign integer labels:

| Directory    | Binary label | 4-class label |
|--------------|-------------|---------------|
| `normal`     | 0           | 0             |
| `1_broken`   | 1           | 1             |
| `2_broken`   | 1           | 2             |
| `3_4_broken` | 1           | 3             |

Collect a list of `(filepath, binary_label, class_label, recording_id)` tuples.
`recording_id` is the unique identifier used for train/test splitting (see step 4).

### 2. Segmentation

Each recording is a full flight (38–109 s). Slice into fixed-length windows **before**
feature extraction so segment boundaries align with the raw waveform:

- Window size: **1 s** (16 000 samples at 16 kHz)
- Hop (stride): **0.5 s** (50% overlap)
- Drop any trailing segment shorter than 1 s

This yields roughly 75–215 segments per file, ~25 000 segments total across all classes.

### 3. Feature extraction

We generate **two representations** from each segment to benchmark which works better
for this task. Both use the same FFT parameters so results are directly comparable.

Common FFT parameters (shared):

| Parameter   | Value | Rationale                                        |
|-------------|-------|--------------------------------------------------|
| `n_fft`     | 512   | ~32 ms frame; resolves UAV harmonics, fast on M4 |
| `hop_length`| 256   | 50% frame overlap                                |
| `fmin`      | 50 Hz | Below UAV fundamental (~80 Hz); avoids DC noise  |
| `fmax`      | 8000 Hz | Nyquist for 16 kHz (no meaningful data above)  |

#### 3a. Log-mel spectrogram

Apply 32 triangular mel filterbanks (logarithmically spaced) to the power spectrum,
then convert to decibels with `librosa.power_to_db(ref=np.max)`.

| Parameter | Value | Rationale                                            |
|-----------|-------|------------------------------------------------------|
| `n_mels`  | 32    | Matches Edge Impulse MFE defaults; fits Arduino RAM  |
| `power`   | 2.0   | Power spectrogram before mel conversion              |

Output shape per segment: **(32 mel bins × 32 time frames)**, 4 KB at float32.

#### 3b. Raw FFT magnitude spectrogram

Take the magnitude of the STFT directly, convert to decibels, and trim to the same
frequency range (50–8000 Hz). With `n_fft=512` this yields 247 usable frequency bins,
which we downsample to **32 bins** by averaging groups of adjacent bins so the output
shape matches the mel spectrogram for a fair model comparison.

Output shape per segment: **(32 bins × 32 time frames)**, 4 KB at float32.

#### Why compare?

Log-mel allocates more resolution to low frequencies where UAV propeller harmonics
live. Raw FFT uses linear spacing. For anomaly detection the autoencoder may learn
to ignore irrelevant high-frequency bins either way — benchmarking both will confirm
whether the mel transformation actually helps for this dataset.

### 4. Train / test split

Split **by recording** (not by segment) to prevent data leakage. Segments from
the same flight must not appear in both sets. Additionally, every propeller
configuration (e.g. 0111, 1011) must appear in both train and test — otherwise
poor performance on a test config could be due to it being unseen rather than
the model failing.

Use scikit-learn's `StratifiedGroupKFold` with:
- **Groups** = recording ID (keeps all segments from one flight together)
- **Stratify by** = `(class, config)` joint label (ensures each config appears on both sides)
- **Split**: 80% train / 20% test, fixed random seed 42

The tightest config is `1001` with 11 recordings → ~2 in test. Results for
individual configs with only 2 test recordings should be interpreted cautiously.

After splitting, flatten each set to a list of `(spectrogram, label)` pairs.

### 5. Normalization

Compute mean and std of all training spectrograms (pixel-wise across the 64×32
grid). Apply z-score normalization:

```
X_norm = (X - train_mean) / (train_std + 1e-6)
```

Fit on train only; apply the same statistics to test. Save `mean` and `std` arrays
alongside the dataset so inference can use them without the raw audio.

### 6. Save outputs

Serialize to disk as NumPy `.npz` files, one set per representation:

```
processing/
├── mel/
│   ├── train.npz   — arrays: X (N×32×32), y_binary (N,), y_class (N,)
│   ├── test.npz    — same structure
│   └── stats.npz   — arrays: mean (32×32), std (32×32)
└── fft/
    ├── train.npz
    ├── test.npz
    └── stats.npz
```

The same train/test recording split is used for both representations so results
are directly comparable.

## Output shape summary

| Array      | Shape          | dtype   |
|------------|----------------|---------|
| X          | (N, 32, 32)    | float32 |
| y_binary   | (N,)           | int8    |
| y_class    | (N,)           | int8    |
| mean / std | (32, 32)       | float32 |

## Dependencies

```
librosa>=0.10
numpy
scikit-learn   # for StratifiedGroupKFold / train_test_split
soundfile      # librosa backend for WAV loading
```
