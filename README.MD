**This README.MD has been made using AI**
It has not been checked on correctness. 


# Accent Classification with Deep Learning

This repository contains two main Jupyter notebooks for accent classification using deep learning on raw audio and spectrogram representations:

- **assignment_a.ipynb**: Approach A — Raw waveform classification (1D CNNs)
- **assignment_b_latest-2.ipynb**: Approach B — Spectrogram-based classification (2D CNNs)

Both approaches use PyTorch and torchaudio, and are designed for multi-class accent classification from short audio clips.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Data Description](#data-description)
- [Approach A: Raw Waveform Classification](#approach-a-raw-waveform-classification)
- [Approach B: Spectrogram Classification](#approach-b-spectrogram-classification)
- [Model Training & Evaluation](#model-training--evaluation)
- [Results & Findings](#results--findings)
- [Requirements & Setup](#requirements--setup)
- [How to Run](#how-to-run)
- [Saved Models](#saved-models)
- [References](#references)

---

## Project Structure

```
.
├── assignment_a.ipynb
├── assignment_b_latest-2.ipynb
├── Train/                # Training audio files (WAV)
├── Test set/             # Test audio files (WAV)
├── saved_models/
│   ├── A/                # Models from Approach A (1D CNNs)
│   └── B/                # Models from Approach B (2D CNNs)
├── EDA.ipynb             # Exploratory Data Analysis
└── ...
```

---

## Data Description

- **Audio files**: Located in `Train/` and `Test set/`, named as `[accent][gender]_[id].wav` (e.g., `1m_1234.wav`).
  - `accent`: 1–5 (class label)
  - `gender`: `m` (male) or `f` (female)
- **Sampling rate**: 16,000 Hz
- **Duration**: ~5 seconds per clip (min: ~1.7s, max: ~13s)
- **Speakers**: 10 (5 male, 5 female)

---

## Approach A: Raw Waveform Classification

- **Notebook**: `assignment_a.ipynb`
- **Input**: Raw audio waveforms (no augmentation)
- **Dataset class**: `AccentRawWaveformDataset`
- **Preprocessing**:
  - Resample to 16kHz if needed
  - Convert to mono
  - Standardize (zero mean, unit variance)
  - Pad/crop to fixed length for batching
- **Models**: Several 1D CNN variants:
  - Baseline
  - + BatchNorm
  - + Dropout (0.3, 0.5)
  - + BatchNorm & Dropout
- **Training**: 10 epochs, Adam optimizer, cross-entropy loss
- **Evaluation**: Overall and gender-specific metrics (accuracy, precision, recall, F1)

---

## Approach B: Spectrogram Classification

- **Notebook**: `assignment_b_latest-2.ipynb`
- **Input**: Spectrograms (optionally Mel-spectrograms)
- **Dataset class**: `AccentSpectrogramDataset`
- **Preprocessing**:
  - Resample to 16kHz if needed
  - Compute (Mel-)Spectrogram
  - Log scaling
  - Pad/crop to fixed width for batching
- **Models**: Several 2D CNN variants:
  - Baseline
  - + BatchNorm
  - + Dropout (0.3, 0.5)
  - + BatchNorm & Dropout
- **Training**: 10 epochs, Adam optimizer, cross-entropy loss
- **Evaluation**: Overall and gender-specific metrics (accuracy, precision, recall, F1)

---

## Model Training & Evaluation

- **Data split**: 80% train, 20% test (random, fixed seed)
- **Batch size**: 4 (adjustable)
- **Device**: Uses MPS (Apple Silicon) if available, else CPU
- **Metrics**: Printed per epoch and as classification reports

---

## Results & Findings

### Approach A (Raw Waveform, Example Results)
- **Best test accuracy**: ~21% (raw, no augmentation)
- **Gender breakdown**: Male: ~15%, Female: ~29%
- **Observation**: 1D CNNs on raw waveforms struggle to learn accent features.

### Approach B (Spectrogram, Example Results)
- **Best test accuracy**: ~87% (not augmented, 2D CNN with BatchNorm & Dropout)
- **Gender breakdown**: Male: ~85%, Female: ~84%
- **Observation**: 2D CNNs on spectrograms (especially with BatchNorm & Dropout) perform much better, capturing accent features more effectively.

*See the end of each notebook for detailed classification reports and per-class/gender breakdowns.*

---

## Requirements & Setup

**Main dependencies:**
- Python 3.11+
- torch (2.7.0)
- torchaudio
- numpy
- matplotlib
- pandas
- scikit-learn
- librosa (for EDA)

**Install with pip:**
```bash
pip install torch torchaudio numpy matplotlib pandas scikit-learn librosa
```

---

## How to Run

1. **Prepare data**: Place WAV files in `Train/` and `Test set/` as described above.
2. **Open notebook**: Use JupyterLab or VSCode to open either `assignment_a.ipynb` or `assignment_b_latest-2.ipynb`.
3. **Set data path**: Update the dataset path in the notebook if needed (default: `/Users/larsheijnen/DL/Train`).
4. **Run all cells**: Execute the notebook top-to-bottom to train and evaluate models.
5. **Saved models**: Trained models are saved in `saved_models/A/` and `saved_models/B/`.

---

## Saved Models

- **Approach A**: `saved_models/A/` (1D CNNs, raw waveform)
- **Approach B**: `saved_models/B/` (2D CNNs, spectrogram)
- **Variants**: Each folder contains models with/without augmentation, and with different regularization (BatchNorm, Dropout).

---

## References

- [PyTorch Documentation](https://pytorch.org/)
- [torchaudio Documentation](https://pytorch.org/audio/stable/index.html)
- [Librosa Documentation](https://librosa.org/doc/latest/index.html)

---

## Notes

- For best results, use Approach B (spectrograms + 2D CNNs).
- Data augmentation and further hyperparameter tuning may improve results.
- For reproducibility, random seeds are set for data splits.

---

