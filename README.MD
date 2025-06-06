# Accent Classification with Deep Learning

This repository contains two main Jupyter notebooks for accent classification using deep learning on raw audio and spectrogram representations:

- **assignment_a.ipynb**: Approach A — Raw waveform classification (1D CNNs)  
- **assignment_b_latest-2.ipynb**: Approach B — Spectrogram-based classification (2D CNNs)

Both approaches use PyTorch and torchaudio, and are designed for multi-class accent classification from short audio clips.

---

## 📁 Table of Contents

- [Project Structure](#project-structure)  
- [Dataset Description](#dataset-description)  
- [Approach A: Raw Waveform Classification](#approach-a-raw-waveform-classification)  
- [Approach B: Spectrogram Classification](#approach-b-spectrogram-classification)  
- [Model Training & Evaluation](#model-training--evaluation)  
- [Results & Findings](#results--findings)  
- [Setup & Installation](#setup--installation)  
- [How to Run](#how-to-run)  
- [Saved Models](#saved-models)  
- [References](#references)  

---

## 🗂️ Project Structure

```
.
├── assignment_a.ipynb
├── assignment_b_latest-2.ipynb
├── Train/                # Training audio files (WAV), excluded from Github
├── Test set/             # Test audio files (WAV), excluded from Github
├── saved_models/
│   ├── A/                # Models from Approach A (1D CNNs)
│   └── B/                # Models from Approach B (2D CNNs)
├── EDA.ipynb             # Exploratory Data Analysis
└── ...
```

---

## 🎧 Dataset Description

- **📁 Location**: All training data is in `Train/`. Test files are in `Test set/`.
- **🔤 Naming Convention**: Files are named `[accent][gender]_[id].wav` (e.g., `1m_1234.wav`)
  - `accent`: 1–5 (class label)  
  - `gender`: `m` (male) or `f` (female)
- **🎼 Audio Properties**:
  - Total clips: 3,166  
  - Sampling rate: 16,000 Hz  
  - Duration: ~1.7s to ~13s (avg. ~5.3s)  
  - Speakers: 10 (5 male, 5 female)
- **🧪 Train/Validation Split**:
  - 80% training, 20% validation  
- **🚫 Note**: Files in `Test set/` are not used during training; their naming may not match expectations.

---

## 🧪 Approach A: Raw Waveform Classification

- **📓 Notebook**: `assignment_a.ipynb`  
- **🎧 Input**: Raw waveforms  
- **⚙️ Preprocessing**:
  - Resample to 16kHz  
  - Convert to mono  
  - Standardize (zero mean, unit variance)  
  - Pad/crop to fixed length  
- **🧠 Model Variants**:
  - Baseline  
  - + BatchNorm  
  - + Dropout (0.3, 0.5)  
  - + BatchNorm & Dropout  
- **Training**: 10 epochs (report: up to 150), Adam optimizer, cross-entropy loss  
- **Evaluation**: Accuracy, precision, recall, F1-score, gender-specific metrics

---

## 🧪 Approach B: Spectrogram Classification

- **📓 Notebook**: `assignment_b_latest-2.ipynb`  
- **🖼️ Input**: Mel-spectrograms  
- **⚙️ Preprocessing**:
  - Resample to 16kHz  
  - Compute Mel-spectrogram  
  - Apply log-scaling  
  - Pad/crop to fixed width  
- **🧠 Model Variants**:
  - Baseline  
  - + BatchNorm  
  - + Dropout (0.3, 0.5)  
  - + BatchNorm & Dropout  
- **Training**: 10–150 epochs, Adam optimizer, cross-entropy loss  
- **Evaluation**: Full classification reports by accent and gender

---

## 📈 Model Training & Evaluation

- **📊 Splits**: 80% train, 20% validation  
- **⚙️ Batch Size**: 4  
- **🖥️ Device**: Uses MPS (Apple Silicon) if available, otherwise CPU  
- **📊 Metrics**:
  - Accuracy, precision, recall, F1-score  
  - Gender-specific performance  
  - Error analysis by class

---

## 🏁 Results & Findings

### Approach A (1D CNN on Raw Waveforms)
- **Best Validation Accuracy**: ~21%  
- **Observation**: Struggled to learn directly from waveforms  
- **Gender Breakdown**: Male ~15%, Female ~29%  

### Approach B (2D CNN on Mel-Spectrograms)
- **Best Validation Accuracy**: ~87%  
- **Observation**: Spectrogram-based models performed significantly better  
- **Gender Breakdown**: Male ~85%, Female ~84%

*See the notebooks for detailed classification reports.*

---

## ⚙️ Setup & Installation

Main dependencies:
- Python 3.11+  
- torch, torchaudio  
- numpy, pandas, matplotlib  
- scikit-learn, librosa, mutagen  

**Install with pip**:

```bash
# (Recommended) Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchaudio numpy pandas matplotlib scikit-learn librosa mutagen
```

---

## ▶️ How to Run

1. **Place Data**: Ensure all `.wav` files are in `Train/`  
2. **Open Notebook**: Use JupyterLab or VSCode to open either notebook  
3. **Check Paths**: Confirm data paths match (default is relative paths)  
4. **Run Cells**: Top to bottom — load data, define models, train, evaluate  
5. **Output**: Trained models saved in `saved_models/`  

---

## 💾 Saved Models

- `saved_models/A/`: 1D CNNs (raw waveform, Approach A)  
- `saved_models/B/`: 2D CNNs (spectrograms, Approach B)  
- Includes variants with/without BatchNorm and Dropout

---

## 🔗 References

- [PyTorch](https://pytorch.org/)  
- [torchaudio](https://pytorch.org/audio/stable/index.html)  
- [Librosa](https://librosa.org/doc/latest/index.html)

---

## 📝 Notes

- For best performance: use Approach B with BatchNorm + Dropout  
- Data augmentation improves results (see report)  
- Reproducibility: fixed random seeds used for data splits  