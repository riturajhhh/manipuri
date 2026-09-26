# 🏔️ Mizo Emotion AI Studio

A state-of-the-art emotion detection system dedicated exclusively to the **Mizo** language (Latin script), powered by advanced multi-granularity subword morphological NLP and a calibrated soft-voting deep ensemble.

---

## ✨ Key Features

- **Exclusive Mizo Focus** — Fully optimized pipeline tailored specifically for the linguistic properties, particles, and morphology of the Mizo language.
- **Advanced Multi-Granularity Subword NLP** — Tri-level Feature Union combining:
  - **Word N-grams (1–3)**: Captures emotional phrases, idioms, and multi-word expressions.
  - **Word-Boundary Character N-grams (2–6)**: Captures prefixes, suffixes, intensifiers (`-tak`, `-lutuk`, `-em`), and lexical roots.
  - **Pure Character N-grams (3–6)**: Handles vowel-length variations, tonal indicators, and colloquial contractions.
- **Calibrated Soft-Voting Deep Ensemble**:
  - Multi-scale LinearSVC classifiers with probability calibration (`CalibratedClassifierCV`) across regularizations (C=0.3, 0.5, 0.7).
  - Regularized Ridge Classifier.
  - Multinomial Logistic Regression with balanced class weights.
  - Passive-Aggressive Classifier.
  - Temperature-scaled softmax probability distributions for smooth, reliable confidence scoring.
- **5 Emotion Classes**:
  - 🌈 **Joy** (*Hlimna / Lawmna*)
  - 🛡️ **Fear** (*Hlauhna / Huphurhna*)
  - 🌊 **Sadness** (*Lungngaihna / Khawharna*)
  - 🌋 **Anger** (*Thinrimna / Khakna*)
  - 😐 **Neutral** (*Ngaihsak loh / Pangngai*)
- **Explainable AI (XAI)** — Real-time word-level attribution heatmap revealing exactly which Mizo words and particles drove the prediction.
- **Multimodal Voice & Acoustic DSP** — Real-time microphone recording and audio file analysis with acoustic feature extraction (F0 pitch, RMS energy, spectral centroid, bandwidth, rolloff, ZCR).
- **Batch CSV Processing** — Bulk emotion classification for large Mizo text files with downloadable enriched CSV exports.
- **Dialogue & Conversational Timeline** — Track emotional shifts across multi-turn Mizo dialogues with valence trajectory mapping (-1.0 to +1.0).
- **Active Learning Feedback Loop** — In-app correction submissions saved directly to an SQLite datastore (`feedback.db`).

---

## 📊 Model Performance (Unseen 15% Stratified Holdout)

| Metric | Score | Details |
|--------|-------|---------|
| **Test Accuracy** | **80.70%** | Measured on strict unseen stratified test split |
| **Weighted F1-Score** | **0.8062** | Balanced across all 5 classes |
| **Macro F1-Score** | **0.7859** | Strong recall on minority classes |
| **Vocabulary Features** | **125,000+** | Multi-granularity subword tokens |
| **Inference Latency** | **< 5 ms** | Ultra-fast real-time inference |

### Per-Class Performance:
- **Joy**: Precision 85.0%, Recall 89.0%, F1 0.87
- **Anger**: Precision 83.0%, Recall 87.0%, F1 0.85
- **Fear**: Precision 84.0%, Recall 80.0%, F1 0.82
- **Sadness**: Precision 75.0%, Recall 71.0%, F1 0.73
- **Neutral**: Precision 66.0%, Recall 66.0%, F1 0.66

---

## 🔬 Benchmark: Pre-Trained Foundation Models vs. Native Subword Ensemble

During architecture selection, we empirically evaluated standard pre-trained foundation models directly on the Mizo dataset:

| Architecture | Model Backbone | Accuracy | Latency | Why it Underperforms |
| :--- | :--- | :--- | :--- | :--- |
| **Multilingual MiniLM** | `paraphrase-multilingual-MiniLM-L12-v2` | **53.52%** | ~45 ms | Tokenizer lacks Mizo vocabulary; splits words into fragmented sub-tokens. |
| **LaBSE** | `sentence-transformers/LaBSE` | **60.07%** | ~120 ms | Trained on 109 high-resource languages; zero native Mizo training data. |
| **CANINE-S** | `google/canine-s` (Char-Transformer) | *Impractical* | > 10,000 ms | Character Transformer is too computationally heavy on CPU for real-time app. |
| **Native Deep Ensemble (Ours)** | **Multi-Granularity Subword Morphological Ensemble** | **80.70%** | **< 4 ms** | Directly learns Mizo roots (`lunggai`, `hlim`, `hlau`) and emotional particles (`tak`, `lutuk`, `em`). |

---

## 🚀 Quick Start

### 1. Installation
```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 2. Train the Mizo Model
```bash
python models/mizo/train_mizo.py
```

### 3. Generate Confusion Matrix
```bash
python generate_confusion_matrices.py
```

### 4. Run the Studio Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📁 Repository Structure

```
new_manipuri/
├── app.py                      # Mizo Emotion AI Studio Dashboard
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
├── feedback.db                 # Active Learning SQLite database
├── cm_mizo.png                 # Test Set Confusion Matrix Image
├── datasets/
│   └── mizotext.csv            # Mizo Emotion Dataset (~5,195 samples)
├── models/
│   └── mizo/
│       ├── train_mizo.py       # SOTA Mizo Training Engine (v3)
│       └── mizo_ultra_final/   # Exported Model Artifacts
│           ├── ultra_pipeline.pkl
│           ├── label_encoder.pkl
│           └── metadata.json
├── audio_dsp.py                # Audio Emotion DSP feature extraction
└── xai_explainer.py            # Word-level attribution & heatmap generator
```

---

## 📝 Sample Mizo Test Sentences

- **Joy 🌈**: `ka pass dawn chiang lutuk` / `ka va hlim tak em ka puak dawn!`
- **Anger 🌋**: `ka thinrim lutuk` / `ka thil neih zawng zawng nen ka hua che !`
- **Sadness 🌊**: `ka va ngai dawn che ve le` / `ka hmu leh tawh dawn lo hi ka va lunggai em`
- **Fear 🛡️**: `ka hlau lutuk` / `thil hlauhawm tawn dawnin ka mur chum chum zel`
- **Neutral 😐**: `dawr ka kal dawn` / `tinge ?`
