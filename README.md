# 🌏 Multilingual Emotion AI

A multilingual emotion detection system supporting **Manipuri** (ꯃꯩꯇꯩ ꯃꯌꯦꯛ), **Assamese** (অসমীয়া), and **Mizo** languages, powered by advanced NLP/ML pipelines with automatic language detection.

---

## ✨ Features

- **Automatic Language Detection** — Detects Manipuri (Meitei Mayek), Assamese (Bengali script), and Mizo (Latin script) from input text
- **Multi-Model Architecture** — Separate optimized models per language
- **11 Emotion Classes** for Manipuri: joy, sadness, anger, fear, surprise, disgust, tired, proud, calm, lonely, excited
- **7 Emotion Classes** for Assamese: joy, sadness, anger, fear, trust, anticipation, neutral
- **5 Emotion Classes** for Mizo: joy, sadness, anger, fear, neutral
- **Optional Transformer Support** — Fine-tune MuRIL for Assamese for higher accuracy
- **Beautiful Glassmorphism UI** — Dark theme with language badges, confidence scores, and detailed analysis

---

## 🚀 Quick Start

### 1. Setup
```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 2. Train Models

**Manipuri** (already trained):
```bash
python models/manipuri/train_manipuri.py
```

**Assamese (Classical ML)**:
```bash
python models/assamese/train_assamese.py
```

**Mizo (Classical ML)**:
```bash
python models/mizo/train_mizo.py
```

**Assamese (Transformer)** — Higher accuracy, requires GPU (optional):
```bash
pip install torch transformers accelerate
python models/assamese/train_assamese_bert.py
```

### 3. Run the App
```bash
streamlit run app.py
```

---

## 📊 Model Performance

| Language | Model | Classes | Accuracy |
|----------|-------|---------|----------|
| Manipuri | BaggedSVM + TF-IDF | 11 | ~80% |
| Assamese | BaggedSVM + TF-IDF | 7 | ~83.3% |
| Mizo     | BaggedSVM + TF-IDF | 5 | ~80% |

---

## 📁 Project Structure

```
new_manipuri/
├── app.py                      # Main Trilingual Dashboard
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
├── datasets/                   # Centralized Datasets
│   ├── assamese_emotion.xlsx
│   ├── mizotext.csv
│   ├── manipuri_emotion_dataset_main1.xlsx
│   └── meitei_emotion_dataset_100k.csv
└── models/                     # Modular Models
    ├── assamese/
    │   ├── train_assamese.py
    │   ├── train_assamese_bert.py
    │   ├── assamese_ultra_final/
    │   └── assamese_transformer_final/
    ├── mizo/
    │   ├── train_mizo.py
    │   └── mizo_ultra_final/
    └── manipuri/
        ├── train_manipuri.py
        └── manipuri_ultra_final/
```

---

## 🛠️ Technology Stack

- **NLP**: TF-IDF (word + char n-grams), MuRIL Transformer
- **ML**: LinearSVC, BaggingClassifier, LogisticRegression, RidgeClassifier
- **Frontend**: Streamlit with custom CSS (glassmorphism, gradients)
- **Languages**: Python 3.10+

---

## 📝 Emotion Classes

### Manipuri (11 classes)
🌈 Joy · 🌊 Sadness · 🌋 Anger · 🛡️ Fear · 🌠 Surprise · 🌿 Disgust · 🔋 Tired · 🦁 Proud · 🧘 Calm · 🌌 Lonely · ✨ Excited

### Assamese (7 classes)
🌈 Joy · 🌊 Sadness · 🌋 Anger · 🛡️ Fear · 🤝 Trust · ⏳ Anticipation · 😐 Neutral

### Mizo (5 classes)
🌈 Joy · 🌊 Sadness · 🌋 Anger · 🛡️ Fear · 😐 Neutral
