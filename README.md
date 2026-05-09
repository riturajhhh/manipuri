# 🎭 Manipuri Emotion Detection

![Dashboard UI](demo.png)

A high-speed, state-of-the-art emotion detection system and interactive dashboard specifically optimized for the Manipuri language (Meitei Mayek & Bengali scripts). 

## 📌 Project Overview

Manipuri is an agglutinative language with complex morphology and unique script structures. This project tackles these challenges using a **Highly Optimized Classical Ensemble Pipeline** that delivers instant predictions without the massive overhead of Transformer models.

Through extensive iterative development, we combined a high-quality human-annotated dataset (~1,200 samples) with a massive 100K synthetic dataset. By applying rigorous deduplication, we eliminated cross-validation leakage and expanded the model's intelligence to recognize **11 distinct emotional states** with an impressive **~80% accuracy**.

### 🧠 Detected Emotions (11 Classes):
*   **Joy** (🌈)
*   **Sadness** (🌊)
*   **Anger** (🌋)
*   **Fear** (🛡️)
*   **Surprise** (🌠)
*   **Disgust** (🌿)
*   **Tired** (🔋)
*   **Proud** (🦁)
*   **Calm** (🧘)
*   **Lonely** (🌌)
*   **Excited** (✨)

---

## 🚀 Key Features

*   **Expanded 11-Class Detection:** Detects nuanced emotions far beyond standard NLP tools.
*   **Advanced Feature Engineering:** Utilizes a Hybrid FeatureUnion of word-level semantics (1-4 n-grams) and overlapping sub-word chunks (2-6 char n-grams) with extremely low thresholds (`min_df=1`) to capture rare but critical linguistic markers.
*   **Robust Ensemble Architecture:** Combines `BaggingClassifier` (with `LinearSVC`), `RidgeClassifier`, and `PassiveAggressiveClassifier` via Hard Voting to maximize generalization on highly sparse text matrices.
*   **Deduplicated & Augmented Data:** Combines multiple data sources into a clean, 1,300+ unique sample training set fortified with natural text augmentation (word duplication, swapping).
*   **Beautiful Premium UI:** A meticulously designed Streamlit dashboard featuring deep gradients, glassmorphism, floating micro-animations, and real-time probability breakdown bars.

---

## 🛠️ Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/riturajhhh/manipuri.git
    cd manipuri
    ```

2.  **Initialize Virtual Environment:**
    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install pandas numpy scikit-learn streamlit openpyxl joblib
    ```

---

## 🏃 Usage

### 1. Launching the Dashboard
To start the interactive UI and test your own sentences:
```bash
streamlit run app.py
```

### 2. Retraining the Intelligence Engine
If you add more data to the CSV/Excel files, you can retrain the entire 11-class ensemble:
```bash
python train_combined_v2.py
```
*This will process both datasets, run the algorithms, and save the optimized pipeline directly to the `manipuri_ultra_final` directory for the app to consume.*

---

## 📂 Project Structure

*   `app.py`: The premium Streamlit web dashboard.
*   `train_combined_v2.py`: The core training script (Data ingestion, Augmentation, Feature Extraction, Algorithm tuning).
*   `manipuri_emotion_dataset_main1.xlsx`: The original high-quality emotion dataset.
*   `meitei_emotion_dataset_100k.csv`: The expanded 100k emotion dataset.
*   `manipuri_ultra_final/`: Directory containing the serialized production models (`ultra_pipeline.pkl`, `label_encoder.pkl`).

---

## 🛡️ Model Performance
*   **Accuracy:** 79.70%
*   **F1-Score:** 80.02%
*   **Validation:** Rigorously tested on held-out clean data.

Developed with ❤️ for the Manipuri Language.
