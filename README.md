# 🎭 Manipuri Emotion Detection System

A high-accuracy machine learning system and interactive Streamlit dashboard designed to detect emotions from Manipuri sentences written in the Meitei Mayek script.

## 📌 Overview

Off-the-shelf NLP tools often struggle with agglutinative languages like Manipuri because of complex morphology and unique script structures. This project solves that problem by using an **Advanced Character N-Gram TF-IDF Vectorizer** combined with a **Calibrated Linear Support Vector Classifier (LinearSVC)**.

This approach effectively bypasses standard word-boundary limitations, allowing the model to learn the mathematical "roots" and structural patterns of emotional words achieving over **90%+ training accuracy**.

### Detected Emotions:
* Joy (ꯅꯨꯡꯉꯥꯏꯕꯥ)
* Sadness (ꯑꯋꯥꯕꯥ)
* Anger (ꯁꯥꯎꯕꯥ)
* Fear (ꯑꯀꯤꯕꯥ)
* Surprise (ꯑꯉꯛꯄꯥ)
* Disgust (ꯌꯥꯊꯤꯅꯥ)

---

## 🚀 Features

- **Custom Text Processing:** Extracts deep character-level overlapping chunks (2-10 n-grams) rather than full words, making it robust against typos and morphological variations.
- **Probability Calibration:** The LinearSVC is wrapped in a `CalibratedClassifierCV` to provide real-world confidence scores (e.g., "85% Joy") rather than just raw distances.
- **Premium UI:** A fully interactive, responsive, dark-themed Streamlit dashboard for real-time predictions.
- **Optimized for Small Data:** Uses strong regularization techniques (`C=1.0`, `sublinear_tf`) to prevent overfitting on the limited (~1,100 samples) dataset.

---

## 🛠️ Installation

1. **Clone or Download** the repository to your local machine.
2. **Open Terminal / Command Prompt** inside the project folder.
3. **Install Dependencies:**
   ```bash
   pip install pandas numpy scikit-learn streamlit openpyxl
   ```

---

## 🏃 Usage

### 1. Running the Dashboard (Real-Time Predictions)
To start the user interface and predict emotions interactively:

```bash
streamlit run app.py
```
This will open a browser window (usually at `http://localhost:8502`). Simply type a Manipuri sentence in the text box and hit "Predict Emotion".

### 2. Re-training the Model (Adding New Data)
If you update the dataset (`manipuri_emotion_dataset_main1.xlsx`) with new sentences to improve accuracy, you must re-train the model. 

```bash
python train.py
```
This script will:
1. Load your new data.
2. Extract the updated TF-IDF features.
3. Train the One-Vs-Rest LinearSVC.
4. Automatically save the updated brain into the `manipuri_emotion_model/` folder for the dashboard to use.

---

## 📂 Project Structure

* `app.py`: The Streamlit web dashboard application.
* `train.py`: The machine learning training pipeline.
* `manipuri_emotion_dataset_main1.xlsx`: The raw dataset containing Manipuri text and corresponding emotion labels.
* `manipuri_emotion_model/`: Auto-generated folder containing the frozen, serialized model files (`model.pkl`, `vectorizer.pkl`, `label_encoder.pkl`) loaded by `app.py`.
