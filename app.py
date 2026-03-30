import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os
import pickle
import numpy as np
from sklearn.pipeline import FeatureUnion

# Config
st.set_page_config(page_title="Manipuri Emotion Detector", page_icon="🎭", layout="centered")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stTextInput > div > div > input {
        background-color: #1e2530;
        color: white;
        border-radius: 10px;
        border: 1px solid #3e4b5b;
    }
    .emotion-card {
        padding: 20px;
        border-radius: 15px;
        background: linear-gradient(135deg, #2c3e50 0%, #000000 100%);
        border: 1px solid #4a5568;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .joy { border-left: 5px solid #ffcc00; }
    .sadness { border-left: 5px solid #3498db; }
    .anger { border-left: 5px solid #e74c3c; }
    .fear { border-left: 5px solid #9b59b6; }
    .surprise { border-left: 5px solid #1abc9c; }
    .disgust { border-left: 5px solid #e67e22; }
    
    .emotion-title { font-size: 24px; font-weight: bold; margin-bottom: 5px; }
    .emotion-confidence { font-size: 16px; opacity: 0.8; }
    </style>
""", unsafe_allow_html=True)

# LOAD MODEL
MODEL_DIR = "./manipuri_emotion_model"

@st.cache_resource
def load_model():
    if not os.path.exists(os.path.join(MODEL_DIR, "model.pkl")):
        return None, None, None, None
    
    # Load components
    with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), 'rb') as f:
        vectorizer = pickle.load(f)
        
    with open(os.path.join(MODEL_DIR, "model.pkl"), 'rb') as f:
        model = pickle.load(f)
        
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), 'rb') as f:
        le = pickle.load(f)
        
    with open(os.path.join(MODEL_DIR, "config.json"), 'r') as f:
        config = json.load(f)
        
    return vectorizer, model, le, config['classes']

vectorizer, model, le, classes = load_model()

# Header
st.title("🎭 Manipuri Emotion Detector")
st.markdown("Developed with Advanced N-gram TF-IDF & LinearSVC for optimized accuracy.")

# Input
user_input = st.text_input("Enter a Manipuri sentence:", placeholder="e.g. ꯑꯩ ꯌꯥꯝꯅ ꯍꯛꯄ ꯐꯥꯔꯦ")

if st.button("Predict Emotion", type="primary"):
    if not user_input:
        st.warning("Please enter some text.")
    elif model is None:
        st.error("Model not found. Please train the model first.")
    else:
        # Prediction
        features = vectorizer.transform([user_input])
        probs = model.predict_proba(features)
        predicted_idx = np.argmax(probs, axis=1)[0]
        confidence = probs[0][predicted_idx]
        
        emotion = le.classes_[predicted_idx]
        conf_val = float(confidence.item())
        
        # Result Card
        st.markdown(f"""
            <div class="emotion-card {emotion.lower()}">
                <div class="emotion-title">{emotion.upper()}</div>
                <div class="emotion-confidence">Confidence: {conf_val:.2%}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Breakdown
        st.subheader("Probability Breakdown")
        for i, prob in enumerate(probs[0]):
            label = le.classes_[i]
            st.progress(float(prob), text=f"{label}: {float(prob):.1%}")

# Sidebar
with st.sidebar:
    st.subheader("About the Model")
    st.info("""
    This model uses an **Advanced N-gram TF-IDF & LinearSVC Ensemble** optimized for the Meitei Mayek script.
    
    It is trained on an emotion-balanced dataset to provide high-accuracy predictions by capturing sub-word patterns unique to Manipuri.
    """)
    if st.checkbox("Show Sample Phrases"):
        samples = {
            "Joy": "ꯑꯩ ꯌꯥꯝꯅ ꯍꯛꯄ ꯐꯥꯔꯦ",
            "Anger": "ꯑꯩ ꯌꯥꯝꯅ ꯁꯥꯎꯕ ꯄꯤꯔꯤ",
            "Sadness": "ꯑꯩ ꯋꯥꯌꯦ",
            "Fear": "ꯑꯩ ꯀꯤꯔꯤ",
            "Surprise": "ꯑꯁ! ꯃꯁꯤ ꯀꯔꯝꯅ ꯑꯣꯏꯔꯤꯕꯅꯤ"
        }
        for k, v in samples.items():
            st.code(f"{k}: {v}")
