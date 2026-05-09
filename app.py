import streamlit as st
import json
import os
import joblib
import numpy as np
import re

# Config
st.set_page_config(page_title="Manipuri Emotion Detection", page_icon="🎭", layout="centered")

# Custom CSS for Beautiful UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] { 
        font-family: 'Outfit', sans-serif; 
    }
    
    /* Background Gradient */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(20, 26, 35) 0%, rgb(8, 12, 17) 90%);
        color: #e6edf3;
    }

    /* Input Field Styling */
    .stTextInput input { 
        font-size: 18px !important;
    }
    
    /* Emotion Result Card (Glassmorphism) */
    .emotion-card { 
        padding: 40px; 
        border-radius: 30px; 
        background: rgba(255, 255, 255, 0.04); 
        backdrop-filter: blur(25px); 
        -webkit-backdrop-filter: blur(25px);
        border: 1px solid rgba(255, 255, 255, 0.1); 
        text-align: center; 
        margin-top: 30px;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }

    .emotion-icon { font-size: 90px; margin-bottom: 10px; text-shadow: 0 0 20px rgba(255,255,255,0.2); }
    .emotion-title { font-size: 45px; font-weight: 800; letter-spacing: 2px; text-transform: uppercase; }
    
    /* Button Styling */
    .stButton > button { 
        width: 100%; 
        background: linear-gradient(135deg, #238636, #2ea043); 
        color: white; 
        border: none; 
        padding: 15px; 
        border-radius: 20px; 
        font-weight: 600; 
        font-size: 20px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(46, 160, 67, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(46, 160, 67, 0.6);
        background: linear-gradient(135deg, #2ea043, #3fb950); 
    }
    
    /* Progress Bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #58a6ff, #bc8cff);
    }
    
    /* Instruction Text */
    .instruction-text {
        font-size: 20px;
        font-weight: 400;
        color: #8b949e;
        margin-bottom: -15px;
        margin-left: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# EMOTION METADATA (11 Classes)
EMOTION_META = {
    'joy': {'icon': '🌈', 'color': '#FFD700'},
    'sadness': {'icon': '🌊', 'color': '#4A90E2'},
    'anger': {'icon': '🌋', 'color': '#FF4B2B'},
    'fear': {'icon': '🛡️', 'color': '#9B59B6'},
    'surprise': {'icon': '🌠', 'color': '#50E3C2'},
    'disgust': {'icon': '🌿', 'color': '#8B572A'},
    'tired': {'icon': '🔋', 'color': '#A5B1C2'},
    'proud': {'icon': '🦁', 'color': '#F39C12'},
    'calm': {'icon': '🧘', 'color': '#1ABC9C'},
    'lonely': {'icon': '🌌', 'color': '#34495E'},
    'excited': {'icon': '✨', 'color': '#E74C3C'}
}

# PREPROCESSING FUNCTION
def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\u0980-\u09FF\uABC0-\uABFF\s\.\,\!\?\-]', '', text)
    return text.strip().lower()

# LOAD MODEL
MODEL_DIR = "./manipuri_ultra_final"

@st.cache_resource
def load_classical_model():
    if not os.path.exists(MODEL_DIR): return None, None, None
    try:
        pipeline = joblib.load(os.path.join(MODEL_DIR, "ultra_pipeline.pkl"))
        le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
        with open(os.path.join(MODEL_DIR, "metadata.json"), 'r') as f:
            config = json.load(f)
        return pipeline, le, config['classes']
    except: return None, None, None

pipeline, le, classes = load_classical_model()

# UI Layout
st.markdown("<h1 style='text-align: center; font-size: 55px; margin-bottom: 30px; background: -webkit-linear-gradient(#58a6ff, #bc8cff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Manipuri Emotion Detection</h1>", unsafe_allow_html=True)

st.write("") # Extra spacing to bring it down
st.write("") # Extra spacing to bring it down

st.markdown("<div class='instruction-text' style='margin-bottom: 10px;'>Enter a Manipuri sentence</div>", unsafe_allow_html=True)
user_input = st.text_input("Input text", placeholder="e.g., ꯑꯩ ꯅꯪꯕꯨ ꯅꯨꯡꯁꯤ", label_visibility="collapsed")

st.write("") # Spacing

submitted = st.button("Detect Emotion")

if submitted or user_input:
    if not user_input:
        st.warning("Please enter some text to analyze.")
    else:
        if pipeline:
            # Predict
            probs = pipeline.predict_proba([user_input])[0]
            idx = np.argmax(probs)
            emotion = classes[idx].lower()
            confidence = probs[idx]

            meta = EMOTION_META.get(emotion, {'icon': '🔮', 'color': '#ffffff'})
            
            # Result Card
            st.markdown(f"""
                <div class="emotion-card">
                    <div class="emotion-icon">{meta['icon']}</div>
                    <div class="emotion-title" style="color: {meta['color']}">{emotion}</div>
                    <div style="color: #c9d1d9; font-size: 22px; margin-top: 10px; font-weight: 300;">Confidence: <b>{confidence:.1%}</b></div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<h3 style='text-align: center; color: #8b949e; margin-bottom: 20px;'>Detailed Analysis</h3>", unsafe_allow_html=True)
            
            # Sort probabilities for display
            sorted_indices = np.argsort(probs)[::-1]
            
            cols = st.columns(2)
            for i, p_idx in enumerate(sorted_indices):
                prob = probs[p_idx]
                if prob > 0.01: # Only show significant probabilities
                    label = classes[p_idx]
                    with cols[i % 2]:
                        st.progress(float(prob), text=f"{label.capitalize()}: {prob:.1%}")
        else:
            st.error("Model not found! Please train the model first.")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png")
    st.markdown("### About")
    st.info("This application uses a highly optimized Machine Learning pipeline to detect 11 different emotions from Manipuri text.")
    
    st.markdown("### Sample Phrases")
    st.code("ꯑꯩ ꯅꯪꯕꯨ ꯅꯨꯡꯁꯤ\n(Love/Joy)")
    st.code("ꯑꯩ ꯌꯥꯝꯅ ꯁꯥꯎꯕ ꯄꯤꯔꯤ\n(Anger)")
    st.code("ꯑꯁ! ꯃꯁꯤ ꯀꯔꯤꯅꯣ\n(Surprise)")
    st.code("ꯑꯩ ꯊꯥꯛꯂꯤ\n(Tired)")
