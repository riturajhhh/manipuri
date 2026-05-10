import streamlit as st
import json
import os
import joblib
import numpy as np
import re

# Config
st.set_page_config(page_title="Multilingual Emotion AI", page_icon="🌏", layout="centered")

# Custom CSS for Beautiful Multilingual UI
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
    
    /* Language Badges */
    .lang-badge {
        display: inline-block;
        padding: 5px 18px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 1px;
        margin-bottom: 15px;
        text-transform: uppercase;
    }
    .lang-badge-manipuri {
        background: linear-gradient(135deg, #58a6ff33, #bc8cff33);
        color: #bc8cff;
        border: 1px solid #bc8cff44;
    }
    .lang-badge-assamese {
        background: linear-gradient(135deg, #f0883e33, #ffa65733);
        color: #ffa657;
        border: 1px solid #ffa65744;
    }
    .lang-badge-mizo {
        background: linear-gradient(135deg, #3fb95033, #56d36433);
        color: #56d364;
        border: 1px solid #56d36444;
    }

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

    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #8b949e;
        margin-top: -20px;
        margin-bottom: 30px;
        font-weight: 300;
        letter-spacing: 3px;
    }

    /* Model info chip */
    .model-chip {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 400;
        background: rgba(255,255,255,0.06);
        color: #8b949e;
        border: 1px solid rgba(255,255,255,0.08);
        margin-top: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# EMOTION METADATA (All languages combined)
# ============================================================
EMOTION_META = {
    'joy':      {'icon': '🌈', 'color': '#FFD700'},
    'sadness':  {'icon': '🌊', 'color': '#4A90E2'},
    'anger':    {'icon': '🌋', 'color': '#FF4B2B'},
    'fear':     {'icon': '🛡️', 'color': '#9B59B6'},
    'surprise': {'icon': '🌠', 'color': '#50E3C2'},
    'disgust':  {'icon': '🌿', 'color': '#8B572A'},
    'tired':    {'icon': '🔋', 'color': '#A5B1C2'},
    'proud':    {'icon': '🦁', 'color': '#F39C12'},
    'calm':     {'icon': '🧘', 'color': '#1ABC9C'},
    'lonely':   {'icon': '🌌', 'color': '#34495E'},
    'excited':  {'icon': '✨', 'color': '#E74C3C'},
    'love':     {'icon': '💖', 'color': '#FF69B4'},
    'neutral':  {'icon': '😐', 'color': '#95A5A6'},
    'trust':    {'icon': '🤝', 'color': '#27AE60'},
    'anticipation': {'icon': '⏳', 'color': '#E67E22'},
}

# ============================================================
# LANGUAGE DETECTION
# ============================================================
def detect_language(text):
    """Detect language from Unicode script ranges.
    
    - Meitei Mayek (U+ABC0-ABFF) → Manipuri
    - Bengali script (U+0980-09FF) → Assamese
    - Latin script → Mizo (since it's the only Latin-script language we support)
    """
    meitei_count = len(re.findall(r'[\uABC0-\uABFF]', text))
    bengali_count = len(re.findall(r'[\u0980-\u09FF]', text))
    latin_count = len(re.findall(r'[a-zA-Z]', text))
    
    if meitei_count > 0 and meitei_count >= bengali_count:
        return 'manipuri'
    elif bengali_count > 0:
        return 'assamese'
    elif latin_count > 0:
        return 'mizo'
    else:
        return 'unknown'

# ============================================================
# PREPROCESSING FUNCTIONS (per language)
# ============================================================
def clean_text_manipuri(text):
    """Clean Manipuri text (Meitei Mayek + Bengali script)."""
    if not isinstance(text, str): return ""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\u0980-\u09FF\uABC0-\uABFF\s\.\,\!\?\-]', '', text)
    return text.strip().lower()

def clean_text_assamese(text):
    """Clean Assamese text (Bengali script + dandas + apostrophes)."""
    if not isinstance(text, str): return ""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r"[^\u0980-\u09FF\u0964-\u0965\s\.\,\!\?\-\'\'\']", '', text)
    return text.strip()

def clean_text_mizo(text):
    """Clean Mizo text (Latin script)."""
    if not isinstance(text, str): return ""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r"[^a-zA-Z\s\.\,\!\?\-\']", '', text)
    return text.strip().lower()

# ============================================================
# MODEL LOADING
# ============================================================
BASE_DIR = os.path.dirname(__file__)

@st.cache_resource
def load_manipuri_model():
    """Load Manipuri classical ML model."""
    model_dir = os.path.join(BASE_DIR, "models", "manipuri", "manipuri_ultra_final")
    if not os.path.exists(model_dir):
        return None, None, None, None

    try:
        pipeline_path = os.path.join(model_dir, "ultra_pipeline.pkl")
        if not os.path.exists(pipeline_path):
            pipeline_path = os.path.join(model_dir, "classical_pipeline.pkl")

        pipeline = joblib.load(pipeline_path)
        le = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
        with open(os.path.join(model_dir, "metadata.json"), 'r') as f:
            config = json.load(f)
        return pipeline, le, config['classes'], config
    except Exception as e:
        st.error(f"Error loading Manipuri model: {e}")
        return None, None, None, None

@st.cache_resource
def load_assamese_classical():
    """Load Assamese classical ML model."""
    model_dir = os.path.join(BASE_DIR, "models", "assamese", "assamese_ultra_final")
    if not os.path.exists(model_dir):
        return None, None, None, None

    try:
        pipeline = joblib.load(os.path.join(model_dir, "ultra_pipeline.pkl"))
        le = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
        with open(os.path.join(model_dir, "metadata.json"), 'r') as f:
            config = json.load(f)
        return pipeline, le, config['classes'], config
    except Exception as e:
        st.error(f"Error loading Assamese classical model: {e}")
        return None, None, None, None

@st.cache_resource
def load_assamese_transformer():
    """Load Assamese transformer model (MuRIL fine-tuned)."""
    model_dir = os.path.join(BASE_DIR, "models", "assamese", "assamese_transformer_final")
    if not os.path.exists(model_dir):
        return None, None, None, None

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()
        le = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
        with open(os.path.join(model_dir, "metadata.json"), 'r') as f:
            config = json.load(f)
        return (model, tokenizer), le, config['classes'], config
    except ImportError:
        return None, None, None, None
    except Exception as e:
        st.error(f"Error loading Assamese transformer: {e}")
        return None, None, None, None

@st.cache_resource
def load_mizo_model():
    """Load Mizo classical ML model."""
    model_dir = os.path.join(BASE_DIR, "models", "mizo", "mizo_ultra_final")
    if not os.path.exists(model_dir):
        return None, None, None, None

    try:
        pipeline = joblib.load(os.path.join(model_dir, "ultra_pipeline.pkl"))
        le = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
        with open(os.path.join(model_dir, "metadata.json"), 'r') as f:
            config = json.load(f)
        return pipeline, le, config['classes'], config
    except Exception as e:
        st.error(f"Error loading Mizo model: {e}")
        return None, None, None, None

# ============================================================
# PREDICTION FUNCTIONS
# ============================================================
def predict_classical(pipeline, text, classes):
    """Predict using classical ML pipeline."""
    probs = pipeline.predict_proba([text])[0]
    idx = np.argmax(probs)
    return classes[idx].lower(), probs[idx], probs

def predict_transformer(model_tuple, text, classes):
    """Predict using transformer model."""
    import torch
    model, tokenizer = model_tuple

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=128, padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0].numpy()

    idx = np.argmax(probs)
    return classes[idx].lower(), probs[idx], probs

# ============================================================
# LOAD ALL MODELS
# ============================================================
manipuri_pipeline, manipuri_le, manipuri_classes, manipuri_config = load_manipuri_model()
assamese_pipeline, assamese_le, assamese_classes, assamese_config = load_assamese_classical()
assamese_transformer, assamese_t_le, assamese_t_classes, assamese_t_config = load_assamese_transformer()
mizo_pipeline, mizo_le, mizo_classes, mizo_config = load_mizo_model()

# Determine best Assamese model
use_transformer_for_assamese = assamese_transformer is not None

# ============================================================
# UI LAYOUT
# ============================================================
st.markdown("<h1 style='text-align: center; font-size: 55px; margin-bottom: 5px; background: -webkit-linear-gradient(#58a6ff, #bc8cff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Multilingual Emotion AI</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>MANIPURI &nbsp;•&nbsp; ASSAMESE &nbsp;•&nbsp; MIZO</div>", unsafe_allow_html=True)

st.write("")

# Language selection
st.markdown("<div class='instruction-text' style='margin-bottom: 10px;'>Enter text in Manipuri, Assamese, or Mizo</div>", unsafe_allow_html=True)
user_input = st.text_input("Input text", placeholder="e.g., ꯑꯩ ꯅꯪꯕꯨ ꯅꯨꯡꯁꯤ  |  মই বহুত আনন্দিত  |  ka va hlim tak", label_visibility="collapsed")

st.write("")

submitted = st.button("Detect Emotion")

# ============================================================
# HELPER: Render result card + detailed analysis
# ============================================================
def render_result(emotion, confidence, probs, classes_used, lang_label, lang_badge_class, model_type):
    """Render the emotion result card and detailed analysis bars."""
    meta = EMOTION_META.get(emotion, {'icon': '🔮', 'color': '#ffffff'})

    st.markdown(f"""
        <div class="emotion-card">
            <div class="lang-badge {lang_badge_class}">{lang_label}</div>
            <div class="emotion-icon">{meta['icon']}</div>
            <div class="emotion-title" style="color: {meta['color']}">{emotion}</div>
            <div style="color: #c9d1d9; font-size: 22px; margin-top: 10px; font-weight: 300;">Confidence: <b>{confidence:.1%}</b></div>
            <div class="model-chip">Model: {model_type}</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center; color: #8b949e; margin-bottom: 20px;'>Detailed Analysis</h3>", unsafe_allow_html=True)

    sorted_indices = np.argsort(probs)[::-1]
    cols = st.columns(2)
    for i, p_idx in enumerate(sorted_indices):
        prob = probs[p_idx]
        if prob > 0.01:
            label = classes_used[p_idx]
            with cols[i % 2]:
                st.progress(float(prob), text=f"{label.capitalize()}: {prob:.1%}")

# ============================================================
# PREDICTION LOGIC
# ============================================================
if submitted or user_input:
    if not user_input:
        st.warning("Please enter some text to analyze.")
    else:
        # Get language mode from sidebar state
        lang_option = st.session_state.get('lang_mode', 'Auto-Detect')

        if lang_option == 'Auto-Detect':
            detected_lang = detect_language(user_input)
        elif lang_option == 'Manipuri':
            detected_lang = 'manipuri'
        elif lang_option == 'Assamese':
            detected_lang = 'assamese'
        elif lang_option == 'Mizo':
            detected_lang = 'mizo'
        else:
            detected_lang = detect_language(user_input)

        # --- MANIPURI ---
        if detected_lang == 'manipuri':
            if manipuri_pipeline:
                cleaned = clean_text_manipuri(user_input)
                emotion, confidence, probs = predict_classical(manipuri_pipeline, cleaned, manipuri_classes)
                model_type = manipuri_config.get('type', 'classical') if manipuri_config else 'classical'
                render_result(emotion, confidence, probs, manipuri_classes,
                              "ꯃꯩꯇꯩ Manipuri", "lang-badge-manipuri", model_type)
            else:
                st.error("Manipuri model not found! Train with `python models/manipuri/train_manipuri.py`")

        # --- ASSAMESE ---
        elif detected_lang == 'assamese':
            if use_transformer_for_assamese:
                cleaned = clean_text_assamese(user_input)
                emotion, confidence, probs = predict_transformer(assamese_transformer, cleaned, assamese_t_classes)
                render_result(emotion, confidence, probs, assamese_t_classes,
                              "অসমীয়া Assamese", "lang-badge-assamese", "MuRIL Transformer")
            elif assamese_pipeline:
                cleaned = clean_text_assamese(user_input)
                emotion, confidence, probs = predict_classical(assamese_pipeline, cleaned, assamese_classes)
                model_type = assamese_config.get('type', 'classical') if assamese_config else 'classical'
                render_result(emotion, confidence, probs, assamese_classes,
                              "অসমীয়া Assamese", "lang-badge-assamese", model_type)
            else:
                st.error("❌ Assamese model not found! Train with `python train_assamese.py`")

        # --- MIZO ---
        elif detected_lang == 'mizo':
            if mizo_pipeline:
                cleaned = clean_text_mizo(user_input)
                emotion, confidence, probs = predict_classical(mizo_pipeline, cleaned, mizo_classes)
                model_type = mizo_config.get('type', 'classical') if mizo_config else 'classical'
                render_result(emotion, confidence, probs, mizo_classes,
                              "🏔️ Mizo", "lang-badge-mizo", model_type)
            else:
                st.error("❌ Mizo model not found! Train with `python train_mizo.py`")

        # --- UNKNOWN ---
        else:
            st.warning("⚠️ Could not detect language. Please select a language from the sidebar or enter text in Manipuri (ꯃꯩꯇꯩ ꯃꯌꯦꯛ), Assamese (অসমীয়া), or Mizo script.")

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png")
    st.markdown("### 🌏 Multilingual Emotion AI")

    st.markdown("---")

    # Language Mode
    lang_mode = st.radio(
        "Language Mode",
        ["Auto-Detect", "Manipuri", "Assamese", "Mizo"],
        index=0,
        help="Auto-detect identifies the script automatically. Override if needed."
    )
    st.session_state['lang_mode'] = lang_mode

    st.markdown("---")

    # Model Status
    st.markdown("### 📊 Model Status")

    if manipuri_pipeline:
        acc = manipuri_config.get('test_accuracy', 0) if manipuri_config else 0
        st.success(f"**Manipuri** ✅\nAccuracy: {acc:.1%}")
    else:
        st.error("**Manipuri** ❌ Not trained")

    if use_transformer_for_assamese:
        acc = assamese_t_config.get('test_accuracy', 0) if assamese_t_config else 0
        st.success(f"**Assamese (MuRIL)** ✅\nAccuracy: {acc:.1%}")
    elif assamese_pipeline:
        acc = assamese_config.get('test_accuracy', 0) if assamese_config else 0
        st.success(f"**Assamese (Classical)** ✅\nAccuracy: {acc:.1%}")
    else:
        st.warning("**Assamese** ⏳ Not trained yet")

    if mizo_pipeline:
        acc = mizo_config.get('test_accuracy', 0) if mizo_config else 0
        st.success(f"**Mizo** ✅\nAccuracy: {acc:.1%}")
    else:
        st.warning("**Mizo** ⏳ Not trained yet")

    st.markdown("---")

    # About
    st.markdown("### About")
    st.info("This app detects emotions from text in **Manipuri** (ꯃꯩꯇꯩ), **Assamese** (অসমীয়া), and **Mizo** using advanced ML/NLP. It auto-detects the input language and routes to the right model.")

    st.markdown("---")

    # Sample Phrases
    st.markdown("### 📝 Sample Phrases")

    st.markdown("**Manipuri (ꯃꯩꯇꯩ)**")
    st.code("ꯑꯩ ꯅꯪꯕꯨ ꯅꯨꯡꯁꯤ\n(Love/Joy)")
    st.code("ꯑꯩ ꯌꯥꯝꯅ ꯁꯥꯎꯕ ꯄꯤꯔꯤ\n(Anger)")
    st.code("ꯑꯁ! ꯃꯁꯤ ꯀꯔꯤꯅꯣ\n(Surprise)")

    st.markdown("**Assamese (অসমীয়া)**")
    st.code("মই বহুত আনন্দিত\n(Joy)")
    st.code("মোৰ বহুত দুখ লাগিছে\n(Sadness)")
    st.code("মই তোমাক ভাল পাওঁ\n(Love)")

    st.markdown("**Mizo**")
    st.code("ka va hlim tak\n(Joy)")
    st.code("ka va lungngai tak em\n(Sadness)")
    st.code("ka thinrim lutuk\n(Anger)")
    st.code("ka hlau lutuk\n(Fear)")
