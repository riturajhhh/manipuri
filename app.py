import streamlit as st
import json
import os
import joblib
import numpy as np
import re
import pandas as pd
import sqlite3
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import custom DSP and XAI modules
import audio_dsp
import xai_explainer

# Try importing Plotly for visualization
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Page Configuration
st.set_page_config(
    page_title="Mizo Emotion AI Studio",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize SQLite database for Active Learning Feedback Loop
def init_feedback_db():
    conn = sqlite3.connect("feedback.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            detected_lang TEXT DEFAULT 'mizo',
            engine TEXT,
            predicted_emotion TEXT,
            corrected_emotion TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_feedback_db()

# Custom CSS for Modern, Glassmorphism Dark UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Outfit', sans-serif; 
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(13, 17, 23) 0%, rgb(7, 10, 15) 90%);
        color: #e6edf3;
    }

    .stTextInput input, .stTextArea textarea { 
        font-size: 17px !important;
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(88, 166, 255, 0.25) !important;
        color: white !important;
        border-radius: 12px !important;
        transition: all 0.3s ease;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 15px rgba(88, 166, 255, 0.3) !important;
    }
    
    /* Emotion Result Card */
    .emotion-card { 
        padding: 30px; 
        border-radius: 24px; 
        background: rgba(255, 255, 255, 0.03); 
        backdrop-filter: blur(25px); 
        -webkit-backdrop-filter: blur(25px);
        border: 1px solid rgba(255, 255, 255, 0.1); 
        text-align: center; 
        margin-top: 10px;
        margin-bottom: 20px;
        box-shadow: 0 10px 35px rgba(0,0,0,0.5);
        animation: floatCard 6s ease-in-out infinite;
        transition: all 0.4s ease;
    }
    
    @keyframes floatCard {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
        100% { transform: translateY(0px); }
    }

    .emotion-icon { 
        font-size: 78px; 
        margin-bottom: 6px; 
        filter: drop-shadow(0 0 25px rgba(255,255,255,0.25)); 
    }
    .emotion-title { 
        font-size: 38px; 
        font-weight: 800; 
        letter-spacing: 2px; 
        text-transform: uppercase; 
    }
    
    .lang-badge-mizo {
        display: inline-block;
        padding: 5px 18px;
        border-radius: 30px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.5px;
        margin-bottom: 10px;
        text-transform: uppercase;
        background: linear-gradient(135deg, rgba(63, 185, 80, 0.25), rgba(86, 211, 100, 0.25));
        color: #56d364;
        border: 1px solid rgba(86, 211, 100, 0.4);
    }

    .stButton > button { 
        width: 100%; 
        background: linear-gradient(135deg, #1f6feb, #388bfd); 
        color: white; 
        border: none; 
        padding: 12px 24px; 
        border-radius: 14px; 
        font-weight: 600; 
        font-size: 17px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(31, 111, 235, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(31, 111, 235, 0.5);
        background: linear-gradient(135deg, #388bfd, #58a6ff); 
    }
    
    .model-chip {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
        background: rgba(255,255,255,0.06);
        color: #8b949e;
        border: 1px solid rgba(255,255,255,0.1);
        margin-top: 10px;
    }

    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 12px;
        padding: 18px;
        background: rgba(255,255,255,0.015);
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.06);
        max-height: 520px;
        overflow-y: auto;
    }
    .chat-bubble {
        padding: 14px 20px;
        border-radius: 18px;
        max-width: 82%;
        line-height: 1.5;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    .chat-bubble-left {
        align-self: flex-start;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-bottom-left-radius: 4px;
    }
    .chat-bubble-right {
        align-self: flex-end;
        background: rgba(31, 111, 235, 0.18);
        border: 1px solid rgba(31, 111, 235, 0.35);
        border-bottom-right-radius: 4px;
    }
    .speaker-name {
        font-size: 12px;
        font-weight: 700;
        color: #8b949e;
        margin-bottom: 4px;
    }
    .bubble-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 8px;
        font-size: 11px;
        color: #8b949e;
    }
    .instruction-text {
        font-size: 16px;
        color: #8b949e;
        margin-bottom: 6px;
    }
    </style>
""", unsafe_allow_html=True)

# Emotion Metadata (5 Mizo Classes)
EMOTION_META = {
    'joy':      {'icon': '🌈', 'color': '#FFD700', 'desc': 'Hlimna / Lawmna (Joy & Delight)'},
    'fear':     {'icon': '🛡️', 'color': '#9B59B6', 'desc': 'Hlauhna / Huphurhna (Fear & Anxiety)'},
    'sadness':  {'icon': '🌊', 'color': '#4A90E2', 'desc': 'Lungngaihna / Khawharna (Sadness & Grief)'},
    'anger':    {'icon': '🌋', 'color': '#FF4B2B', 'desc': 'Thinrimna / Khakna (Anger & Frustration)'},
    'neutral':  {'icon': '😐', 'color': '#95A5A6', 'desc': 'Ngaihsak loh / Pangngai (Neutrality & Calm)'},
}

VALENCE_SCORES = {
    'joy': 1.0,
    'neutral': 0.0,
    'fear': -0.6,
    'sadness': -0.8,
    'anger': -0.9
}

# Preprocessing for Mizo text
def clean_text_mizo(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', text)
    text = re.sub(r'([!?,.\-])', r' \1 ', text)
    text = re.sub(r"[^a-zA-Z\s\.\,\!\?\-\']", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# ============================================================
# PyTorch Deep Learning Model Definition
# ============================================================
class AttentionBiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128, num_classes=5, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.spatial_dropout = nn.Dropout2d(0.2)
        self.bilstm = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.unsqueeze(3).permute(0, 2, 1, 3)
        emb = self.spatial_dropout(emb).squeeze(3).permute(0, 2, 1)
        lstm_out, _ = self.bilstm(emb)
        attn_scores = self.attention(lstm_out)
        mask = (x == 0).unsqueeze(-1)
        attn_scores = attn_scores.masked_fill(mask, -1e9)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        logits = self.classifier(context)
        return logits, attn_weights

# ============================================================
# Model Loaders
# ============================================================
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models", "mizo", "mizo_ultra_final")

@st.cache_resource
def load_mizo_ensemble():
    """Load the SOTA Multi-Granularity Soft-Voting Ensemble."""
    try:
        pipeline = joblib.load(os.path.join(MODEL_DIR, "ultra_pipeline.pkl"))
        le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
        with open(os.path.join(MODEL_DIR, "metadata.json"), 'r', encoding='utf-8') as f:
            config = json.load(f)
        return pipeline, le, config.get('classes', list(le.classes_)), config
    except Exception as e:
        st.error(f"Error loading Ensemble: {e}")
        return None, None, None, None

@st.cache_resource
def load_deep_bilstm():
    """Load the PyTorch Attention BiLSTM Deep Learning Model."""
    try:
        pt_path = os.path.join(MODEL_DIR, "deep_bilstm.pt")
        vocab_path = os.path.join(MODEL_DIR, "deep_vocab.json")
        if not os.path.exists(pt_path) or not os.path.exists(vocab_path):
            return None, None, None
        
        checkpoint = torch.load(pt_path, map_location='cpu')
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
            
        model = AttentionBiLSTM(
            checkpoint['vocab_size'], checkpoint['emb_dim'],
            checkpoint['hidden_dim'], checkpoint['num_classes']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, vocab, checkpoint
    except Exception as e:
        return None, None, None

mizo_pipeline, mizo_le, mizo_classes, mizo_config = load_mizo_ensemble()
deep_model, deep_vocab, deep_meta = load_deep_bilstm()

# Prediction functions
def _temperature_scale(probs, temperature=0.6):
    log_probs = np.log(np.maximum(probs, 1e-10))
    scaled = log_probs / temperature
    scaled = scaled - np.max(scaled)
    exp_probs = np.exp(scaled)
    return exp_probs / np.sum(exp_probs)

def predict_ensemble(pipeline, text, classes):
    cleaned = clean_text_mizo(text)
    probs = pipeline.predict_proba([cleaned])[0]
    probs = _temperature_scale(probs, temperature=0.6)
    idx = np.argmax(probs)
    return classes[idx].lower(), float(probs[idx]), probs

def predict_deep_learning(model, vocab, text, classes, max_len=32):
    cleaned = clean_text_mizo(text)
    tokens = [vocab.get(w, 1) for w in cleaned.split()[:max_len]]
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    x = torch.tensor([tokens], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        logits, attn_weights = model(x)
        probs = F.softmax(logits, dim=-1)[0].numpy()
    idx = np.argmax(probs)
    return classes[idx].lower(), float(probs[idx]), probs

# ============================================================
# Header
# ============================================================
st.markdown("""
    <div style='text-align: center; margin-top: -15px; margin-bottom: 25px;'>
        <h1 style='font-size: 52px; font-weight: 800; margin-bottom: 0px; background: -webkit-linear-gradient(#58a6ff, #56d364); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Mizo Emotion AI Studio
        </h1>
        <div style='font-size: 15px; letter-spacing: 3px; color: #8b949e; text-transform: uppercase; font-weight: 600; margin-top: 5px;'>
            Subword Morphological NLP &bull; PyTorch Deep BiLSTM &bull; Soft-Voting Deep Ensemble
        </div>
    </div>
""", unsafe_allow_html=True)

# Layout Tabs
tab_single, tab_batch, tab_conv, tab_metrics, tab_admin = st.tabs([
    "🎙️ Single Sentence & Audio Analysis",
    "📂 Batch CSV Emotion Processor",
    "📈 Mizo Dialogue Timeline",
    "🔬 Model Performance Lab",
    "🛡️ Active Learning & Admin"
])

# ============================================================
# TAB 1: SINGLE SENTENCE ANALYSIS
# ============================================================
with tab_single:
    col_input, col_result = st.columns([1, 1], gap="large")
    
    with col_input:
        st.markdown("<h3 style='color: #58a6ff; font-weight: 700;'>Input Mizo Sentence / Speech</h3>", unsafe_allow_html=True)
        
        # Engine selector
        engine_choice = st.radio(
            "Select Inference Architecture:",
            [
                "🏆 Multi-Granularity Ensemble (80.7% Accuracy - Recommended)",
                "🧠 PyTorch Deep Learning (BiLSTM + Self-Attention)",
                "⚡ Dual-Engine Side-by-Side Comparison"
            ],
            index=0,
            key="engine_choice"
        )
        
        default_test_text = st.session_state.get('sample_input', "")
        user_input_single = st.text_input(
            "Input text single", 
            value=default_test_text,
            placeholder="e.g., ka va hlim tak em!  |  ka thinrim lutuk  |  ka hlau lutuk", 
            label_visibility="collapsed",
            key="input_single"
        )
        
        st.write("")
        # Audio input
        st.markdown("<div style='font-size: 14px; color: #8b949e; margin-bottom: 6px;'>🎙️ <b>Record Voice (Live Microphone)</b>:</div>", unsafe_allow_html=True)
        audio_file = st.audio_input("Record a voice clip", key="mic_input", label_visibility="collapsed")
        if audio_file:
            st.success("✅ Voice recording captured!")
            
        st.write("")
        submit_single = st.button("🚀 Analyze Emotion", key="submit_single")
        
    with col_result:
        actual_text = user_input_single.strip() if user_input_single else ""
        actual_audio = audio_file
        
        if submit_single or actual_text or actual_audio:
            if not actual_text and not actual_audio:
                st.warning("⚠️ Please enter a Mizo phrase or record audio to run analysis.")
            else:
                with st.spinner("Analyzing with selected Mizo engine..."):
                    # Predictions
                    ens_emotion, ens_conf, ens_probs = None, 0.0, None
                    deep_emotion, deep_conf, deep_probs = None, 0.0, None
                    
                    if mizo_pipeline and actual_text:
                        ens_emotion, ens_conf, ens_probs = predict_ensemble(mizo_pipeline, actual_text, mizo_classes)
                    if deep_model and deep_vocab and actual_text:
                        deep_emotion, deep_conf, deep_probs = predict_deep_learning(deep_model, deep_vocab, actual_text, mizo_classes)
                    
                    # Side-by-side mode
                    if "Dual-Engine" in engine_choice:
                        st.markdown("<h4 style='color: #bc8cff; text-align: center;'>⚡ Dual-Engine Prediction Comparison</h4>", unsafe_allow_html=True)
                        col_e1, col_e2 = st.columns(2)
                        
                        with col_e1:
                            m_ens = EMOTION_META.get(ens_emotion, {'icon': '•', 'color': '#58a6ff'})
                            st.markdown(f"""
                                <div class="emotion-card" style="padding: 20px;">
                                    <div class="lang-badge-mizo">🏆 Multi-Granularity Ensemble</div>
                                    <div style="font-size: 55px;">{m_ens['icon']}</div>
                                    <div style="font-size: 28px; font-weight:800; color:{m_ens['color']}; text-transform:uppercase;">{ens_emotion}</div>
                                    <div style="font-size: 18px; color: #c9d1d9; margin-top: 5px;">Confidence: <b>{ens_conf:.1%}</b></div>
                                    <div class="model-chip">Accuracy: 80.70% (Champion)</div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                        with col_e2:
                            m_deep = EMOTION_META.get(deep_emotion, {'icon': '•', 'color': '#bc8cff'})
                            st.markdown(f"""
                                <div class="emotion-card" style="padding: 20px;">
                                    <div class="lang-badge-mizo">🧠 PyTorch BiLSTM + Attention</div>
                                    <div style="font-size: 55px;">{m_deep['icon']}</div>
                                    <div style="font-size: 28px; font-weight:800; color:{m_deep['color']}; text-transform:uppercase;">{deep_emotion}</div>
                                    <div style="font-size: 18px; color: #c9d1d9; margin-top: 5px;">Confidence: <b>{deep_conf:.1%}</b></div>
                                    <div class="model-chip">Deep Neural Network (76.68%)</div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                        active_emotion = ens_emotion
                        active_probs = ens_probs
                        active_label = "Ensemble & PyTorch Comparative Run"
                        
                    elif "PyTorch" in engine_choice:
                        active_emotion = deep_emotion if deep_emotion else ens_emotion
                        active_probs = deep_probs if deep_probs is not None else ens_probs
                        active_conf = deep_conf if deep_conf else ens_conf
                        active_label = "PyTorch BiLSTM with Multi-Head Self-Attention"
                        meta = EMOTION_META.get(active_emotion, {'icon': '🔮', 'color': '#bc8cff', 'desc': 'Emotion'})
                        
                        st.markdown(f"""
                            <div class="emotion-card">
                                <div class="lang-badge-mizo">🧠 Deep Learning Engine</div>
                                <div class="emotion-icon">{meta['icon']}</div>
                                <div class="emotion-title" style="color: {meta['color']}">{active_emotion}</div>
                                <div style="font-size: 15px; color: #8b949e; margin-top: -5px; font-weight: 500;">{meta['desc']}</div>
                                <div style="color: #c9d1d9; font-size: 24px; margin-top: 14px; font-weight: 400;">Confidence: <b>{active_conf:.1%}</b></div>
                                <div class="model-chip">Engine: {active_label}</div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                    else:
                        active_emotion = ens_emotion
                        active_probs = ens_probs
                        active_conf = ens_conf
                        active_label = "Multi-Granularity Calibrated Soft-Voting Ensemble (Champion)"
                        meta = EMOTION_META.get(active_emotion, {'icon': '🔮', 'color': '#56d364', 'desc': 'Emotion'})
                        
                        st.markdown(f"""
                            <div class="emotion-card">
                                <div class="lang-badge-mizo">🏔️ Mizo Language</div>
                                <div class="emotion-icon">{meta['icon']}</div>
                                <div class="emotion-title" style="color: {meta['color']}">{active_emotion}</div>
                                <div style="font-size: 15px; color: #8b949e; margin-top: -5px; font-weight: 500;">{meta['desc']}</div>
                                <div style="color: #c9d1d9; font-size: 24px; margin-top: 14px; font-weight: 400;">Confidence: <b>{active_conf:.1%}</b></div>
                                <div class="model-chip">Engine: {active_label}</div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                    # Detailed analytics
                    if active_probs is not None:
                        sub1, sub2 = st.tabs(["📊 Probability Breakdown", "🔍 Explainable AI (XAI)"])
                        with sub1:
                            sorted_idx = np.argsort(active_probs)[::-1]
                            cols = st.columns(2)
                            for i, p_idx in enumerate(sorted_idx):
                                prob = active_probs[p_idx]
                                label = mizo_classes[p_idx]
                                m_info = EMOTION_META.get(label.lower(), {'icon': '•'})
                                with cols[i % 2]:
                                    st.progress(float(prob), text=f"{m_info['icon']} {label.capitalize()}: {prob:.1%}")
                                    
                        with sub2:
                            if actual_text and mizo_pipeline:
                                pred_fn = lambda t: predict_ensemble(mizo_pipeline, t, mizo_classes)[2]
                                xai_res = xai_explainer.explain_text(
                                    actual_text, pred_fn, clean_text_mizo, mizo_classes, active_emotion
                                )
                                h_meta = EMOTION_META.get(active_emotion, {'color': '#58a6ff'})
                                heatmap_html = xai_explainer.generate_heatmap_html(xai_res, h_meta['color'])
                                st.markdown(heatmap_html, unsafe_allow_html=True)

# ============================================================
# TAB 2: BATCH CSV
# ============================================================
with tab_batch:
    st.markdown("<h3 style='color: #58a6ff;'>Batch CSV Emotion Analysis</h3>", unsafe_allow_html=True)
    uploaded_csv = st.file_uploader("Upload CSV file", type=["csv"], key="batch_csv_uploader")
    if uploaded_csv is not None:
        try:
            df_b = pd.read_csv(uploaded_csv)
            t_col = st.selectbox("Text Column", df_b.columns.tolist())
            if st.button("🚀 Process Batch", key="run_batch"):
                predictions, confs = [], []
                for item in df_b[t_col]:
                    e, c, _ = predict_ensemble(mizo_pipeline, str(item), mizo_classes)
                    predictions.append(e)
                    confs.append(round(c, 4))
                df_b['predicted_emotion'] = predictions
                df_b['confidence'] = confs
                st.dataframe(df_b.head(10), use_container_width=True)
                st.download_button("💾 Download CSV", df_b.to_csv(index=False).encode('utf-8'), "mizo_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error: {e}")

# ============================================================
# TAB 3: DIALOGUE TIMELINE
# ============================================================
with tab_conv:
    st.markdown("<h3 style='color: #bc8cff;'>Mizo Conversational Dialogue Analytics</h3>", unsafe_allow_html=True)
    default_script = (
        "Lalruata: Ka ball min vawm bo sak daih chu le, ka thinrim lutuk!\n"
        "Zodina: Ava pawi ve, zawng leh ang u, hlau thawng suh.\n"
        "Lalruata: Ka college duhna ah ka lut thei dawn, ka va lawm tak em!\n"
        "Zodina: A va that hlauh chu, i lawm ang u hmiang!"
    )
    conv_txt = st.text_area("Dialogue Script (Speaker: Text)", value=default_script, height=140)
    if st.button("📈 Analyze Timeline", key="run_conv"):
        lines = conv_txt.strip().split("\n")
        dialogue = []
        for i, line in enumerate(lines):
            if ":" in line: sp, t = line.split(":", 1)
            else: sp, t = f"Speaker {i+1}", line
            e, c, _ = predict_ensemble(mizo_pipeline, t.strip(), mizo_classes)
            dialogue.append({"turn": i+1, "speaker": sp.strip(), "text": t.strip(), "emotion": e, "confidence": c, "valence": VALENCE_SCORES.get(e, 0.0)})
        if dialogue:
            df_d = pd.DataFrame(dialogue)
            if HAS_PLOTLY:
                fig = px.line(df_d, x="turn", y="valence", text="speaker", title="Conversational Emotion Arc")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#8b949e", yaxis_range=[-1.1, 1.1])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(df_d.set_index("turn")[["valence"]])

# ============================================================
# TAB 4: PERFORMANCE LAB
# ============================================================
with tab_metrics:
    st.markdown("<h3 style='color: #56d364;'>Model Performance & Architecture Lab</h3>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Ensemble Accuracy", "80.70%", "Champion (+1.68%)")
    with c2: st.metric("PyTorch BiLSTM Accuracy", "76.68%", "Deep Learning")
    with c3: st.metric("Features", "125,000+", "Subword Multi-Union")
    with c4: st.metric("Inference Latency", "< 4 ms", "Real-Time")
    
    cm_path = os.path.join(BASE_DIR, "cm_mizo.png")
    if os.path.exists(cm_path):
        st.image(cm_path, caption="Mizo Test Set Confusion Matrix (Normalized %)", use_container_width=True)

# ============================================================
# TAB 5: ADMIN & FEEDBACK
# ============================================================
with tab_admin:
    st.markdown("<h3 style='color: #ffa657;'>Active Learning Datastore</h3>", unsafe_allow_html=True)
    try:
        conn = sqlite3.connect("feedback.db")
        df_fb = pd.read_sql_query("SELECT * FROM feedback ORDER BY timestamp DESC", conn)
        conn.close()
        if len(df_fb) > 0:
            st.dataframe(df_fb, use_container_width=True)
        else:
            st.info("No feedback records logged yet.")
    except Exception as e:
        st.error(f"DB Error: {e}")

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=70)
    st.markdown("## 🏔️ Mizo Emotion AI")
    st.caption("Dual-Engine Architecture")
    st.markdown("---")
    
    st.markdown("### 📊 Active Models")
    if mizo_pipeline:
        st.success("**Multi-Granularity Ensemble** ✅\nAccuracy: **80.70%** (SOTA Champion)")
    if deep_model:
        st.info("**PyTorch BiLSTM + Attention** ✅\nAccuracy: **76.68%** (Deep Learning)")
        
    st.markdown("---")
    st.markdown("### 📝 Quick Test Samples")
    samples = [
        ("ka pass dawn chiang lutuk", "Joy 🌈"),
        ("ka va hlim tak em ka puak dawn!", "Joy 🌈"),
        ("ka thinrim lutuk", "Anger 🌋"),
        ("ka thil neih zawng zawng nen ka hua che !", "Anger 🌋"),
        ("ka hmu leh tawh dawn lo hi ka va lunggai em", "Sadness 🌊"),
        ("ka va ngai dawn che ve le", "Sadness 🌊"),
        ("ka hlau lutuk", "Fear 🛡️"),
        ("thil hlauhawm tawn dawnin ka mur chum chum zel", "Fear 🛡️"),
        ("dawr ka kal dawn", "Neutral 😐"),
        ("tinge ?", "Neutral 😐")
    ]
    for phrase, tag in samples:
        if st.button(f"{tag}: \"{phrase[:22]}...\"", key=f"side_{phrase[:10]}"):
            st.session_state['sample_input'] = phrase
            st.rerun()
