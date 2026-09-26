"""
Generate Confusion Matrix visualization for the Mizo Emotion Detection model.
Saves PNG image to the output or artifacts directory.
"""
import pandas as pd
import numpy as np
import joblib
import re
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

SEED = 42
OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(__file__)
BASE = os.path.dirname(__file__)

def clean_text_mizo(text):
    if not isinstance(text, str): return ""
    text = text.strip()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', text)
    text = re.sub(r'([!?,.\-])', r' \1 ', text)
    text = re.sub(r"[^a-zA-Z\s\.\,\!\?\-\']", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def plot_cm(y_true, y_pred, classes, title, filename, accuracy):
    labels = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.zeros_like(cm, dtype=float)
    nz = row_sums > 0
    cm_pct[nz.flatten()] = cm[nz.flatten()].astype('float') / row_sums[nz.flatten()] * 100

    fig, ax = plt.subplots(figsize=(8, 6.5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                linewidths=0.5, linecolor='#21262d',
                cbar_kws={'label': 'Percentage (%)'},
                ax=ax)

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j + 0.5, i + 0.72, f'({cm[i][j]})',
                    ha='center', va='center', fontsize=8, color='#8b949e')

    ax.set_xlabel('Predicted Emotion', fontsize=12, color='white', labelpad=10)
    ax.set_ylabel('Actual Emotion', fontsize=12, color='white', labelpad=10)
    ax.set_title(f'{title}\nAccuracy: {accuracy:.1%}', fontsize=14, color='white', pad=15, fontweight='bold')
    ax.tick_params(colors='white', labelsize=10)

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.tick_params(colors='white')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  Saved: {path}")
    return path

print("=" * 60)
print("GENERATING MIZO EMOTION CONFUSION MATRIX")
print("=" * 60)

try:
    pipe_path = os.path.join(BASE, 'models', 'mizo', 'mizo_ultra_final', 'ultra_pipeline.pkl')
    le_path = os.path.join(BASE, 'models', 'mizo', 'mizo_ultra_final', 'label_encoder.pkl')
    
    pipe = joblib.load(pipe_path)
    le = joblib.load(le_path)

    df = pd.read_csv(os.path.join(BASE, 'datasets', 'mizotext.csv'))
    df = df.rename(columns={'Emotion': 'emotion', 'Text': 'text'}).dropna()
    df['emotion'] = df['emotion'].astype(str).str.strip().str.lower()
    df['clean_text'] = df['text'].apply(clean_text_mizo)
    df = df[df['clean_text'].str.len() > 0].drop_duplicates(subset=['clean_text', 'emotion'])

    df['label'] = le.transform(df['emotion'])
    _, X_test, _, y_test = train_test_split(
        df['clean_text'].values, df['label'].values,
        test_size=0.15, random_state=SEED, stratify=df['label'].values
    )
    
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    plot_cm(y_test, y_pred, le.classes_, 'Mizo Emotion Detection (Ultra Final)', 'cm_mizo.png', acc)
    print(f"  Accuracy: {acc:.2%}")
    print(f"  Weighted F1: {f1_weighted:.4f}")
    print(f"  Macro F1: {f1_macro:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"  Error: {e}")
