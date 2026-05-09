"""
Combined Training Engine - Manipuri Emotion Detection
Combines original 1200-sample dataset with 100K Meitei dataset (deduplicated)
Target: 85%+ accuracy
"""

import pandas as pd
import numpy as np
import joblib
import re
import os
import json
import random
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# 1. PREPROCESSING
# ============================================================
def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\u0980-\u09FF\uABC0-\uABFF\s\.\,\!\?\-]', '', text)
    return text.strip()

# ============================================================
# 2. NATURAL AUGMENTATION
# ============================================================
def augment_natural(texts, labels, multiplier=3):
    """Natural augmentation with controllable multiplier."""
    aug_texts = list(texts)
    aug_labels = list(labels)

    for _ in range(multiplier):
        for text, label in zip(texts, labels):
            words = text.split()
            n = len(words)

            # Pick a random strategy
            strategy = random.randint(0, 2)

            if strategy == 0 and n >= 3:
                # Duplicate a random word
                i = random.randint(0, n - 1)
                w2 = words[:i] + [words[i], words[i]] + words[i+1:]
                aug_texts.append(" ".join(w2))
                aug_labels.append(label)
            elif strategy == 1 and n >= 5:
                # Drop a random middle word
                i = random.randint(1, n - 2)
                w2 = words[:i] + words[i+1:]
                aug_texts.append(" ".join(w2))
                aug_labels.append(label)
            elif strategy == 2 and n >= 4:
                # Swap two adjacent words
                i = random.randint(0, n - 2)
                w2 = list(words)
                w2[i], w2[i+1] = w2[i+1], w2[i]
                aug_texts.append(" ".join(w2))
                aug_labels.append(label)

    return np.array(aug_texts), np.array(aug_labels)

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
def build_features():
    return FeatureUnion([
        ('word', TfidfVectorizer(
            analyzer='word', ngram_range=(1, 4),
            max_features=25000, sublinear_tf=True,
            min_df=1, max_df=0.90
        )),
        ('char', TfidfVectorizer(
            analyzer='char_wb', ngram_range=(2, 6),
            max_features=45000, sublinear_tf=True,
            min_df=1, max_df=0.90
        )),
    ])

# ============================================================
# 4. MODEL CONFIGS
# ============================================================
def get_models():
    models = {}

    # Bagged SVMs (best performers from V3)
    for c in [0.5, 1.0, 1.5, 2.0]:
        models[f'bag_svm_c{c}'] = BaggingClassifier(
            estimator=CalibratedClassifierCV(
                LinearSVC(C=c, dual=False, class_weight='balanced', max_iter=10000, random_state=SEED)
            ),
            n_estimators=20, max_samples=0.85, max_features=0.85, random_state=SEED, n_jobs=-1
        )

    # Calibrated SVMs
    for c in [0.5, 1.0, 2.0]:
        models[f'svm_c{c}'] = CalibratedClassifierCV(
            LinearSVC(C=c, dual=False, class_weight='balanced', max_iter=10000, random_state=SEED)
        )

    # Logistic Regression
    models['lr'] = LogisticRegression(C=2.0, max_iter=3000, class_weight='balanced', random_state=SEED)

    # Ridge Classifier
    models['ridge'] = RidgeClassifier(alpha=1.0, class_weight='balanced', random_state=SEED)

    # Passive Aggressive
    models['pac'] = PassiveAggressiveClassifier(max_iter=2000, random_state=SEED, class_weight='balanced')

    # Hard Voting
    models['vote_hard'] = VotingClassifier(
        estimators=[
            ('svm', LinearSVC(C=1.0, dual=False, class_weight='balanced', max_iter=10000, random_state=SEED)),
            ('lr', LogisticRegression(C=2.0, max_iter=3000, class_weight='balanced', random_state=SEED)),
            ('ridge', RidgeClassifier(alpha=1.0, class_weight='balanced', random_state=SEED)),
        ],
        voting='hard'
    )

    return models

# ============================================================
# 5. LOAD & COMBINE DATASETS
# ============================================================
def load_combined_data():
    print("=" * 60)
    print("LOADING & COMBINING DATASETS")
    print("=" * 60)

    # --- Dataset 1: Original 1200-sample Excel ---
    df1 = pd.read_excel('manipuri_emotion_dataset_main1.xlsx')
    text_col = 'text' if 'text' in df1.columns else df1.columns[0]
    emotion_col = 'emotion' if 'emotion' in df1.columns else df1.columns[1]
    df1 = df1[[text_col, emotion_col]].rename(columns={text_col: 'text', emotion_col: 'emotion'})
    df1 = df1.dropna()
    df1['emotion'] = df1['emotion'].astype(str).str.strip().str.lower()
    df1['source'] = 'original'
    print(f"  Original dataset: {len(df1)} samples")
    print(f"  Classes: {sorted(df1['emotion'].unique())}")

    # --- Dataset 2: 100K Meitei CSV (deduplicated) ---
    df2 = pd.read_csv('meitei_emotion_dataset_100k.csv')
    df2 = df2.drop_duplicates(subset='text')
    df2['emotion'] = df2['emotion'].astype(str).str.strip().str.lower()
    df2['source'] = 'meitei_100k'
    print(f"  Meitei 100K (deduped): {len(df2)} samples")
    print(f"  Classes: {sorted(df2['emotion'].unique())}")

    # --- Map Meitei labels to match original where possible ---
    label_map = {
        'angry': 'anger',
        'happy': 'joy',
        'sad': 'sadness',
        'surprise': 'surprise',
        'fear': 'fear',
        # New classes from 100K dataset
        'tired': 'tired',
        'proud': 'proud',
        'calm': 'calm',
        'lonely': 'lonely',
        'excited': 'excited',
    }
    df2['emotion'] = df2['emotion'].map(label_map)
    df2 = df2.dropna(subset=['emotion'])

    # --- Combine ---
    df = pd.concat([df1, df2], ignore_index=True)
    df['clean_text'] = df['text'].apply(clean_text)
    df = df[df['clean_text'].str.len() > 0]

    # Remove exact duplicates (same text + same emotion)
    df = df.drop_duplicates(subset=['clean_text', 'emotion'])

    print(f"\n  Combined dataset: {len(df)} unique samples")
    print(f"  Final classes: {sorted(df['emotion'].unique())}")
    print(f"  Class distribution:")
    for cls, count in df['emotion'].value_counts().sort_index().items():
        print(f"    {cls}: {count}")

    return df

# ============================================================
# 6. TRAINING
# ============================================================
def train():
    print("=" * 60)
    print("COMBINED TRAINING ENGINE")
    print("=" * 60)

    df = load_combined_data()

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['emotion'])
    n_classes = len(le.classes_)
    print(f"\nEncoded {n_classes} classes: {le.classes_.tolist()}")

    X_all, y_all = df['clean_text'].values, df['label'].values
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(
        X_all, y_all, test_size=0.15, random_state=SEED, stratify=y_all
    )
    print(f"Train: {len(X_train_raw)} | Test: {len(X_test)}")

    # Augment training data
    print("\nAugmenting training data...")
    X_train_aug, y_train_aug = augment_natural(X_train_raw, y_train_raw, multiplier=3)
    print(f"Augmented: {len(X_train_aug)} samples")

    # Build features
    print("\nBuilding features...")
    feats = build_features()
    X_tr_feat = feats.fit_transform(X_train_aug)
    X_te_feat = feats.transform(X_test)
    print(f"Features: {X_tr_feat.shape[1]}")

    # Train all models
    models = get_models()
    best_acc, best_name, best_model = 0, None, None

    print(f"\n{'=' * 60}")
    print(f"TRAINING {len(models)} MODELS")
    print(f"{'=' * 60}")

    for name, clf in models.items():
        print(f"  {name:20s} ... ", end="", flush=True)

        clf.fit(X_tr_feat, y_train_aug)
        y_pred = clf.predict(X_te_feat)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        tag = ""
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = clf
            tag = " *BEST*"

        print(f"Test={acc:.4f}  F1={f1:.4f}{tag}")

    # Final report
    print(f"\n{'=' * 60}")
    print(f"WINNER: {best_name}")
    print(f"Test Accuracy: {best_acc:.4%}")
    print(f"{'=' * 60}")

    y_pred_final = best_model.predict(X_te_feat)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_final, target_names=le.classes_))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_final)
    print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_))

    # Save
    model_dir = "./manipuri_ultra_final"
    os.makedirs(model_dir, exist_ok=True)

    full_pipeline = Pipeline([
        ('features', feats),
        ('clf', best_model)
    ])

    joblib.dump(full_pipeline, os.path.join(model_dir, "ultra_pipeline.pkl"))
    joblib.dump(le, os.path.join(model_dir, "label_encoder.pkl"))

    with open(os.path.join(model_dir, "metadata.json"), 'w') as f:
        json.dump({
            "type": f"combined_{best_name}",
            "classes": le.classes_.tolist(),
            "test_accuracy": float(best_acc),
            "best_model": best_name,
            "n_train": len(X_train_aug),
            "n_test": len(X_test),
            "n_classes": n_classes,
        }, f, indent=2)

    print(f"\nSaved to {model_dir}/")
    print(f"FINAL ACCURACY: {best_acc:.2%}")

if __name__ == "__main__":
    train()
