"""
Training Engine - Assamese Emotion Detection (Classical ML)
Dataset: 80K balanced samples, 8 emotion classes
Architecture: TF-IDF (word + char) → Ensemble ML classifiers
"""

import pandas as pd
import numpy as np
import joblib
import re
import os
import json
import random
import warnings
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# 1. PREPROCESSING (Assamese / Bengali script)
# ============================================================
def clean_text_assamese(text):
    """Clean and normalize Assamese text, keeping Bengali script + essential punctuation."""
    if not isinstance(text, str): return ""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Keep Assamese/Bengali script (U+0980-U+09FF), dandas (U+0964-0965),
    # apostrophes/quotes (used in Assamese like গ'ল), and basic punctuation
    text = re.sub(r"[^\u0980-\u09FF\u0964-\u0965\s\.\,\!\?\-\'\'\']", '', text)
    return text.strip()

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
def build_features():
    """Build TF-IDF feature union optimized for Assamese text."""
    return FeatureUnion([
        ('word', TfidfVectorizer(
            analyzer='word', ngram_range=(1, 3),
            max_features=30000, sublinear_tf=True,
            min_df=2, max_df=0.90
        )),
        ('char', TfidfVectorizer(
            analyzer='char_wb', ngram_range=(2, 5),
            max_features=50000, sublinear_tf=True,
            min_df=2, max_df=0.90
        )),
    ])

# ============================================================
# 3. MODEL CONFIGS
# ============================================================
def get_models():
    """Get ensemble of classifiers optimized for large dataset."""
    models = {}

    # Bagged SVMs (top performers)
    for c in [0.5, 1.0, 1.5, 2.0]:
        models[f'bag_svm_c{c}'] = BaggingClassifier(
            estimator=CalibratedClassifierCV(
                LinearSVC(C=c, dual=False, class_weight='balanced', max_iter=10000, random_state=SEED)
            ),
            n_estimators=10, max_samples=0.8, max_features=0.8, random_state=SEED, n_jobs=-1
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

    return models

# ============================================================
# 4. LOAD DATASET
# ============================================================
def load_data():
    print("=" * 60)
    print("LOADING ASSAMESE EMOTION DATASET")
    print("=" * 60)

    # Dataset is in ../../datasets/
    dataset_path = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'assamese_emotion.xlsx')
    dataset_path = os.path.abspath(dataset_path)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    df = pd.read_excel(dataset_path)
    print(f"  Raw dataset: {len(df)} samples")
    
    # Map Column1 to text and Column2 to emotion
    df = df.rename(columns={'Column1': 'text', 'Column2': 'emotion'})
    
    # Normalize
    df = df.dropna(subset=['text', 'emotion'])
    df['emotion'] = df['emotion'].astype(str).str.strip().str.lower()
    
    # Remove header-like row
    df = df[df['emotion'] != 'emotion']
    
    df['clean_text'] = df['text'].apply(clean_text_assamese)
    df = df[df['clean_text'].str.len() > 0]
    df = df.drop_duplicates(subset=['clean_text', 'emotion'])

    print(f"  Cleaned dataset: {len(df)} samples")
    print(f"  Classes: {sorted(df['emotion'].unique())}")
    print(f"  Class distribution:")
    for cls, count in df['emotion'].value_counts().sort_index().items():
        print(f"    {cls}: {count}")

    return df

# ============================================================
# 6. TRAINING
# ============================================================
def train():
    start_time = time.time()

    print("=" * 60)
    print("ASSAMESE EMOTION DETECTION - CLASSICAL ML TRAINING")
    print("=" * 60)

    df = load_data()

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['emotion'])
    n_classes = len(le.classes_)
    print(f"\nEncoded {n_classes} classes: {le.classes_.tolist()}")

    X_all, y_all = df['clean_text'].values, df['label'].values
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(
        X_all, y_all, test_size=0.15, random_state=SEED, stratify=y_all
    )
    print(f"Train: {len(X_train_raw)} | Test: {len(X_test)}")

    # Augment training data (dataset is very small ~880, so augmentation helps)
    print("\nAugmenting training data...")
    X_train_aug, y_train_aug = augment_natural(X_train_raw, y_train_raw, multiplier=2)
    print(f"Augmented: {len(X_train_aug)} samples")

    # Build features
    print("\nBuilding TF-IDF features...")
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
        t0 = time.time()

        clf.fit(X_tr_feat, y_train_aug)
        y_pred = clf.predict(X_te_feat)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        elapsed = time.time() - t0

        tag = ""
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = clf
            tag = " *BEST*"

        print(f"Test={acc:.4f}  F1={f1:.4f}  ({elapsed:.1f}s){tag}")

    # Final report
    print(f"\n{'=' * 60}")
    print(f"WINNER: {best_name}")
    print(f"Test Accuracy: {best_acc:.4%}")
    print(f"{'=' * 60}")

    y_pred_final = best_model.predict(X_te_feat)
    print("\nClassification Report:")
    unique_labels = np.unique(np.concatenate((y_test, y_pred_final)))
    target_names = [le.classes_[i] for i in unique_labels]
    print(classification_report(y_test, y_pred_final, labels=unique_labels, target_names=target_names))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_final, labels=unique_labels)
    print(pd.DataFrame(cm, index=target_names, columns=target_names))

    # Save to local model directory
    model_dir = os.path.join(os.path.dirname(__file__), "assamese_ultra_final")
    os.makedirs(model_dir, exist_ok=True)

    full_pipeline = Pipeline([
        ('features', feats),
        ('clf', best_model)
    ])

    joblib.dump(full_pipeline, os.path.join(model_dir, "ultra_pipeline.pkl"))
    joblib.dump(le, os.path.join(model_dir, "label_encoder.pkl"))

    with open(os.path.join(model_dir, "metadata.json"), 'w') as f:
        json.dump({
            "type": f"assamese_{best_name}",
            "language": "assamese",
            "classes": le.classes_.tolist(),
            "test_accuracy": float(best_acc),
            "best_model": best_name,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_classes": n_classes,
        }, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nSaved to {model_dir}/")
    print(f"FINAL ACCURACY: {best_acc:.2%}")
    print(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} min)")

if __name__ == "__main__":
    train()
