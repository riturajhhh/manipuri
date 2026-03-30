import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
import os

# 1. LOAD DATA
def load_data(file_path):
    print("Loading dataset...")
    df = pd.read_excel(file_path)
    # Remove any extra spaces and standardize labels
    df['emotion'] = df['emotion'].str.strip().str.lower()
    df['text'] = df['text'].str.strip()
    return df

# 2. TRAINING
def train():
    dataset_path = 'manipuri_emotion_dataset_main1.xlsx'
    df = load_data(dataset_path)
    
    X = df['text'].tolist()
    y_raw = df['emotion'].tolist()
    
    # Label Encoding
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    # Stratified Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Powerful TF-IDF parameters for unique scripts (captures sub-words better)
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 10),
        max_features=50000,
        sublinear_tf=True,
        min_df=2
    )
    
    print("Extracting features with large character n-gram range...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # LinearSVC is often better for sparse text data than regular SVC
    # OneVsRest approach to handle multiclass effectively
    from sklearn.calibration import CalibratedClassifierCV
    base_model = LinearSVC(C=1.0, dual=False, random_state=42, max_iter=2000)
    model = OneVsRestClassifier(CalibratedClassifierCV(base_model, cv=3))
    
    print("Training One-Vs-Rest LinearSVC Model...")
    model.fit(X_train_tfidf, y_train)
    
    # Final Evaluation (after calibration)
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Save Model
    print("Saving model components...")
    if not os.path.exists("./manipuri_emotion_model"):
        os.makedirs("./manipuri_emotion_model")
        
    with open('./manipuri_emotion_model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    with open('./manipuri_emotion_model/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
        
    with open('./manipuri_emotion_model/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
        
    import json
    with open('./manipuri_emotion_model/config.json', 'w') as f:
        json.dump({"type": "final_ovr", "classes": le.classes_.tolist()}, f)
    
    print("Done!")
    return acc

if __name__ == "__main__":
    train()
