"""
Transformer Fine-Tuning - Assamese Emotion Detection using MuRIL
Model: google/muril-base-cased (pretrained on 17 Indian languages including Assamese)
Dataset: 80K balanced samples, 8 emotion classes

REQUIREMENTS:
    pip install torch transformers accelerate

NOTE:
    - GPU recommended for faster training (~30 min on GPU vs ~6-12 hours on CPU)
    - Automatically detects GPU/CPU
"""

import pandas as pd
import numpy as np
import re
import os
import json
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Check Dependencies
# ============================================================
try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, EarlyStoppingCallback
    )
    from torch.utils.data import Dataset
    HAS_TRANSFORMERS = True
except ImportError:
    print("=" * 60)
    print("ERROR: Missing dependencies!")
    print("Please install: pip install torch transformers accelerate")
    print("=" * 60)
    HAS_TRANSFORMERS = False
    exit(1)

SEED = 42
MODEL_NAME = "google/muril-base-cased"
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5

# ============================================================
# 1. PREPROCESSING
# ============================================================
def clean_text_assamese(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r"[^\u0980-\u09FF\u0964-\u0965\s\.\,\!\?\-\'\'\']", '', text)
    return text.strip()

# ============================================================
# 2. DATASET CLASS
# ============================================================
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ============================================================
# 3. METRICS
# ============================================================
def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)
    acc = accuracy_score(labels, preds)
    f1_weighted = f1_score(labels, preds, average='weighted')
    f1_macro = f1_score(labels, preds, average='macro')
    return {'accuracy': acc, 'f1_weighted': f1_weighted, 'f1_macro': f1_macro}

# ============================================================
# 4. TRAINING
# ============================================================
def train():
    start_time = time.time()

    print("=" * 60)
    print("ASSAMESE EMOTION DETECTION - MuRIL TRANSFORMER TRAINING")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load dataset
    dataset_path = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'assamese_emotion.xlsx')
    dataset_path = os.path.abspath(dataset_path)

    print(f"\nLoading dataset from: {dataset_path}")
    df = pd.read_excel(dataset_path)
    
    # Map Column1 to text and Column2 to emotion
    df = df.rename(columns={'Column1': 'text', 'Column2': 'emotion'})
    
    df = df.dropna(subset=['text', 'emotion'])
    df['emotion'] = df['emotion'].astype(str).str.strip().str.lower()
    
    # Remove header-like row
    df = df[df['emotion'] != 'emotion']
    
    df['clean_text'] = df['text'].apply(clean_text_assamese)
    df = df[df['clean_text'].str.len() > 0]
    df = df.drop_duplicates(subset=['clean_text', 'emotion'])

    print(f"Dataset: {len(df)} samples, {df['emotion'].nunique()} classes")
    print(f"Classes: {sorted(df['emotion'].unique())}")

    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['emotion'])
    n_classes = len(le.classes_)
    label2id = {label: int(i) for i, label in enumerate(le.classes_)}
    id2label = {int(i): label for i, label in enumerate(le.classes_)}

    # Split
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=SEED, stratify=df['label'])
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=SEED, stratify=train_df['label'])

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Load tokenizer & model
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=n_classes,
        label2id=label2id,
        id2label=id2label
    )

    # Create datasets
    train_dataset = EmotionDataset(
        train_df['clean_text'].tolist(),
        train_df['label'].tolist(),
        tokenizer
    )
    val_dataset = EmotionDataset(
        val_df['clean_text'].tolist(),
        val_df['label'].tolist(),
        tokenizer
    )
    test_dataset = EmotionDataset(
        test_df['clean_text'].tolist(),
        test_df['label'].tolist(),
        tokenizer
    )

    # Training arguments
    output_dir = os.path.join(os.path.dirname(__file__), "assamese_transformer")
    batch_size = BATCH_SIZE if device.type == 'cuda' else 16

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=100,
        fp16=device.type == 'cuda',
        dataloader_num_workers=0,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Train
    print(f"\nStarting training ({EPOCHS} epochs, batch_size={batch_size})...")
    print("This may take a while on CPU. GPU is recommended.")
    train_result = trainer.train()

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("EVALUATION ON TEST SET")
    print("=" * 60)
    test_results = trainer.evaluate(test_dataset)
    print(f"Test Accuracy:    {test_results['eval_accuracy']:.4%}")
    print(f"Test F1 Weighted: {test_results['eval_f1_weighted']:.4%}")
    print(f"Test F1 Macro:    {test_results['eval_f1_macro']:.4%}")

    # Detailed predictions
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    true_labels = test_df['label'].values

    from sklearn.metrics import classification_report, confusion_matrix
    print("\nClassification Report:")
    print(classification_report(true_labels, preds, target_names=le.classes_))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, preds)
    print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_))

    # Save model, tokenizer, and metadata
    final_dir = os.path.join(os.path.dirname(__file__), "assamese_transformer_final")
    os.makedirs(final_dir, exist_ok=True)

    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    import joblib
    joblib.dump(le, os.path.join(final_dir, "label_encoder.pkl"))

    with open(os.path.join(final_dir, "metadata.json"), 'w') as f:
        json.dump({
            "type": "transformer_muril",
            "language": "assamese",
            "base_model": MODEL_NAME,
            "classes": le.classes_.tolist(),
            "test_accuracy": float(test_results['eval_accuracy']),
            "test_f1_weighted": float(test_results['eval_f1_weighted']),
            "test_f1_macro": float(test_results['eval_f1_macro']),
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "n_classes": n_classes,
            "max_length": MAX_LENGTH,
            "epochs": EPOCHS,
        }, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nSaved transformer model to {final_dir}/")
    print(f"FINAL TEST ACCURACY: {test_results['eval_accuracy']:.2%}")
    print(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} min)")

if __name__ == "__main__":
    train()
