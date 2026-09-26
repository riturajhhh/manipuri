import os
import re
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
import joblib

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def clean_text_mizo(text):
    if not isinstance(text, str): return ""
    text = text.strip()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', text)
    text = re.sub(r'([!?,.\-])', r' \1 ', text)
    text = re.sub(r"[^a-zA-Z\s\.\,\!\?\-\']", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

base_dir = os.path.dirname(__file__)
dataset_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'datasets', 'mizotext.csv'))
output_dir = os.path.abspath(os.path.join(base_dir, 'mizo_ultra_final'))
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(dataset_path).dropna()
df['clean_text'] = df['Text'].apply(clean_text_mizo)
df = df[df['clean_text'].str.len() > 0]
df = df.drop_duplicates(subset=['clean_text', 'Emotion'])

le = joblib.load(os.path.join(output_dir, 'label_encoder.pkl'))
df['label'] = le.transform(df['Emotion'].str.strip().str.lower())
classes = le.classes_

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['clean_text'].values, df['label'].values, test_size=0.15, random_state=SEED, stratify=df['label'].values
)

# Build word vocabulary
word_counts = Counter()
for t in train_texts:
    word_counts.update(t.split())

vocab = {"<PAD>": 0, "<UNK>": 1}
for word, count in word_counts.items():
    if count >= 1:
        vocab[word] = len(vocab)

MAX_LEN = 32

def encode_text(t):
    tokens = [vocab.get(w, 1) for w in t.split()[:MAX_LEN]]
    if len(tokens) < MAX_LEN:
        tokens += [0] * (MAX_LEN - len(tokens))
    return tokens

X_train_seq = torch.tensor([encode_text(t) for t in train_texts], dtype=torch.long)
y_train_t = torch.tensor(train_labels, dtype=torch.long)
X_test_seq = torch.tensor([encode_text(t) for t in test_texts], dtype=torch.long)
y_test_t = torch.tensor(test_labels, dtype=torch.long)

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

train_dataset = TensorDataset(X_train_seq, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AttentionBiLSTM(len(vocab), emb_dim=128, hidden_dim=128, num_classes=len(classes)).to(device)

class_weights = torch.tensor(
    len(train_labels) / (len(classes) * np.bincount(train_labels)), dtype=torch.float32
).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

print("Training PyTorch BiLSTM with Self-Attention...")
best_acc = 0
best_state = None

for epoch in range(1, 16):
    model.train()
    total_loss = 0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        out, _ = model(bx)
        loss = criterion(out, by)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        
    model.eval()
    with torch.no_grad():
        test_out, _ = model(X_test_seq.to(device))
        test_preds = torch.argmax(test_out, dim=-1).cpu().numpy()
        acc = accuracy_score(test_labels, test_preds)
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict().copy()
            print(f"  Epoch {epoch:2d}: Test Acc = {acc:.4f} ({acc*100:.2f}%) *** BEST ***")

# Save model weights & vocab
model.load_state_dict(best_state)
torch_path = os.path.join(output_dir, 'deep_bilstm.pt')
vocab_path = os.path.join(output_dir, 'deep_vocab.json')

torch.save({
    'model_state_dict': best_state,
    'vocab_size': len(vocab),
    'emb_dim': 128,
    'hidden_dim': 128,
    'num_classes': len(classes),
    'max_len': MAX_LEN,
    'best_accuracy': float(best_acc)
}, torch_path)

with open(vocab_path, 'w', encoding='utf-8') as f:
    json.dump(vocab, f)

print(f"Saved PyTorch model to {torch_path}")
print(f"Saved Vocab to {vocab_path}")
print(f"Final PyTorch Model Accuracy: {best_acc*100:.2f}%")
