import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from transformers import get_linear_schedule_with_warmup

# 1. Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.texts)

# 2. Load and prepare data
print("Loading dataset...")
df = pd.read_csv("./data/data/emotions_dataset.csv")
texts = df['Text'].tolist()
emotions = df['Emotion'].tolist()

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(emotions)
num_labels = len(label_encoder.classes_)
print(f"Classes found: {label_encoder.classes_}")

joblib.dump(label_encoder, "./data/label_encoder.pkl")
print("Label encoder saved.")

# Train-validation split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.1, random_state=42
)

# Tokenizer and datasets
tokenizer = DistilBertTokenizer.from_pretrained('./models/final_emotion_tokenizer')

train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
val_dataset = EmotionDataset(val_texts, val_labels, tokenizer)

# DataLoaders with optimized settings
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    pin_memory=True,
    num_workers=2
)
val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    pin_memory=True,
    num_workers=2
)

# Load model and move to device
model = DistilBertForSequenceClassification.from_pretrained(
    './models/final_emotion_model',
    num_labels=num_labels
)
model.to(device)

# Optimizer and scheduler setup
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 3
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# Mixed precision scaler
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

# Gradient accumulation steps (adjust as needed)
gradient_accumulation_steps = 4
print(f"Using gradient accumulation over {gradient_accumulation_steps} steps")

    # Validation
print("Running validation...")
model.eval()
preds = []
true_labels = []
with torch.no_grad():
    for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1)

            preds.extend(pred.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, preds)
    print(f"Validation Accuracy : {acc:.4f}")
    report=classification_report(true_labels, preds, output_dict=True, target_names=label_encoder.classes_)

    print(report)

    #  confusion matrix
    cm = confusion_matrix(true_labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    # Save and log it
    plt.tight_layout()
    plt.savefig("./reports/confusion_matrix.png")
    plt.close()

# Save metrics
with open("reports/emotion_evaluation.json", "w") as f:
    json.dump(report, f, indent=2)

print("✅ Emotion model evaluation complete.")