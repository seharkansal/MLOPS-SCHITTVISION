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
import logging
import mlflow.pytorch
from transformers import TextClassificationPipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from transformers import get_linear_schedule_with_warmup
import mlflow
import os
import dagshub

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "seharkansal"
# repo_name = "MLOPS-SCHITTVISION"

# # Set up MLflow tracking URI
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# # -------------------------------------------------------------------------------------

# # Below code block is for local use
# # -------------------------------------------------------------------------------------
# # mlflow.set_tracking_uri('https://dagshub.com/vikashdas770/YT-Capstone-Project.mlflow')
# dagshub.init(repo_owner='seharkansal', repo_name='MLOPS-SCHITTVISION', mlflow=True)

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
df = pd.read_csv("/home/sehar/MLOPS/MLOPS-SCHITTVISION/MLOPS-SCHITTVISION/data/data/emotions_dataset.csv")
texts = df['Text'].tolist()
emotions = df['Emotion'].tolist()

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(emotions)
num_labels = len(label_encoder.classes_)
print(f"Classes found: {label_encoder.classes_}")

joblib.dump(label_encoder, "/home/sehar/MLOPS/MLOPS-SCHITTVISION/MLOPS-SCHITTVISION/data/label_encoder.pkl")
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
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']

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

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.debug('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise

print("✅ Emotion model evaluation complete.")



# ========== Constants ==========
CSV_PATH = "/home/sehar/MLOPS/MLOPS-SCHITTVISION/MLOPS-SCHITTVISION/data/data/emotions_dataset.csv"
MODEL_PATH = "/home/sehar/MLOPS/MLOPS-SCHITTVISION/MLOPS-SCHITTVISION/models/final_emotion_model"
TOKENIZER_PATH = "/home/sehar/MLOPS/MLOPS-SCHITTVISION/MLOPS-SCHITTVISION/models/final_emotion_tokenizer"
ENCODER_PATH = "/home/sehar/MLOPS/MLOPS-SCHITTVISION/MLOPS-SCHITTVISION/data/label_encoder.pkl"
METRICS_JSON_PATH = "/home/sehar/MLOPS/MLOPS-SCHITTVISION/MLOPS-SCHITTVISION/reports/emotion_evaluation.json"
CONFUSION_MATRIX_PATH = "/home/sehar/MLOPS/MLOPS-SCHITTVISION/MLOPS-SCHITTVISION/reports/confusion_matrix.png"
EXPERIMENT_NAME = "SchittVision-Emotion-BERT"

# Helper to flatten classification report dict
def flatten_metrics(metrics, parent_key='', sep='_'):
    flat_dict = {}
    for k, v in metrics.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            flat_dict.update(flatten_metrics(v, new_key, sep=sep))
        elif isinstance(v, (int, float)):
            flat_dict[new_key] = v
    return flat_dict

# ========== Skip if Already Trained ==========
if os.path.exists(MODEL_PATH):
    print("✅ Model already exists. Skipping training.")
else:
    print("⚠️ No model found. Please train before evaluation.")
    exit()

# ========== MLflow Tracking ==========
mlflow.set_experiment(EXPERIMENT_NAME)
with mlflow.start_run() as run:
    # Log Params
    mlflow.log_params({
        "model": "distilbert-base-uncased",
        "epochs": 3,
        "batch_size": 8,
        "lr": 2e-5,
        "max_length":128,
        "scheduler": "linear_warmup",
        "gradient_accumulation_steps": 4
    })

    # Load and log metrics JSON
    with open(METRICS_JSON_PATH, "r") as f:
        metrics = json.load(f)

    flat_metrics = flatten_metrics(metrics)
    for k, v in flat_metrics.items():
        mlflow.log_metric(k, v)

    # Log artifacts
    mlflow.log_artifact(METRICS_JSON_PATH, artifact_path="eval_reports")
    if os.path.exists(CONFUSION_MATRIX_PATH):
        mlflow.log_artifact(CONFUSION_MATRIX_PATH, artifact_path="eval_reports")

    # Log model
    mlflow.pytorch.log_model(model, artifact_path="emotion_model")
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)
    mlflow.transformers.log_model(transformers_model=pipeline, artifact_path="emotion_pipeline")

    # Save model info
    save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')

    # Log encoder
    mlflow.log_artifact(ENCODER_PATH, artifact_path="label_encoder")

    # Set tags
    mlflow.set_tags({
        "author": "seharkansal",
        "project": "SchittsVision",
        "stage": "Evaluation Logging"
    })

print("✅ Model, tokenizer, encoder, metrics, and artifacts logged to MLflow.")