import os
import torch
import pandas as pd
import joblib
import mlflow
import json
import mlflow.pytorch
from transformers import TextClassificationPipeline
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)

mlflow.set_tracking_uri("http://127.0.0.1:5000")

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

# ========== Load Model & Tokenizer ==========
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = DistilBertTokenizer.from_pretrained(TOKENIZER_PATH)

# ========== MLflow Tracking ==========
mlflow.set_experiment(EXPERIMENT_NAME)
with mlflow.start_run(run_name="bert_emotion_classifier"):
    # Log Params
    mlflow.log_params({
        "model": "distilbert-base-uncased",
        "epochs": 3,
        "batch_size": 8,
        "lr": 2e-5,
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

    # Log encoder
    mlflow.log_artifact(ENCODER_PATH, artifact_path="label_encoder")

    # Set tags
    mlflow.set_tags({
        "author": "seharkansal",
        "project": "SchittsVision",
        "stage": "Evaluation Logging"
    })

print("✅ Model, tokenizer, encoder, metrics, and artifacts logged to MLflow.")
