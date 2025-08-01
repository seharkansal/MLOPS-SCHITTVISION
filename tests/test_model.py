# load test + signature test + performance test

import unittest
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, AutoTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json
import logging
import mlflow.pytorch
from transformers import TextClassificationPipeline
from sklearn.metrics import confusion_matrix
from transformers import get_linear_schedule_with_warmup
import mlflow
import os
import dagshub

dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
        raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "seharkansal"
repo_name = "MLOPS-SCHITTVISION"

        # Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# ---------- MLflow Utility Function ----------
def get_latest_model_uri(model_name, stage="Staging"):
    client = mlflow.MlflowClient()
    latest = client.get_latest_versions(model_name, stages=[stage])
    if not latest:
        latest = client.get_latest_versions(model_name, stages=["None"])
    if not latest:
        raise ValueError(f"No versions found for model '{model_name}'")
    return f"models:/{model_name}/{latest[0].version}"

# # Load emotion model
# emotion_tokenizer = AutoTokenizer.from_pretrained("mlflow-artifacts:/777957520883453524/f7675dc40496467c9b8fa9dc7284d546/artifacts/emotion_tokenizer")
# emotion_model = AutoModelForSequenceClassification.from_pretrained("runs:/f7675dc40496467c9b8fa9dc7284d546/emotion_model").eval()
label_encoder = joblib.load("./data/label_encoder.pkl")

# # # Load GPT model
# gpt_tokenizer = AutoTokenizer.from_pretrained("/home/sehar/MLOPS/MLOPS-SCHITTVISION/MLOPS-SCHITTVISION/models/final_tokenizer")
# # gpt_model = AutoModelForCausalLM.from_pretrained("runs:/f7675dc40496467c9b8fa9dc7284d546/gpt_model").eval()
# gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

# gpt_tokenizer.save_pretrained("/home/sehar/MLOPS/MLOPS-SCHITTVISION/MLOPS-SCHITTVISION/models/final_tokenizer")

# ---------- Load Registered Models ----------
# Emotion Detection Model
# emotion_model_uri = get_latest_model_uri("emotion_model")
# emotion_model = mlflow.pytorch.load_model(emotion_model_uri).eval()
# emotion_tokenizer = AutoTokenizer.from_pretrained("/home/sehar/MLOPS/MLOPS-SCHITTVISION/MLOPS-SCHITTVISION/models/final_emotion_tokenizer")

# emotion_tokenizer.save_pretrained("/home/sehar/MLOPS/MLOPS-SCHITTVISION/MLOPS-SCHITTVISION/models/final_emotion_tokenizer")
# GPT Response Model
# gpt_model_uri = get_latest_model_uri("gpt_model")
# gpt_model = mlflow.pytorch.load_model(gpt_model_uri).eval()
# gpt_tokenizer = AutoTokenizer.from_pretrained(gpt_model_uri)

class TestModelLoading_emotion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
          # Load the new model from MLflow model registry
        cls.new_emotion_model_name = "emotion_model"
        # emotion_model_uri = get_latest_model_uri(cls.new_model_name)
        # cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        cls.emotion_model_uri = get_latest_model_uri(cls.new_emotion_model_name)
        cls.new_emotion_model = mlflow.pytorch.load_model(cls.emotion_model_uri)

        cls.label_encoder = joblib.load("./data/label_encoder.pkl")

        cls.emotion_tokenizer = AutoTokenizer.from_pretrained("./models/final_emotion_tokenizer")

        cls.emotion_tokenizer.save_pretrained("./models/final_emotion_tokenizer")

         # Load holdout test data
        cls.holdout_data = pd.read_csv('./data/data/test.txt')

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_emotion_model)

    def test_emotion_model_signature(self):
        # Make dummy input and check output shape
        sample_text = "I feel happy"
        inputs = self.emotion_tokenizer(sample_text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = self.new_emotion_model(**inputs).logits
        self.assertEqual(logits.shape[0], 1)  # batch size
        self.assertEqual(logits.shape[1], len(self.label_encoder.classes_))  # num classes

    def test_model_performance(self):
        # Extract features and labels from holdout test data
        X_holdout = self.holdout_data.iloc[:,0:-1]
        y_holdout = self.holdout_data.iloc[:,-1]

        # Predict using the new model
        texts = X_holdout.iloc[:, 0].tolist()
        inputs = self.emotion_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            logits = self.new_emotion_model(**inputs).logits
            preds = torch.argmax(logits, dim=1).numpy()
        y_pred = preds

        # Calculate performance metrics for the new model
        accuracy_new = accuracy_score(y_holdout, y_pred)
        precision = precision_score(y_holdout, y_pred, average='weighted')
        recall = recall_score(y_holdout, y_pred, average='weighted')
        f1 = f1_score(y_holdout, y_pred, average='weighted')

        # Define expected thresholds for the performance metrics
        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        # Assert that the new model meets the performance thresholds
        self.assertGreaterEqual(accuracy_new, expected_accuracy, f'Accuracy should be at least {expected_accuracy}')
        self.assertGreaterEqual(precision, expected_precision, f'Precision should be at least {expected_precision}')
        self.assertGreaterEqual(recall, expected_recall, f'Recall should be at least {expected_recall}')
        self.assertGreaterEqual(f1, expected_f1, f'F1 score should be at least {expected_f1}')

class TestModelLoading_gpt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
          # Load the new model from MLflow model registry
        cls.new_model_name = "gpt_model"
        # emotion_model_uri = get_latest_model_uri(cls.new_model_name)
        # cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        cls.gpt_model_uri = get_latest_model_uri(cls.new_model_name)
        cls.new_gpt_model = mlflow.pytorch.load_model(cls.gpt_model_uri)

        cls.gpt_tokenizer = AutoTokenizer.from_pretrained("./models/final_tokenizer")
        # gpt_model = AutoModelForCausalLM.from_pretrained("runs:/f7675dc40496467c9b8fa9dc7284d546/gpt_model").eval()
        cls.gpt_tokenizer.pad_token = cls.gpt_tokenizer.eos_token

        cls.gpt_tokenizer.save_pretrained("./models/final_tokenizer")

        cls.new_emotion_model_name = "emotion_model"
        # emotion_model_uri = get_latest_model_uri(cls.new_model_name)
        # cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        cls.emotion_model_uri = get_latest_model_uri(cls.new_emotion_model_name)

        cls.new_emotion_model = mlflow.pytorch.load_model(cls.emotion_model_uri)

        cls.label_encoder = joblib.load("./data/label_encoder.pkl")

        cls.emotion_tokenizer = AutoTokenizer.from_pretrained("./models/final_emotion_tokenizer")

        cls.emotion_tokenizer.save_pretrained("./models/final_emotion_tokenizer")

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_gpt_model)

    def test_gpt_model_signature(self):
        # Prepare dummy prompt tokens
        prompt = "<<USER>> <<happy>>: I am happy <|response|> <<MOIRA>>"
        inputs = self.gpt_tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = self.new_gpt_model.generate(
                **inputs,
                max_new_tokens=10,
                top_k=10,
                top_p=0.9,
                temperature=0.8,
                pad_token_id=self.gpt_tokenizer.eos_token_id
            )
        self.assertTrue(output_ids.shape[1] > inputs.input_ids.shape[1])  # generated tokens > input tokens

    def test_inference_consistency(self):
        # Test full inference pipeline logic here (like your detect_emotion + generate_response)
        sample_text = "I am so excited!"
        # Emotion detection
        inputs = self.emotion_tokenizer(sample_text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = self.new_emotion_model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()
        predicted_emotion = self.label_encoder.inverse_transform([pred])[0]
        self.assertIn(predicted_emotion, self.label_encoder.classes_)

        # GPT response generation
        prompt = f"<<USER>> <<{predicted_emotion}>>: {sample_text} <|response|> <<MOIRA>>"
        inputs = self.gpt_tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = self.new_gpt_model.generate(
                **inputs,
                max_new_tokens=10,
                top_k=10,
                top_p=0.9,
                temperature=0.8,
                pad_token_id=self.gpt_tokenizer.eos_token_id
            )
        response = self.gpt_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)


if __name__ == "__main__":
    unittest.main()


        