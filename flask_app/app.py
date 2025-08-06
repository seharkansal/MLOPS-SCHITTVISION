from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import joblib
import json
import mlflow
from mlflow.tracking import MlflowClient
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import mlflow.pytorch
from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import joblib
import dagshub
from flask import Flask, request, jsonify, send_from_directory
import os


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
repo_name = "MLOPS-SCHITTVISION"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------

# Below code block is for local use
# -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri('https://dagshub.com/vikashdas770/YT-Capstone-Project.mlflow')
# dagshub.init(repo_owner='seharkansal', repo_name='MLOPS-SCHITTVISION', mlflow=True)

from mlflow.tracking import MlflowClient

client = MlflowClient()

registered_models = client.search_registered_models()
print("Registered models:", [rm.name for rm in registered_models])

app = Flask(__name__)

# ---------- MLflow Utility Function ----------
def get_latest_model_uri(model_name, stage="Staging"):
    client = MlflowClient()
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
gpt_tokenizer = AutoTokenizer.from_pretrained("./models/final_tokenizer")
# gpt_model = AutoModelForCausalLM.from_pretrained("runs:/f7675dc40496467c9b8fa9dc7284d546/gpt_model").eval()
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

gpt_tokenizer.save_pretrained("./models/final_tokenizer")

# ---------- Load Registered Models ----------
# Emotion Detection Model
emotion_model_uri = get_latest_model_uri("emotion_model")
emotion_model = mlflow.pytorch.load_model(emotion_model_uri).eval()
emotion_tokenizer = AutoTokenizer.from_pretrained("./models/final_emotion_tokenizer")

emotion_tokenizer.save_pretrained("./models/final_emotion_tokenizer")
# GPT Response Model
gpt_model_uri = get_latest_model_uri("gpt_model")
gpt_model = mlflow.pytorch.load_model(gpt_model_uri).eval()
# gpt_tokenizer = AutoTokenizer.from_pretrained(gpt_model_uri)


def detect_emotion(text):
    print("label_encoder type:", type(label_encoder))
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
    pred = torch.argmax(F.softmax(logits, dim=1), dim=1).item()
    print("predicted label index:", pred)
    print("decoded label:", label_encoder.inverse_transform([pred])[0])
    return f"<<{label_encoder.inverse_transform([pred])[0]}>>"

def generate_response(user_input, character):
    emotion = detect_emotion(user_input)
    print(emotion)
    prompt = f"<<USER>> {emotion}: {user_input} <|response|> {character}"
    inputs = gpt_tokenizer(prompt, return_tensors="pt").to(gpt_model.device)

    with torch.no_grad():
        output_ids = gpt_model.generate(
            **inputs,
            max_new_tokens=50,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=gpt_tokenizer.eos_token_id
        )
    generated = gpt_tokenizer.decode(output_ids[0], skip_special_tokens=False)
    response = generated.replace(prompt, "").strip()
    return emotion,response


# Flask route
@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    print("Request data:", data)
    user_input = data.get('user_input')  or data.get('text') 
    character = data.get('character', '<<MOIRA>>')

    if not user_input:
        return jsonify({"error": "Missing 'user_input'"}), 400

    emotion,response = generate_response(user_input, character)
    print(f"Response: {response}, Emotion: {emotion}")

    return jsonify({
        "character": character,
        "detected_emotion": emotion,
        "response": response
    })

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5001)
