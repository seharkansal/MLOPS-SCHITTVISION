from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import joblib
import json
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
repo_name = "MLOPS-SCHITTVISION"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------

# Below code block is for local use
# -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri('https://dagshub.com/vikashdas770/YT-Capstone-Project.mlflow')
# dagshub.init(repo_owner='seharkansal', repo_name='MLOPS-SCHITTVISION', mlflow=True)
# -------------------------------------------------------------------------------------


# Load emotion model
emotion_tokenizer = AutoTokenizer.from_pretrained("./models/final_emotion_tokenizer")
emotion_model = AutoModelForSequenceClassification.from_pretrained("./models/final_emotion_model").eval()
label_encoder = joblib.load("./data/label_encoder.pkl")
print(type(label_encoder))  # Should print something like <class 'sklearn.preprocessing._label.LabelEncoder'>

# Load GPT model
gpt_tokenizer = AutoTokenizer.from_pretrained("./models/final_tokenizer")
gpt_model = AutoModelForCausalLM.from_pretrained("./models/final_model").eval()
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

gpt_tokenizer.save_pretrained("./models/final_tokenizer")
emotion_tokenizer.save_pretrained("./models/final_emotion_tokenizer")

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
    return response, emotion


# Example run
if __name__ == "__main__":
    with open("./data/inference_input.json", "r") as f:
        input_data = json.load(f)
    user_input = input_data["text"]
    character = input_data["character"]
    response = generate_response(user_input, character)
    final_response=f"{character} responds: {response}"

    # Save generated response to JSON file
    output_path = "reports/generated_responses.json"
    # Save generated output
    with open("reports/generated_responses.json", "w") as f:
        json.dump(final_response, f, indent=2)

 # MLflow logging
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("SchittVision-GPT-Inference")

    with mlflow.start_run(run_name="gpt_inference_run"):
        # Log input params used for generation
        mlflow.log_params({
            "max_new_tokens": 50,
            "top_k": 50,
            "top_p": 0.94,
            "per_device_train_batch_size":4,
            "per_device_eval_batch_size":4,
            "num_train_epochs":3,
            "temperature": 0.8
        })

        # Log artifacts: models, tokenizer, label encoder, generated response json
        mlflow.pytorch.log_model(gpt_model, artifact_path="gpt_model")
        mlflow.log_artifacts("./models/final_tokenizer", artifact_path="tokenizer")
        mlflow.pytorch.log_model(emotion_model, artifact_path="emotion_model")
        mlflow.log_artifacts("./models/final_emotion_tokenizer", artifact_path="emotion_tokenizer")
        mlflow.log_artifact("./data/label_encoder.pkl", artifact_path="label_encoder")
        mlflow.log_artifact(output_path, artifact_path="generated_responses")

        # Log tags/metadata
        mlflow.set_tags({
            "author": "seharkansal",
            "project": "SchittsVision",
            "stage": "inference"
        })

    print("✅ Inference and artifacts logged to MLflow successfully.")