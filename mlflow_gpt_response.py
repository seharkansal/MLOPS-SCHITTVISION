import mlflow
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import joblib

# Paths - update as needed
GPT_MODEL_PATH = "/content/drive/My Drive/schittVision/final_model"
GPT_TOKENIZER_PATH = "/content/drive/My Drive/schittVision/final_tokenizer"
EMOTION_LABEL_ENCODER_PATH = "/content/drive/MyDrive/schittVision/emotion_label_encoder.pkl"

# Example prompts for inference logging
example_prompts = [
    "I’m just feeling really overwhelmed lately.",
    "What do you think about our chances?",
    "Did you hear about the new plan?"
]

# Load models and tokenizer
gpt_tokenizer = AutoTokenizer.from_pretrained(GPT_TOKENIZER_PATH)
gpt_model = AutoModelForCausalLM.from_pretrained(GPT_MODEL_PATH).eval()
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token  # Avoid warnings

# Optional: Load label encoder if you want to log or use emotion info
label_encoder = joblib.load(EMOTION_LABEL_ENCODER_PATH)

# Initialize text generation pipeline
gpt_pipeline = pipeline("text-generation", model=gpt_model, tokenizer=gpt_tokenizer, device=-1)

# Generate inference samples
generated_samples = []
for prompt in example_prompts:
    outputs = gpt_pipeline(prompt, max_length=60, temperature=0.8, top_k=50, top_p=0.95)
    generated_samples.append({
        "prompt": prompt,
        "generated_text": outputs[0]['generated_text']
    })

# Save generated samples locally to log as artifact
samples_path = "gpt_generated_samples.json"
with open(samples_path, "w") as f:
    json.dump(generated_samples, f, indent=2)

# Start MLflow logging
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("SchittVision-GPT")

with mlflow.start_run(run_name="gpt_inference_log"):

    # Log model params from your training args manually here (example values)
    mlflow.log_params({
        "model_name": "GPT2 (fine-tuned)",
        "epochs": 3,              # from your training args
        "batch_size": 4,          # from your training args
        "learning_rate": 5e-5,    # update if different
        "max_new_tokens": 60,
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.95
    })

    # Log the model and tokenizer (pytorch flavor)
    mlflow.pytorch.log_model(gpt_model, artifact_path="gpt_model")
    # You can also save tokenizer files as artifacts
    mlflow.log_artifact(GPT_TOKENIZER_PATH, artifact_path="tokenizer")

    # Log generated inference samples as JSON artifact
    mlflow.log_artifact(samples_path, artifact_path="inference_samples")

    # Log the label encoder if useful
    mlflow.log_artifact(EMOTION_LABEL_ENCODER_PATH, artifact_path="label_encoder")

    # Add tags to describe the run
    mlflow.set_tags({
        "author": "seharkansal",
        "project": "SchittsVision",
        "stage": "inference_logging"
    })

print("✅ GPT model, tokenizer, inference samples, and params logged to MLflow successfully.")
