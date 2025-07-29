from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import joblib
import json

# Load emotion model
emotion_tokenizer = AutoTokenizer.from_pretrained("/home/sehar/MLOPS/MLOPS-SCHITTVISION/MLOPS-SCHITTVISION/models/final_emotion_tokenizer")
emotion_model = AutoModelForSequenceClassification.from_pretrained("/home/sehar/MLOPS/MLOPS-SCHITTVISION/MLOPS-SCHITTVISION/models/final_emotion_model").eval()
label_encoder = joblib.load("/home/sehar/MLOPS/MLOPS-SCHITTVISION/MLOPS-SCHITTVISION/data/label_encoder.pkl")

# Load GPT model
gpt_tokenizer = AutoTokenizer.from_pretrained("/home/sehar/MLOPS/MLOPS-SCHITTVISION/MLOPS-SCHITTVISION/models/final_tokenizer")
gpt_model = AutoModelForCausalLM.from_pretrained("/home/sehar/MLOPS/MLOPS-SCHITTVISION/MLOPS-SCHITTVISION/models/final_model").eval()
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

def detect_emotion(text):
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
    pred = torch.argmax(F.softmax(logits, dim=1), dim=1).item()
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
    return generated.replace(prompt, "").strip()


# Example run
if __name__ == "__main__":
    user_input = "I’m just feeling really overwhelmed lately."
    character = "<<MOIRA>>"  # User selects the character
    response = generate_response(user_input, character)
    final_response=f"{character} responds: {response}"

    # Save generated output
with open("reports/generated_responses.json", "w") as f:
    json.dump(final_response, f, indent=2)