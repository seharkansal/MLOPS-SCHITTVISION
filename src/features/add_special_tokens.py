import pandas as pd

# Load your dataset (update path as needed)
df = pd.read_csv("/home/sehar/MLOPS/MLOPS-SCHITTVISION/MLOPS-SCHITTVISION/data/raw/final_dataset.csv")

import re

character_map = {
    "Moira": "<Moira>",
    "Johnny": "<Johnny>",
    "David": "<David>",
    "Lawyer": "<Lawyer>",
    "Alexis":"<Alexis>"
}

emotion_map = {
    "anger": "<anger>",
    "joy": "<joy>",
    "sadness": "<sadness>",
    "neutral": "<neutral>",
    "fear":"<fear>"
}

def tokenize_line(line):
    # Try to match "Character (emotion): dialogue"
    match_with_emotion = re.match(r"(\w+)\s*\((\w+)\):\s*(.*)", line.strip())
    if match_with_emotion:
        character, emotion, dialogue = match_with_emotion.groups()
    else:
        # Try to match "Character: dialogue"
        match_without_emotion = re.match(r"(\w+):\s*(.*)", line.strip())
        if match_without_emotion:
            character, dialogue = match_without_emotion.groups()
            emotion = "neutral"  # default emotion
        else:
            return "<UNK>", "<neutral>", line.strip()
    
    char_token = character_map.get(character, "<UNK>")
    emotion_token = emotion_map.get(emotion.lower(), "<neutral>")
    return char_token, emotion_token, dialogue


def format_dialogue_string(input_dialogue_str, target_dialogue_str):
    turns = input_dialogue_str.split('|')
    formatted_turns = []
    for turn in turns:
        char, emo, dialog = tokenize_line(turn)
        formatted_turns.append(f"<{char.upper()}> <{emo.lower()}>: {dialog.strip()}")
    t_char, t_emo, t_dialog = tokenize_line(target_dialogue_str)
    target_formatted = f"<{t_char.upper()}> <{t_emo.lower()}>: {t_dialog.strip()}"
    
    # Join input turns with ' | ' and separate target with <|response|>
    full_text = " | ".join(formatted_turns) + " <|response|> " + target_formatted
    return full_text

# Apply formatting
df["gpt_format"] = df.apply(lambda row: format_dialogue_string(row["Formatted_Input"], row["Formatted_Target"]), axis=1)


print(df["gpt_format"].head().iloc[0])
# Save to file
# df["gpt_format"].to_csv("formatted_dataset_for_fine-tuning.csv", index=False, header=False)

print(format_dialogue_string("Lawyer (fear): ""Eli really did a number, Johnny. They're still looking for him, they think he's in the Caymans.""  |  Johnny: ""He was our business manager, he's supposed to pay taxes!""","Lawyer (joy): ""There is a very small amount set aside for you, and one asset the government has allowed you to retain. The children are dependents, Moira. You bought a small town in 1991, Johnny."""))