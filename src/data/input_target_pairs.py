import os
import logging
import csv

# Set up logging
log_dir = '/home/sehar/MLOPS/MLOPS-SCHITTVISION/MLOPS-SCHITTVISION/logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('input_target_pairs')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dir, 'input_target_pairs.log'))

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def generate_input_target_pairs(file_path, context_window_size):
    input_target_pairs = []
    current_scene = []

    def process_scene(scene, window_size):
        pairs = []
        for i in range(len(scene) - 1):
            input_dialogues = " | ".join(
                [f"{char}: \"{dialogue}\"" for char, dialogue in scene[max(0, i - window_size + 1): i + 1]]
            )
            target_dialogue = f"{scene[i + 1][0]}: \"{scene[i + 1][1]}\""
            pairs.append((input_dialogues, target_dialogue))
        return pairs

    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        # ✅ Validate headers
        required_fields = {"Character", "Dialogue"}
        missing = required_fields - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

        for row in reader:
            character = (row.get("Character") or "").strip()
            dialogue = (row.get("Dialogue") or "").strip()

            if not character or not dialogue:
                logger.warning(f"Skipping row with missing values: {row}")
                continue

            # Detect scene breaks
            if character.lower() == "unknown" or not dialogue.strip():
                if current_scene:
                    input_target_pairs += process_scene(current_scene, context_window_size)
                current_scene = []
                continue

            # Merge lines from same character
            if current_scene and current_scene[-1][0] == character:
                current_scene[-1] = (character, f"{current_scene[-1][1]} {dialogue}")
            else:
                current_scene.append((character, dialogue))

        if current_scene:
            input_target_pairs += process_scene(current_scene, context_window_size)

    return input_target_pairs


# ===== Main execution =====
data_path = "/home/sehar/MLOPS/MLOPS-SCHITTVISION/MLOPS-SCHITTVISION/data/interim/schitts_creek_dialogues_cleaned.csv"
output_path = "/home/sehar/MLOPS/MLOPS-SCHITTVISION/MLOPS-SCHITTVISION/data/external/input_target_pairs_final.csv"
context_window_size = 3

try:
    pairs = generate_input_target_pairs(data_path, context_window_size)

    with open(output_path, mode="w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Input", "Target"])
        writer.writerows(pairs)

    print(f"✅ Input-Target pairs have been saved to {output_path}")
    logger.info(f"Saved {len(pairs)} input-target pairs to {output_path}")

except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
except Exception as e:
    logger.error(f"An error occurred: {e}")
