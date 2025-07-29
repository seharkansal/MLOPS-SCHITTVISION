from transformers import AutoTokenizer
from datasets import Dataset
from datasets import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import pandas as pd
import os

# Load formatted data
df = pd.read_csv("/home/sehar/MLOPS/MLOPS-SCHITTVISION/MLOPS-SCHITTVISION/data/special_tokens/formatted_dataset_for_fine-tuning.csv",header=None, names=["gpt_format"])  # Replace with actual path

# Check the data

# Split before tokenizing
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# Convert to Hugging Face Datasets
train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

print(df.head())

# Make sure it has the column
print(df.columns)
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Add custom special tokens
special_tokens = {
    "additional_special_tokens": [
        "<<MOIRA>>", "<<DAVID>>", "<<ALEXIS>>", "<<JOHNNY>>",
        "<<anger>>", "<<joy>>", "<<fear>>", "<<neutral>>"
    ]
}
tokenizer.add_special_tokens(special_tokens)

# Save tokenizer (optional but useful)
tokenizer.save_pretrained("/home/sehar/MLOPS/MLOPS-SCHITTVISION/MLOPS-SCHITTVISION/models/tokenizer")

# Convert to Hugging Face dataset
# dataset = Dataset.from_pandas(df[["gpt_format"]])

# Tokenize
def tokenize(example):
    tokenized=tokenizer(
        example["gpt_format"],
        padding="max_length",
        truncation=True,
        max_length=128
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Apply tokenization
train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

# # Save tokenized dataset
# train_ds.save_to_disk("/home/sehar/MLOPS/MLOPS-SCHITTVISION/MLOPS-SCHITTVISION/data/train_tokenized_dataset")
# val_ds.save_to_disk("/home/sehar/MLOPS/MLOPS-SCHITTVISION/MLOPS-SCHITTVISION/data/val_tokenized_dataset")

print(train_ds[0])