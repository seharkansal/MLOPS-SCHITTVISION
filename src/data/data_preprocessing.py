import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re
import pandas as pd

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Setting up logger
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def preprocess_df(df):
    """
    Preprocesses the DataFrame by encoding the target column, removing duplicates, and transforming the text column.
    """
    try:
        logger.debug('Starting preprocessing for DataFrame')

        # Remove duplicate rows
        df = df.dropna(subset=['Dialogue'])  # Drop rows where Dialogue is NaN
        df = df[df['Dialogue'].str.strip() != '']  # Drop rows with empty Dialogue
        logger.debug('Duplicates removed')
    
        # Apply text transformation to the specified text column
        print(df.shape[0])  # Total number of rows
        print(df.isnull().sum())  # Count NaN values
        print(df[df['Dialogue'].str.strip() == ''])  # Check for empty strings
        df = df.reset_index(drop=True)
        logger.debug('dialogues cleaned')
        return df
    
    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise

# Helper function to clean the dialogue
def clean_dialogue(dialogue):
    # Remove unwanted characters or excessive spaces
    dialogue = re.sub(r'[^a-zA-Z0-9\s.,!?;\'\"-]', '', dialogue)
    dialogue = re.sub(r'\s+', ' ', dialogue)  # Replace multiple spaces with single space
    return dialogue.strip()

def main():
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    try:
        # Fetch the data from data/raw
        # Load the CSV file
        df = pd.read_csv("/home/sehar/MLOPS/MLOPS-SCHITTVISION/MLOPS-SCHITTVISION/data/raw/schitts_creek_combined_dialogues.csv")
        # train_data = pd.read_csv('./data/raw/train.csv')
        # test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded properly')

        # Clean the dialogue column first
        df['Dialogue'] = df['Dialogue'].apply(clean_dialogue)

        # Drop short/empty dialogues
        df = df[df['Dialogue'].apply(lambda x: len(x.split()) > 3)]
        logger.debug(f"Remaining rows after filtering: {len(df)}")

         # Store the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        # Save cleaned data for reference
        cleaned_path = "./data/interim/schitts_creek_dialogues_cleaned.csv"
        os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
        df.to_csv(cleaned_path, index=False)
        print(f"Cleaned data saved to {cleaned_path}")

        # Then preprocess (e.g., create input-target pairs, formatting, etc.)
        processed_data = preprocess_df(df)

        # Save final processed data
        interim_path = "./data/interim/schitts_creek_dialogues_cleaned.csv"
        os.makedirs(os.path.dirname(interim_path), exist_ok=True)
        processed_data.to_csv(interim_path, index=False)
        print(f"Processed data saved to {interim_path}")

        
        logger.debug('Processed data saved to %s', interim_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()