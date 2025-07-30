# data ingestion
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import os
from sklearn.model_selection import train_test_split
import yaml
import logging
# from src.logger import logging
from src.connections import s3_connection

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except Exception as e:
        logging.error('Error loading params: %s', e)
        raise

def load_data(data_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        logging.info(f'Data loaded from {data_path}')
        return df
    except Exception as e:
        logging.error(f'Error loading data: {e}')
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logging.info("Preprocessing data...")
        # Drop duplicate rows if any
        df = df.drop_duplicates()
        
        # Drop rows with empty dialogue or character
        df = df.dropna(subset=['Character', 'Dialogue'])
        
        # Example text normalization: lowercase dialogues
        df['Dialogue'] = df['Dialogue'].str.lower()
        
        logging.info('Data preprocessing completed')
        return df
    except Exception as e:
        logging.error(f'Error during preprocessing: {e}')
        raise

def save_data(train_data: pd.DataFrame, data_path: str) -> None:
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        # test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logging.info(f'Train and test data saved to {raw_data_path}')
    except Exception as e:
        logging.error(f'Error saving data: {e}')
        raise

def main():
    try:
        # params = load_params(params_path='params.yaml')
        # test_size = params['data_ingestion']['test_size']
        test_size = 0.2
        
        # s3 = s3_connection.s3_operations("bucket-name", "accesskey", "secretkey")
        # df = s3.fetch_file_from_s3("data.csv")

        data_path = 'https://raw.githubusercontent.com/seharkansal/MLOPS-SCHITTVISION/master/schitts_creek_combined_dialogues.csv'
        df = load_data(data_path)

        
        # final_df = preprocess_data(df)
        # train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)
        save_data(df, data_path='./data/raw')
        
        # processed_df = preprocess_data(df)
        
        # train_data, test_data = train_test_split(processed_df, test_size=test_size, random_state=42)
        # save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logging.error(f'Failed to complete the data ingestion process: {e}')
        print(f"Error: {e}")

if __name__ == '__main__':
    main()