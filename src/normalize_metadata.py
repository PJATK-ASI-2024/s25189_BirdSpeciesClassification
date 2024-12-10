import os
import logging
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler

def initialize_logger():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, '..', 'logs', 'normalization')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f'data_normalization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def load_dataframe(metadata_path):
    df = pd.read_csv(metadata_path)
    return df

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logger = initialize_logger()
    
    logger.info('Normalization logger initialized successfully.')

    logger.info('Loading metadata.csv')
    df = load_dataframe(os.path.join(script_dir, '..', 'data', 'train', 'metadata.csv'))
    print(df.head())
    logger.info('Metadata loaded successfully.')

    feature_cols = [c for c in df.columns if c not in ['class_id', 'image_id', 'image_path', 'classes_name']]

    data = df.copy()

    logger.info('Standardizing the num_pixels column.')
    scaler = StandardScaler()
    data['num_pixels'] = scaler.fit_transform(data[['num_pixels']])
    logger.info('num_pixels column standardized successfully.')
    print(data['num_pixels'].head())

    
    logger.info('Saving the normalized metadata to a CSV file.')
    data.to_csv(os.path.join(script_dir, '..', 'data', 'train', 'normalized_metadata.csv'), index=False)
    logger.info('Normalized metadata saved to data/train/normalized_metadata.csv.')


if __name__ == '__main__':
    main()