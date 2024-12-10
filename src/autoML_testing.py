import pandas as pd
import logging
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

def initialize_logger():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, '..', 'logs', 'automl')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f'data_automl_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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
    logger.info('AutoML logger initialized successfully.')
    logger.info('Loading metadata.csv')

    metadata_path = os.path.join(script_dir, '..', 'data', 'train', 'metadata.csv')
    df = load_dataframe(metadata_path)
    print(df.head())

    logger.info('Metadata loaded successfully.')

    logger.info('Splitting the dataset into features and target variable.')
    y = df['class_id']
    x = df.drop(columns=['class_id', 'image_id', 'image_path', 'classes_name'], axis=1)

    logger.info('Splitting the dataset into training and validation sets.')
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    logger.info('Training the TPOT model.') 
    tpot = TPOTClassifier(generations=5, population_size=20, verbosity=3, scoring='accuracy', random_state=42, n_jobs=2)
    
    logger.info('Fitting the TPOT model.')
    tpot.fit(X_train, y_train)

    logger.info('TPOT model fitted successfully.')

    logger.info('Evaluating the TPOT model.')

    score = tpot.score(X_val, y_val)

    logger.info('TPOT model evaluated successfully. TPOT best pipeline score: %s', score)

    pipeline_path = os.path.join(script_dir, 'tpot_pipeline.py')
    tpot.export(pipeline_path)

if __name__ == '__main__':
    main()
