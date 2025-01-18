import os
import logging
import pandas as pd
from datetime import datetime
import shutil
import random

def initialize_logger():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, '..', '..', 'logs', 'cnn', 'prep')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f'cnn_prep_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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

def load_metadata(metadata_path):
    df = pd.read_csv(metadata_path)
    return df

def get_label_for_image(image_path, df):
    return df[df['image_path'] == image_path]['class_id'].values

def create_image_labels(script_dir, logger, train_df, test_df):
    train_labels = []
    for idx, row in train_df.iterrows():
        image_path = row['image_path']
        label = row['class_id']
        train_labels.append((image_path, label))
    train_labels_df = pd.DataFrame(train_labels, columns=['image_path', 'label'])
    logger.info('Training labels generated successfully. Saving labels.csv to disk.')
    train_labels_df.to_csv(os.path.join(script_dir, '..', '..', 'data', 'train', 'split', 'train', 'train_labels.csv'), index=False)

    test_labels = []
    for idx, row in test_df.iterrows():
        image_path = row['image_path']
        label = row['class_id']
        test_labels.append((image_path, label))
    test_labels_df = pd.DataFrame(test_labels, columns=['image_path', 'label'])
    logger.info('Test labels generated successfully. Saving labels.csv to disk.')
    test_labels_df.to_csv(os.path.join(script_dir, '..', '..', 'data', 'train', 'split', 'test', 'test_labels.csv'), index=False)

def split_train_test(script_dir):
    """80/20 Train/test split"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.realpath(os.path.join(script_dir, "..", "..", "data", "train", "metadata.csv"))
    print("path_to_meta:", data_path)
    meta_df = load_metadata(data_path)
    print(meta_df.head())
    all_indices = meta_df.index.tolist()
    print(len(all_indices))
    random.shuffle(all_indices)
    
    train_size = int(0.8 * len(all_indices))
    train_indices = all_indices[:train_size]
    test_indices = all_indices[train_size:]

    train_df = meta_df.loc[train_indices].reset_index(drop=True)
    test_df = meta_df.loc[test_indices].reset_index(drop=True)

    print(train_df.head())
    print(test_df.head())

    return train_df, test_df

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logger = initialize_logger()
    logger.info('CNN prep logger initialized successfully.')

    # 1. Load metadata (has labels and image paths)
    logger.info('Loading metadata.csv')
    train_df, test_df = split_train_test(script_dir)
    logger.info('Metadata loaded successfully.')

    # 2. Create directories
    logger.info('Creating directories')
    if not os.path.exists(os.path.join(script_dir, '..', '..', 'data', 'train', 'split', 'train')):
        logger.info('Creating train directory')
        os.makedirs(os.path.join(script_dir, '..', '..', 'data', 'train', 'split', 'train'))
        os.makedirs(os.path.join(script_dir, '..', '..', 'data', 'train', 'split', 'train', 'images', 'raw'))
        os.makedirs(os.path.join(script_dir, '..', '..', 'data', 'train', 'split', 'train', 'images', 'processed'))
    if not os.path.exists(os.path.join(script_dir, '..', '..', 'data', 'train', 'split', 'test')):
        logger.info('Creating test directory')
        os.makedirs(os.path.join(script_dir, '..', '..', 'data', 'train', 'split', 'test'))
        os.makedirs(os.path.join(script_dir, '..', '..', 'data', 'train', 'split', 'test', 'images'))
    logger.info('Directories created successfully.')

    # 3. Preprocess the images
    logger.info('Preprocessing images')
    logger.info('Generating labels.csv')

    create_image_labels(script_dir, logger, train_df, test_df)

    logger.info('Preprocessing images completed successfully.')

    # 4. Save the raw images

    for idx, row in train_df.iterrows():
        src_path = os.path.join(script_dir, '..', '..', 'data', 'train', 'images', row['image_path'])
        dest_path = os.path.join(script_dir, '..', '..', 'data', 'train', 'split', 'train', 'images', 'raw', row['image_path'])
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(src_path, dest_path)
    
    for idx, row in test_df.iterrows():
        src_path = os.path.join(script_dir, '..', '..', 'data', 'train', 'images', row['image_path'])
        dest_path = os.path.join(script_dir, '..', '..', 'data', 'train', 'split', 'test', 'images', row['image_path'])
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(src_path, dest_path)

if __name__ == '__main__':
    main()