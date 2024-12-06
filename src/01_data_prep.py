import kagglehub
import logging
import os
import shutil
import random
import tarfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image

# Global variables
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"
TRAIN_DIR = "data/train"
FINETUNE_DIR = "data/finetune"

def main():

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    log_file_path = f'logs/prep/data_prep_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logging.info("Downloading data from Kaggle")
    RAW_DATA_PATH = download_archive()

    logging.info("Checking if data directories exist")
    if not os.path.exists(PROCESSED_DATA_PATH) or not os.path.exists(TRAIN_DIR) or not os.path.exists(FINETUNE_DIR):
        os.makedirs(PROCESSED_DATA_PATH)
        os.makedirs(os.path.join(TRAIN_DIR))
        os.makedirs(os.path.join(FINETUNE_DIR))
        logging.info("Data directories created")
    else:
        logging.info("Data directories already exist")
    
    # Add the handlers to the logger
    logging.getLogger().addHandler(file_handler)

    logging.info("Starting data preparation script")

    unzip_data(RAW_DATA_PATH, PROCESSED_DATA_PATH, TRAIN_DIR, FINETUNE_DIR)

    # Extract the data
    logging.info("Extracting data")

    extract_tgz(os.path.join(RAW_DATA_PATH, "CUB_200_2011.tgz"), PROCESSED_DATA_PATH)
    extract_tgz(os.path.join(RAW_DATA_PATH, "segmentations.tgz"), PROCESSED_DATA_PATH)

    logging.info("Data extraction complete")

    # Load the data
    logging.info("Loading data")

    meta_df = load_and_merge_data(PROCESSED_DATA_PATH)

    meta_df = meta_df.join(meta_df["image_path"].apply(get_image_stats))

    logging.info("Data loaded")

    print("-"*50)

    # Display basic information
    logging.info("Displaying basic information")
    logging.info("Number of images: %d", len(meta_df))
    logging.info("Headers of metadata:\n %s", meta_df.head())

    # Display class distribution
    logging.info("Displaying class distribution")
    class_counts = meta_df["class_id"].value_counts()
    logging.info("Class counts: %s", class_counts)

    # Plot class distribution
    logging.info("Plotting class distribution")
    class_counts.plot(kind="bar", figsize=(12, 4), title="Class Distribution")
    plt.savefig("reports/figures/class_distribution.png")

    # split data into training and finetuning
    logging.info("Splitting data into training and finetuning")
    all_indices = meta_df.index.tolist()
    random.shuffle(all_indices)
    train_size = int(len(all_indices) * 0.7)
    train_indices = all_indices[:train_size]
    finetune_indices = all_indices[train_size:]

    train_df = meta_df.loc[train_indices].reset_index(drop=True)
    finetune_df = meta_df.loc[finetune_indices].reset_index(drop=True)

    logging.info("Training data size: %d", len(train_df))
    logging.info("Finetuning data size: %d", len(finetune_df))

    
    # Copy images to training and finetuning directories
    copy_images(train_df, TRAIN_DIR)
    copy_images(finetune_df, FINETUNE_DIR)

    # Save metadata to cs in training and finetuning directories
    train_df.to_csv(os.path.join(TRAIN_DIR, "metadata.csv"), index=False)
    finetune_df.to_csv(os.path.join(FINETUNE_DIR, "metadata.csv"), index=False)

    logging.info("Metadata saved to training and finetuning directories")
    
    # Statistics for training and finetuning data
    logging.info("Statistics for training data")
    logging.info(train_df.describe())

    logging.info("Statistics for finetuning data")
    logging.info(finetune_df.describe())

    logging.info("Data preparation complete")
    logging.info("Log file saved to: %s", log_file_path)

    
def copy_images(df: pd.DataFrame, dest_dir: str):
    logging.info("Copying images to %s", dest_dir)

    images_dest_dir = os.path.join(dest_dir, "images")
    if not os.path.exists(images_dest_dir):
        os.makedirs(images_dest_dir)

    for i, row in df.iterrows():
        src_path = os.path.join(PROCESSED_DATA_PATH, "CUB_200_2011", "images", row["image_path"])
        dest_path = os.path.join(images_dest_dir, row["image_path"])
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(src_path, dest_path)

def get_image_stats(image_path: str) -> dict:
    img_path = os.path.join(PROCESSED_DATA_PATH, "CUB_200_2011", "images", image_path)
    img = Image.open(img_path).convert("RGB")
    img_arr = np.array(img)

    mean_r = img_arr[:, :, 0].mean()
    mean_g = img_arr[:, :, 1].mean()
    mean_b = img_arr[:, :, 2].mean()

    h, w = img_arr.shape[0], img_arr.shape[1]

    num_pixels = h * w

    return pd.Series([mean_r, mean_g, mean_b, h, w, num_pixels], 
                     index=["mean_r", "mean_g", "mean_b", "img_width_px", "img_height_px", "num_pixels"])

def load_and_merge_data(PROCESSED_DATA_PATH: str) -> pd.DataFrame:
    images_df = pd.read_csv(
        os.path.join(PROCESSED_DATA_PATH, "CUB_200_2011", "images.txt"),
        sep=" ",
        header=None,
        names=["image_id", "image_path"],
    )

    labels_df = pd.read_csv(
        os.path.join(PROCESSED_DATA_PATH, "CUB_200_2011", "image_class_labels.txt"),
        sep=" ",
        header=None,
        names=["image_id", "class_id"],
    )

    bbox_df = pd.read_csv(
        os.path.join(PROCESSED_DATA_PATH, "CUB_200_2011", "bounding_boxes.txt"),
        sep=" ",
        header=None,
        names=["image_id", "x", "y", "width", "height"],
    )

    classes_df = pd.read_csv(
        os.path.join(PROCESSED_DATA_PATH, "CUB_200_2011", "classes.txt"),
        sep=" ",
        header=None,
        names=["class_id", "classes_name"]
    )

    logging.info("Data loaded")

    # Merge the data
    logging.info("Merging data")
    meta_df = images_df.merge(labels_df, on="image_id").merge(bbox_df, on="image_id").merge(classes_df, on="class_id")

    return meta_df


def download_archive() -> str:
    path = kagglehub.dataset_download("veeralakrishna/200-bird-species-with-11788-images")
    logging.info("Data downloaded to: %s", path)

    return path


def unzip_data(RAW_DATA_PATH: str, PROCESSED_DATA_PATH: str, TRAIN_DIR: str, FINETUNE_DIR: str):
    logging.info("Checking if data directories exist")
        
    # Download data from Kaggle
    logging.info("Downloading data from Kaggle")
    
    # CUB -> training data
    # segmentation -> segmentation data (additional information)

def extract_tgz(tgz_path: str, dest_path: str):
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(dest_path)

if __name__ == "__main__":
    main()


