import logging
import os
import shutil
import random
import tarfile
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image

# Global variables
PROCESSED_DATA_PATH = "data/processed"
TRAIN_DIR = "data/train"
FINETUNE_DIR = "data/finetune"

def setup_logging() -> str:
    """Set up logging configuration and return the log file path."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    log_file_path = f'logs/prep/data_prep_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
    return log_file_path

def create_directories():
    """Create necessary directories if they do not exist."""
    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH)
    if not os.path.exists(TRAIN_DIR):
        os.makedirs(TRAIN_DIR)
    if not os.path.exists(FINETUNE_DIR):
        os.makedirs(FINETUNE_DIR)
    logging.info("Data directories checked/created")

def download_archive() -> str:
    """Download the dataset from Kaggle and return the path to the downloaded data."""
    path = kagglehub.dataset_download("veeralakrishna/200-bird-species-with-11788-images")
    logging.info("Data downloaded to: %s", path)
    return path

def unzip_data(RAW_DATA_PATH: str):
    """Unzip the downloaded data."""
    logging.info("Unzipping data")
    extract_tgz(os.path.join(RAW_DATA_PATH, "CUB_200_2011.tgz"), PROCESSED_DATA_PATH)
    extract_tgz(os.path.join(RAW_DATA_PATH, "segmentations.tgz"), PROCESSED_DATA_PATH)
    logging.info("Data extraction complete")

def extract_tgz(tgz_path: str, dest_path: str):
    """Extract a .tgz file to the specified destination path."""
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(dest_path)

def load_and_merge_data() -> pd.DataFrame:
    """Load and merge data from the processed data path."""
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

    logging.info("Data loaded and merged")
    meta_df = images_df.merge(labels_df, on="image_id").merge(bbox_df, on="image_id").merge(classes_df, on="class_id")
    return meta_df

def get_image_stats(image_path: str) -> pd.Series:
    """Calculate and return image statistics."""
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

def copy_images(df: pd.DataFrame, dest_dir: str):
    """Copy images to the specified destination directory."""
    logging.info("Copying images to %s", dest_dir)
    images_dest_dir = os.path.join(dest_dir, "images")
    if not os.path.exists(images_dest_dir):
        os.makedirs(images_dest_dir)

    for _, row in df.iterrows():
        src_path = os.path.join(PROCESSED_DATA_PATH, "CUB_200_2011", "images", row["image_path"])
        dest_path = os.path.join(images_dest_dir, row["image_path"])
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(src_path, dest_path)

def split_data(meta_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """Split the data into training and finetuning datasets."""
    all_indices = meta_df.index.tolist()
    random.shuffle(all_indices)
    train_size = int(len(all_indices) * 0.7)
    train_indices = all_indices[:train_size]
    finetune_indices = all_indices[train_size:]

    train_df = meta_df.loc[train_indices].reset_index(drop=True)
    finetune_df = meta_df.loc[finetune_indices].reset_index(drop=True)
    return train_df, finetune_df

def main():
    log_file_path = setup_logging()
    logging.info("Starting data preparation script")

    create_directories()
    RAW_DATA_PATH = download_archive()
    unzip_data(RAW_DATA_PATH)

    meta_df = load_and_merge_data()
    meta_df = meta_df.join(meta_df["image_path"].apply(get_image_stats))

    logging.info("Number of images: %d", len(meta_df))
    logging.info("Headers of metadata:\n %s", meta_df.head())

    class_counts = meta_df["class_id"].value_counts()
    logging.info("Class counts: %s", class_counts)

    class_counts.plot(kind="bar", figsize=(12, 4), title="Class Distribution")
    plt.savefig("reports/figures/class_distribution.png")

    train_df, finetune_df = split_data(meta_df)

    logging.info("Training data size: %d", len(train_df))
    logging.info("Finetuning data size: %d", len(finetune_df))

    copy_images(train_df, TRAIN_DIR)
    copy_images(finetune_df, FINETUNE_DIR)

    train_df.to_csv(os.path.join(TRAIN_DIR, "metadata.csv"), index=False)
    finetune_df.to_csv(os.path.join(FINETUNE_DIR, "metadata.csv"), index=False)

    logging.info("Metadata saved to training and finetuning directories")
    logging.info("Statistics for training data:\n%s", train_df.describe())
    logging.info("Statistics for finetuning data:\n%s", finetune_df.describe())

    logging.info("Data preparation complete")
    logging.info("Log file saved to: %s", log_file_path)

if __name__ == "__main__":
    main()
