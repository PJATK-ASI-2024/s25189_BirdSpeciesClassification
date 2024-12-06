import kagglehub
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Log the start of the script
logging.info("Starting data preparation script")

# Download latest version
path = kagglehub.dataset_download("veeralakrishna/200-bird-species-with-11788-images")

logging.info("Data downloaded to: %s", path)
