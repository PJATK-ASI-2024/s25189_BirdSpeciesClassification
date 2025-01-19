import os
import random
import shutil
import json
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate random test images and metadata.")
parser.add_argument("--num-images", type=int, default=None, help="Number of random images to test (default: one per class).")
args = parser.parse_args()

# Define paths
dataset_path = os.path.abspath(os.path.join("..", "..", "data", "train", "images"))
output_test_folder = os.path.abspath(os.path.join("test", "images"))
metadata_output_path = os.path.abspath(os.path.join("test", "metadata.json"))
class_mapping_path = os.path.abspath(os.path.join("model", "class_mappings", "updated_class_mapping.json"))

# Load class mapping
with open(class_mapping_path, "r") as file:
    class_mapping = json.load(file)

# Create test folder if not exists
if not os.path.exists(output_test_folder):
    os.makedirs(output_test_folder)

# Initialize metadata list
metadata = []

# Get a list of all class folders
class_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

# Shuffle and limit class folders if num-images is specified
if args.num_images:
    class_folders = random.sample(class_folders, min(args.num_images, len(class_folders)))

# Generate test images and metadata
for class_folder in class_folders:
    class_folder_path = os.path.join(dataset_path, class_folder)
    images = [f for f in os.listdir(class_folder_path) if os.path.isfile(os.path.join(class_folder_path, f))]

    if images:
        # Pick one random image
        random_image = random.choice(images)
        src_path = os.path.join(class_folder_path, random_image)
        dest_path = os.path.join(output_test_folder, f"{class_folder}_{random_image}")
        shutil.copy(src_path, dest_path)

        # Extract class_id and class_name
        class_id = int(class_folder.split(".")[0])
        class_name = class_mapping.get(str(class_id), "Unknown")

        # Append metadata entry
        metadata.append({
            "image_name": f"{class_folder}_{random_image}",
            "class_id": class_id,
            "class_name": class_name
        })

# Save metadata as JSON
with open(metadata_output_path, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"Random images copied to '{output_test_folder}'")
print(f"Metadata saved to '{metadata_output_path}'")