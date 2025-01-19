import pandas as pd
import json
import os

# Define paths
PROCESSED_DATA_PATH = "data/processed"  # Adjust if necessary
OUTPUT_PATH = "class_mapping.json"

# Load class data
def generate_class_mapping():
    classes_file = os.path.join(PROCESSED_DATA_PATH, "CUB_200_2011", "classes.txt")
    
    # Read the classes file
    classes_df = pd.read_csv(classes_file, sep=" ", header=None, names=["class_id", "class_name"])
    
    # Convert to dictionary
    class_mapping = dict(zip(classes_df["class_id"], classes_df["class_name"]))
    
    # Save as JSON
    with open(OUTPUT_PATH, "w") as f:
        json.dump(class_mapping, f, indent=4)
    print(f"Class mapping saved to {OUTPUT_PATH}")


    # Load the class mapping file
    with open("class_mapping.json", "r") as file:
        class_mapping = json.load(file)

    # Process the class names
    updated_class_mapping = {
        class_id: " ".join(class_name.split(".", 1)[1].replace("_", " ").split())
        for class_id, class_name in class_mapping.items()
    }

    # Save the updated mapping to a new JSON file
    with open("updated_class_mapping.json", "w") as file:
        json.dump(updated_class_mapping, file, indent=4)

    print("Updated class mapping saved to 'updated_class_mapping.json'")


if __name__ == "__main__":
    generate_class_mapping()