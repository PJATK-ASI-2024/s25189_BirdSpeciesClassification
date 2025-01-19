import os
import sys
import json
import subprocess
import requests
import time
import pytest
from sklearn.metrics import accuracy_score, classification_report
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Test the prediction endpoint locally or in a Docker container.")
parser.add_argument("--mode", choices=["local", "container"], required=True, help="Run mode: 'local' or 'container'")
args = parser.parse_args()

# Constant
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure paths based on mode
if args.mode == "local":
    TEST_FOLDER = os.path.join(BASE_DIR, "images")
    METADATA_FILE = os.path.join(BASE_DIR, "metadata.json")
    API_START_COMMAND = ["uvicorn", "main:main", "--host", "127.0.0.1", "--port", "5000"]
elif args.mode == "container":
    TEST_FOLDER = "/app/images"  # Adjust to the Docker container structure
    METADATA_FILE = "/app/test/metadata.json"
    API_START_COMMAND = ["python", "main.py"]  # Update if container uses a different command

ENDPOINT_URL = "http://localhost:5000/predict"
API_READY_TIMEOUT = 10  # Time to wait for the API to start (in seconds)

# Load metadata
with open(METADATA_FILE, "r") as f:
    metadata = json.load(f)

# Test data collection from metadata
test_data = [
    {
        "image_path": os.path.join(TEST_FOLDER, item["image_name"]),
        "class_id": item["class_id"],
        "class_name": item["class_name"]
    }
    for item in metadata
]


def start_endpoint():
    """Start the API endpoint."""
    print(f"Starting the API endpoint in {args.mode} mode...")
    process = subprocess.Popen(API_START_COMMAND)
    time.sleep(API_READY_TIMEOUT)  # Wait for the server to start
    return process


def stop_endpoint(process):
    """Stop the API endpoint."""
    print(f"Stopping the API endpoint in {args.mode} mode...")
    process.terminate()
    process.wait()


@pytest.fixture(scope="session", autouse=True)
def api_server():
    """Fixture to start and stop the API server for the test session."""
    process = start_endpoint()
    yield  # Run the tests
    stop_endpoint(process)


# Test function
@pytest.mark.parametrize("data", test_data)
def test_predict(data):
    image_path = data["image_path"]
    true_class_id = data["class_id"]

    # Open and send the image to the API
    with open(image_path, "rb") as img:
        response = requests.post(ENDPOINT_URL, files={"file": img})

    # Ensure the API call is successful
    assert response.status_code == 200, f"API call failed for {image_path}"

    # Parse the prediction
    prediction = response.json()
    predicted_class_id = prediction.get("class_id")

    # Ensure prediction is returned
    assert predicted_class_id is not None, f"No prediction returned for {image_path}"

    # Compare true vs predicted
    assert predicted_class_id == true_class_id, f"Incorrect prediction for {image_path}: expected {true_class_id}, got {predicted_class_id}"


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def evaluate_predictions():
    y_true = []
    y_pred = []

    for data in test_data:
        image_path = data["image_path"]
        true_class_id = data["class_id"]

        with open(image_path, "rb") as img:
            response = requests.post(ENDPOINT_URL, files={"file": img})

            # Ensure the API response is successful
            if response.status_code != 200:
                print(f"Error: API call failed for {image_path}. Status code: {response.status_code}")
                continue

            predicted_class_id = response.json().get("class_id")

            # Log if prediction is missing
            if predicted_class_id is None:
                print(f"Error: No prediction returned for {image_path}.")
                continue

            y_true.append(true_class_id)
            y_pred.append(predicted_class_id)

    # Check if y_true and y_pred have valid data
    if not y_true or not y_pred:
        print("Error: No valid predictions or ground truth data available for evaluation.")
        return

    # Extract unique class IDs from y_true and y_pred
    unique_class_ids = sorted(set(y_true) | set(y_pred))

    # Create a mapping for target names
    id_to_class_name = {item["class_id"]: item["class_name"] for item in metadata}
    target_names = [id_to_class_name.get(class_id, f"Unknown ({class_id})") for class_id in unique_class_ids]

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=unique_class_ids, target_names=target_names, zero_division=0, output_dict=True)

    # Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_class_ids)

    # Summarize misclassifications
    misclassifications = cm.sum(axis=1) - cm.diagonal()
    misclassified_classes = [
        (id_to_class_name.get(class_id, f"Unknown ({class_id})"), misclassifications[i])
        for i, class_id in enumerate(unique_class_ids)
    ]
    misclassified_classes = sorted(misclassified_classes, key=lambda x: x[1], reverse=True)[:10]  # Top 10 misclassified classes

    # Plot top misclassified classes
    misclassified_figure_path = os.path.join(BASE_DIR, "top_misclassified_classes.png")
    plt.figure(figsize=(12, 6))
    class_names, misclass_counts = zip(*misclassified_classes)
    plt.barh(class_names, misclass_counts, color="salmon")
    plt.xlabel("Number of Misclassifications")
    plt.ylabel("Class Name")
    plt.title("Top Misclassified Classes")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(misclassified_figure_path)
    plt.close()

    # Generate report content
    markdown_content = f"# Test Report\n\n"
    markdown_content += f"## Summary\n\n"
    markdown_content += f"- **Accuracy**: {accuracy:.2f}\n"
    markdown_content += f"- **Number of Images Tested**: {len(y_true)}\n"
    markdown_content += f"- **Unique Classes Tested**: {len(unique_class_ids)}\n\n"

    markdown_content += f"## Classification Report\n\n"
    markdown_content += "| Class Name | Precision | Recall | F1-Score | Support |\n"
    markdown_content += "|------------|-----------|--------|----------|---------|\n"

    for class_id, metrics in zip(unique_class_ids, report.values()):
        if isinstance(metrics, dict):
            class_name = id_to_class_name.get(class_id, f"Unknown ({class_id})")
            precision = metrics["precision"]
            recall = metrics["recall"]
            f1_score = metrics["f1-score"]
            support = metrics["support"]
            markdown_content += f"| {class_name} | {precision:.2f} | {recall:.2f} | {f1_score:.2f} | {int(support)} |\n"

    markdown_content += f"\n## Top Misclassified Classes\n\n"
    markdown_content += f"![Top Misclassified Classes](top_misclassified_classes.png)\n\n"

    # Save to file
    report_file = os.path.join(BASE_DIR, "test_report.md")
    with open(report_file, "w") as f:
        f.write(markdown_content)

    print(f"Test report saved to '{report_file}'")
    print(f"Top misclassified classes chart saved to '{misclassified_figure_path}'")





if __name__ == "__main__":
    if args.mode == "local":
        # For local runs, start the API and run tests
        process = start_endpoint()
        try:
            evaluate_predictions()
        finally:
            stop_endpoint(process)
    elif args.mode == "container":
        # For containerized runs, directly execute the test
        evaluate_predictions()
