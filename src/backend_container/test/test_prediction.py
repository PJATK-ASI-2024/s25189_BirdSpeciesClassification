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

# Constants
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
    target_names = [id_to_class_name.get(class_id, "Unknown") for class_id in unique_class_ids]

    # Calculate and display metrics
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=unique_class_ids, target_names=target_names, zero_division=0)

    print(f"Accuracy: {accuracy:.2f}")
    print(report)



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
