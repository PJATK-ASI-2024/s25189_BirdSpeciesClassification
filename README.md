# Bird Species Classification

## Introduction

The main aim of this project is to create a machine learning model that can classify bird species based on images and metadata such as:

- Longitude and Latitude of the picture
- Time of year or month the picture was taken (some species are seasonal)
- Size of the image
- Mean RGB values (or other color ranges)
- Amount of pixels dominated by a particular color (e.g., brown, white, black)

## Business Problem

Bird species classification requires extensive knowledge in ornithology. Automating this process can:

- Reduce the time needed to analyze and predict bird species
- Decrease the workload for scientists and hobbyists
- Enable usage in mobile applications (e.g., take a photo, analyze it on the go, and identify the bird species)

## Data and Source

The data comes from a public repository on Kaggle: [200 Bird Species](https://www.kaggle.com/datasets/veeralakrishna/200-bird-species-with-11788-images/code).

It consists of:

- 11,788 images showing 200 different bird species
- Annotations per image:
    - 15 part locations
    - 312 binary attributes
    - 1 bounding box

The data will be split 70/30 for training and fine-tuning.

## Project Aim and Structure

The aim of this project is to build and train a machine learning model that can classify bird species based on a photo and its metadata.

## Testing the Model

### 1. Building and Running the Docker Container

#### Build the Docker Image

Run the following command in the `backend_container` directory containing the `Dockerfile`:

```sh
docker build -t bird-classifier-api .
```

#### Run the Docker Container

```sh
docker run -p 5000:5000 bird-classifier-api
```

### 2. Testing the API

#### Overview

A REST API for classifying bird images using a pre-trained ResNet model.

#### Requirements

- Docker
- Python 3.9+

#### Using `curl`

```sh
curl -X POST -F "file=@path/to/image.jpg" http://localhost:5000/predict
```

#### Using Postman

1. Select the `POST` method.
2. Set the URL to `http://localhost:5000/predict`.
3. Under `Body`, choose `form-data` and add a key named `file` with the image file as its value.

#### Expected Result

```json
{"class_id": 74, "class_name": "Florida Jay"}
```

#### API Endpoints

- POST `/predict`: Accepts an image and returns the predicted class.

### 3. Running Prepared Automated Testing

The API endpoints run in two modes: local (`--mode local`) and container (pulled from the environment variable).

#### Generating Test Data

From the previously downloaded dataset (using `01_data_prep.py`), you can generate random photographs and metadata to evaluate the API:

```sh
generate_test_images.py --num-images-per-class 3 --max-images 20
```

- `num-images`: The amount of test data you want to generate
- `num-images-per-class`: The number of images per class to include

After that, you will have a random selection of images and a `metadata.json` file for testing.

#### Running the Test

To run the test, launch `test_prediction.py` with the argument `--mode local`:

```sh
python -u "absolute_path_to_test_file/test_prediction.py" --mode local
```

This will generate a test report in .md format.

#### Example Test Report

# Test Report

## Summary

- **Accuracy**: 0.80
- **Number of Images Tested**: 5
- **Unique Classes Tested**: 6

## Classification Report

| Class Name | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| Unknown | 0.00 | 0.00 | 0.00 | 0 |
| Pelagic Cormorant | 0.00 | 0.00 | 0.00 | 1 |
| Scissor-tailed Flycatcher | 1.00 | 1.00 | 1.00 | 1 |
| Pied-billed Grebe | 1.00 | 1.00 | 1.00 | 1 |
| Heermann's Gull | 1.00 | 1.00 | 1.00 | 1 |
| Cape May Warbler | 1.00 | 1.00 | 1.00 | 1 |

## Top Misclassified Classes

Here will be a bar chart representing top misclassified classes.