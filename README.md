# Bird species classification

## Introduction

Main aim of this project is to create a machine learning model that can be used to classify bird species based on images and metadata attached to an image like:

    - Longitude and Latitude of the picture made
    - Time of year or month of the picture taken (some species show up seasonally)
    - Size of an image
    - Mean RGB values (or other colour range)
    - Amount of pixels dominated by a particular colour (brown, white, black etc. -> feathering of a bird)

## Business problem

Bird species classification requires extensive knowledge in the field of ornitology. Automatization of this process can allow for:

    - Less time taken to analyze and predict an actual species
    - Take off some workload from scientists and hobbyist.
    - Usage in mobile application (ie. take a photo, analyze it on the go and show exactly what kind of bird it is)

## Data and its source

Data come from a public repository on Kaggle [200 Bird Species](https://www.kaggle.com/datasets/veeralakrishna/200-bird-species-with-11788-images/code)

It consists of:

    - 11,788 Images showing 200 different birds
    - Annotations per image: 
      - 15 Part location
      - 312 Binary attributes
      - 1 Bounding box

Data will be split 70/30 (training/fine-tuning)

## Aim of this projec and structure

    The aim of this project is to build and train a ML model. It will take a photo and its metadata and based on that will classify a bird to a particular species. 
