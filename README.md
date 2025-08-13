# Automated Multiclass Dermatological Diagnosis Prediction

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Results](#results)
- [Contribution Guidelines](#contribution-guidelines)

## Overview

This repository contains a deep learning model for automated multiclass dermatological diagnosis prediction from skin images. The model is based on an improved Mobile-Net architecture and has been trained on the HAM10000 dataset. It includes features for real-time prediction from various camera sources and provides AI-based health suggestions.

## Features

-   **Multiple Input Sources:**
    -   **Webcam:** Perform real-time diagnosis using your computer's webcam.
    -   **IP Camera:** Connect to an IP camera stream for remote monitoring and prediction.
    -   **Image Upload:** Upload a static image for analysis.
-   **AI-Based Health Suggestions:** After a diagnosis is made, the application provides custom health suggestions and precautions based on the predicted skin condition. These suggestions are sourced from an internal knowledge base (dictionary).

## Dataset

The dataset used for training and evaluation is the HAM10000 dataset, which consists of images of various skin lesions.

## Model Architecture

The model is built using an improved Mobile-Net architecture, optimized for dermatological image classification. The architecture has been fine-tuned to achieve better accuracy and performance in predicting multiclass skin diseases.

## Usage

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/sethumadhavan7/skp.git
    cd skp
    ```

2. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Application:**

    ```bash
    python app.py
    ```

## Results

The model achieves a high accuracy on the test set from the HAM10000 dataset. 


## Contribution Guidelines

If you'd like to contribute to this project, please follow these guidelines:

1. Fork the repository.
2. Create a new branch for your feature: `git checkout -b feature-name`.
3. Make your changes and commit them: `git commit -m 'Description of changes'`.
4. Push to the branch: `git push origin feature-name`.
5. Submit a pull request.
