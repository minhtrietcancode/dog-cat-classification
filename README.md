# Dog-Cat Image Classification

This project implements a deep learning solution for classifying images as either a dog or a cat. It leverages transfer learning with a pre-trained VGG-16 model, fine-tuned on a custom dataset.

## Table of Contents

- [Project Structure](#project-structure)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
- [Training the Model](#training-the-model)
  - [Using Google Colab (Recommended)](#using-google-colab-recommended)
- [Making Predictions](#making-predictions)
- [AutoCrawler Reference](#autocrawler-reference)

## Project Structure

```
dog-cat-classification/
├── AutoCrawler/            # Scripts for crawling images (e.g., dog, cat) from the web.
├── data/                   # Raw images downloaded by AutoCrawler.
├── train_test_data/        # Organized datasets for training, validation, and testing.
│   ├── testing_data/
│   │   ├── cat/
│   │   └── dog/
│   ├── training_data/
│   │   ├── cat/
│   │   └── dog/
│   └── validate_data/
│       ├── cat/
│       └── dog/
├── src/                    # Source code for the classification model.
│   └── model/
│       ├── model.py        # Defines the VGG-16 based model architecture.
│       ├── train.py        # Contains the training, validation, and testing logic.
│       └── predict.py      # Script for making predictions on new images.
├── evaluation/             # (Automatically created during training) Stores confusion matrices.
├── dog_cat_classifier_best.pth # (Generated during training) Trained model weights.
└── requirements.txt        # List of Python dependencies for the project.
└── README.md               # This README file.
```

## Features

-   **Automated Data Collection**: Utilizes an `AutoCrawler` (referenced below) to gather dog and cat images from the web.
-   **Efficient Data Preprocessing**: Images are resized and normalized on-the-fly during model training, eliminating the need for pre-saved transformed datasets.
-   **Transfer Learning**: Fine-tunes a pre-trained VGG-16 model, leveraging its robust feature extraction capabilities.
-   **GPU Accelerated Training**: Designed for efficient training on GPU environments like Google Colab.
-   **Automated Evaluation**: Generates and saves confusion matrices during validation and testing to visually assess model performance.
-   **Prediction Module**: A dedicated script to easily classify new images using the trained model.

## Getting Started

Follow these steps to set up and run the project.

### Prerequisites

-   Python 3.x
-   Git
-   A GitHub account (if cloning from your repository)
-   Google Account (for Google Colab, recommended for training)

### Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/dog-cat-classification.git
    cd dog-cat-classification
    ```
    *(Replace `your-username` with your actual GitHub username)*

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Data Preparation

Ensure your image data is organized in the `train_test_data/` directory as follows:

```
train_test_data/
├── testing_data/
│   ├── cat/
│   └── dog/
├── training_data/
│   ├── cat/
│   └── dog/
└── validate_data/
    ├── cat/
    └── dog/
```

-   Each `cat/` subdirectory should contain cat images.
-   Each `dog/` subdirectory should contain dog images.

You can use the `AutoCrawler` tool (referenced below) to collect images into the `data/` directory, and then manually sort/split them into the `train_test_data/` structure.

## Training the Model

Training deep learning models can be resource-intensive. It is **highly recommended** to train the model on a GPU-accelerated environment like Google Colab to avoid straining your local machine.

### Using Google Colab (Recommended)

1.  **Open Google Colab**: Go to [Google Colab](https://colab.research.google.com/) and create a new notebook (`File > New notebook`).

2.  **Change Runtime Type to GPU**: In the Colab notebook, go to `Runtime > Change runtime type` and select `GPU` under `Hardware accelerator`, then click `Save`.

3.  **Clone your GitHub Repository**:
    In the first code cell, replace `your-username` and `dog-cat-classification` with your actual GitHub details and run:
    ```python
    !git clone https://github.com/your-username/dog-cat-classification.git
    %cd dog-cat-classification
    ```

4.  **Install Dependencies**:
    In a new code cell, run:
    ```python
    !pip install -r requirements.txt
    ```

5.  **Run the Training Script**:
    In a new code cell, navigate to the model directory and execute the training script:
    ```python
    %cd src/model
    !python train.py
    ```
    This will start the training process. The best performing model weights (`dog_cat_classifier_best.pth`) and confusion matrices will be saved in the `evaluation/` directory (created at the project root).

6.  **Download Results (Optional)**:
    To download the trained model or evaluation images:
    ```python
    # To go back to project root
    %cd ../..
    
    from google.colab import files
    # Download the best model weights
    files.download('src/model/dog_cat_classifier_best.pth')
    # Example: Download a specific confusion matrix (adjust filename)
    files.download('evaluation/confusion_matrix_epoch_1.png') 
    ```

## Trained Model Checkpoint

The trained `dog_cat_classifier_best.pth` model, which is too large for direct GitHub hosting, can be downloaded from the following Google Drive link:

[Download Trained Model from Google Drive](https://colab.research.google.com/drive/10_YAyNpqjxdOa4_CF6qcuByzKd56F4Fp?usp=sharing)

Please download this file and place it into your `src/model/` directory before attempting to make predictions.

## Making Predictions

To use your trained model to classify new images:

1.  **Download the trained model checkpoint** (`dog_cat_classifier_best.pth`) from the link in the [Trained Model Checkpoint](#trained-model-checkpoint) section and place it into your `src/model/` directory.
2.  Navigate to the `src/model/` directory in your terminal:
    ```bash
    cd src/model
    ```
3.  Run the `predict.py` script. You will need to modify the `image_path` variable within the `predict.py` file to point to the image you want to classify.

    ```bash
    python predict.py
    ```
    The script will print the predicted class for the specified image.

## AutoCrawler Reference

This project utilizes the `AutoCrawler` for initial image data collection. You can find more details about it here:

[AutoCrawler GitHub Repository](https://github.com/YoongiKim/AutoCrawler)
