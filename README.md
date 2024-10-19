# Vision Transformer (ViT) for Auslan Hand Sign Classification

This repository contains the implementation of a Vision Transformer (ViT) model for classifying Auslan hand sign digits (0-9). The model is trained on a dataset of grayscale images and evaluated using metrics such as accuracy, precision, recall, and F1-score.

## Table of Contents
- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Training Process](#training-process)
- [Results](#results)


## Introduction
The Vision Transformer (ViT) model processes images by dividing them into small patches, embedding each patch, and passing these embeddings through transformer encoder layers. This model uses self-attention mechanisms to capture relationships between image regions, making it highly effective for tasks that require global feature understanding.

This project focuses on classifying Auslan hand sign digits from grayscale images using the ViT architecture. The model is designed to handle real-world variability in hand signs and is tested on a dataset of 20,000 images.

## Model Architecture
The ViT model architecture used in this project includes:
- **Patch Embedding**: Each input image is divided into fixed-size patches (e.g., 16x16). These patches are flattened and passed through a linear layer to generate embeddings.
- **Positional Embedding**: To retain spatial information, positional embeddings are added to the patch embeddings.
- **Transformer Encoder**: The embedded patches are processed through a series of transformer encoder layers. Each encoder consists of:
  - Multi-Head Self-Attention: Allows the model to attend to different parts of the image.
  - Feed-Forward Neural Network (MLP): Applies non-linear transformations to learn complex patterns.
- **Classification Token ([CLS])**: A special token is added to the sequence, whose output is used for classification.
- **MLP Head**: After processing the embedded patches, the output of the [CLS] token is passed through a fully connected layer to predict the class.

For more details, see the diagram below (from the `ViT_model_diagram.png` file in this repo).

## Dataset
The dataset consists of 20,000 grayscale images of Auslan hand signs representing digits (0-9). Each image is 224x224 pixels, organized into folders named by the respective digits. Images were augmented using techniques such as random rotation, horizontal flipping, and Gaussian blur to improve model generalization.

### Dataset Structure:
```
data/
├── 0/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── 1/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
...
├── 9/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
```


## Training Process
The model is trained on the 20,000 images, with an 80/20 train-validation split. Training involves:
- **Optimizer**: Adam optimizer with a learning rate of 1e-4.
- **Loss Function**: Cross-entropy loss with class weighting to handle class imbalance.
- **Batch Size**: 4.
- **Epochs**: 20 epochs with early stopping based on validation accuracy.

Training metrics such as loss, accuracy, precision, recall, and F1-score are tracked across epochs.

## Results
Here are the key results from training and validation:

- **Training Loss**: The training loss decreases steadily over epochs, indicating that the model is learning effectively.
- **Validation Accuracy**: Fluctuating, suggesting that the model needs further generalization improvements.
- **Precision/Recall/F1-Score**: These metrics are relatively low, signaling that the model struggles to consistently identify all classes accurately.

Visual results for loss, accuracy, precision, recall, and F1-score are available in the `results/` folder.


