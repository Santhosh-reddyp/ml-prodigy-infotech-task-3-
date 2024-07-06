# Image Classification Using Convolutional Neural Networks

## Overview
This project is part of my internship at Prodigy Infotech, where I developed a Convolutional Neural Network (CNN) to classify images of cats and dogs. The model achieves high accuracy by leveraging TensorFlow and Keras.

## Technical Implementation

- **Frameworks Used:** TensorFlow, Keras
- **Model Architecture:**
  - **Conv2D Layers:**
    - Layer 1: 32 filters, kernel size (3, 3), ReLU activation
    - Layer 2: 64 filters, kernel size (3, 3), ReLU activation
    - Layer 3: 128 filters, kernel size (3, 3), ReLU activation
  - **Additional Layers:**
    - Batch Normalization after each Conv2D layer
    - MaxPooling2D with pool size (2, 2) after each Conv2D layer
    - Dropout (0.1) after each dense layer
  - **Dense Layers:**
    - Dense layer with 128 units, ReLU activation
    - Dense layer with 64 units, ReLU activation
    - Output layer with 1 unit, Sigmoid activation

## Training and Validation

- **Dataset Preparation:**
  - Training images from `/content/train`
  - Validation images from `/content/test`
  - Images resized to 256x256 pixels
  - Images normalized to [0, 1] range

- **Model Compilation:**
  - Optimizer: Adam
  - Loss function: Binary Crossentropy
  - Metrics: Accuracy

- **Training:**
  - Epochs: 10
  - Batch size: 32

## Performance Metrics

- **Validation Accuracy:** 85% after 10 epochs
- **Training Loss:** Converged to 0.3

## Plots

```python
import matplotlib.pyplot as plt

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], color='red', label='train')
plt.plot(history.history['val_accuracy'], color='blue', label='validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], color='red', label='train')
plt.plot(history.history['val_loss'], color='blue', label='validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()
