# MNIST-Handwritten-Digit-Recognition

A neural network built with Keras to classify handwritten digits from the MNIST dataset.

## Overview

This project trains a fully-connected (dense) neural network to recognize handwritten digits (0–9) using the classic MNIST dataset. It covers the full pipeline: data loading, preprocessing, model training, evaluation, and inference on an external image.

## Project Structure
```
MNIST_Image_Recognition.ipynb   # Main notebook
```

## Requirements
```
numpy
matplotlib
keras / tensorflow
Pillow
opencv-python (cv2)
requests
```

Install with:
```bash
pip install numpy matplotlib tensorflow Pillow opencv-python requests
```

## Dataset

- **Source:** MNIST via `keras.datasets.mnist`
- **Training set:** 60,000 grayscale images (28×28 pixels)
- **Test set:** 10,000 grayscale images (28×28 pixels)
- **Classes:** 10 (digits 0–9), roughly balanced (~5,400–6,700 samples per class)

## How It Works

### 1. Data Loading & Validation
The MNIST dataset is loaded and validated with assertions to confirm image dimensions and label alignment.

### 2. Preprocessing
- **One-hot encoding** applied to labels using `to_categorical`
- **Normalization:** pixel values scaled from [0, 255] to [0, 1]
- **Flattening:** 28×28 images reshaped to 784-element vectors for the dense layers

### 3. Model Architecture

A sequential fully-connected network:

| Layer  | Units | Activation |
|--------|-------|------------|
| Input  | 10    | ReLU       |
| Hidden | 30    | ReLU       |
| Hidden | 10    | ReLU       |
| Output | 10    | Softmax    |

- **Optimizer:** Adam (lr=0.01)
- **Loss:** Categorical cross-entropy
- **Metric:** Accuracy

### 4. Training
```
Epochs:     10
Batch size: 200
Validation: 10% split
```

Training and validation loss/accuracy are plotted per epoch. Overfitting can occur at higher epoch counts as validation loss diverges from training loss.

### 5. Inference on External Image

A custom digit image is fetched from the web, preprocessed with OpenCV (resized to 28×28, converted to grayscale, inverted), normalized, and passed through the trained model for prediction.

## Results

The model achieves approximately **~95% test accuracy** after 10 epochs.

## Notes

- `model.predict_classes()` is deprecated in newer TensorFlow versions. Use `np.argmax(model.predict(x), axis=-1)` instead.
- A linear/dense network is **not scalable** for more complex image datasets — a CNN would be the natural next step.
