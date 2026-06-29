# Week 5 Assignment: Image Noise Removal Using a Convolutional Autoencoder

## Objective

Design and implement a Deep Learning model capable of removing noise from images using a Convolutional Autoencoder. The model is trained on the MNIST handwritten digit dataset and learns to reconstruct clean images from noisy inputs.

## Problem Statement

Image denoising is a common computer vision task where the goal is to recover a clean image from a noisy version. In this project, Gaussian noise is added to MNIST digit images, and a Convolutional Autoencoder is trained to remove the noise while preserving important image features.

## Dataset

- Dataset: MNIST Handwritten Digits
- Number of Training Images: 60,000
- Number of Test Images: 10,000
- Image Size: 28 × 28 pixels
- Color Format: Grayscale

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

## Project Workflow

### 1. Data Loading and Preprocessing
- Load MNIST dataset.
- Normalize pixel values to the range [0,1].
- Reshape images to include channel dimensions.

### 2. Noise Generation
- Add Gaussian noise to clean images.
- Clip pixel values to maintain valid image ranges.

### 3. Model Architecture
A Convolutional Autoencoder consisting of:

#### Encoder
- Conv2D Layer
- MaxPooling2D Layer
- Conv2D Layer
- MaxPooling2D Layer

#### Decoder
- Conv2D Layer
- UpSampling2D Layer
- Conv2D Layer
- UpSampling2D Layer
- Output Conv2D Layer with Sigmoid Activation

### 4. Training
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam
- Early Stopping used to prevent overfitting

### 5. Evaluation
- Test MSE
- Test MAE
- Visual comparison of:
  - Original Images
  - Noisy Images
  - Denoised Images

### 6. Model Saving
The trained model is saved in Keras format for future inference.

## Results

The autoencoder successfully learns to remove Gaussian noise from MNIST images and reconstructs cleaner versions while preserving digit structure and important visual features.

## Output Visualizations

- Clean vs Noisy Images
- Training and Validation Loss Curves
- Clean vs Noisy vs Denoised Image Comparison

## Conclusion

This project demonstrates the effectiveness of Convolutional Autoencoders for image denoising tasks. The model successfully reconstructs clean images from noisy inputs and can be extended to more complex image restoration applications.

## Author

Akshat Agarwal

## Internship

Celebal Technologies – Celebal Excellence Internship (CEI) 2026
