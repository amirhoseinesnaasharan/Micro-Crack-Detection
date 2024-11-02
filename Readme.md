# Micro-Crack Detection in Solar Panels

This project aims to detect micro-cracks in solar panel images using a Convolutional Neural Network (CNN). The model preprocesses images, trains with labeled data, and predicts defects cell-wise, saving the results in a CSV file for analysis.

## Features

- **Image Preprocessing**: Load and preprocess images, including resizing and normalization.
- **CNN Model**: Build and train a CNN for binary classification (defect/no defect).
- **Cell-wise Detection**: Split images into individual cells and detect defects in each cell.
- **CSV Output**: Save the detection results in a structured CSV format.

## Requirements

- Python 3.x
- OpenCV
- TensorFlow
- NumPy
- scikit-learn
- tkinter (for file selection)