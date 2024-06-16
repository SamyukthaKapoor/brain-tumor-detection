# Brain Tumor Detection

## Introduction
This project aims to detect brain tumors using machine learning techniques. It involves the preprocessing of MRI images, training a convolutional neural network (CNN) to classify the images, and evaluating the model's performance.

## Project Structure
- **Data Preprocessing**: Includes steps such as resizing images, normalizing pixel values, and splitting the dataset into training and testing sets.
- **Model Building**: Utilizes a CNN built with TensorFlow and Keras to classify MRI images as either containing a tumor or not.
- **Model Evaluation**: Assesses the performance of the model using metrics such as accuracy, precision, recall, and F1-score.

## Files in the Repository
- **brain_tumor_detection.ipynb**: Jupyter notebook containing the code for data preprocessing, model building, and evaluation.
- **dataset/**: Directory containing the MRI images used for training and testing the model.
- **README.md**: Project overview and instructions.

## Requirements
To run this project, you need the following libraries installed:
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib

You can install the required libraries using the following command:
  ## ```bash
pip install tensorflow keras opencv-python numpy matplotlib

## Data Preprocessing

1. Loading the Data: The dataset contains MRI images labeled as either having a tumor or not.
2. Image Resizing: All images are resized to 128x128 pixels.
3. Normalization: Pixel values are normalized to the range [0, 1].
4. Data Splitting: The dataset is split into training and testing sets.

## Model Building
The model is a convolutional neural network (CNN) with the following architecture:

Conv2D: 32 filters, 3x3 kernel, ReLU activation
MaxPooling2D: 2x2 pool size
Conv2D: 64 filters, 3x3 kernel, ReLU activation
MaxPooling2D: 2x2 pool size
Flatten
Dense: 128 units, ReLU activation
Dropout: 0.5
Dense: 1 unit, Sigmoid activation

## Model Evaluation
The model's performance is evaluated using the following metrics:

Accuracy
Precision
Recall
F1-score

## Results
The model achieved an accuracy of 92%, demonstrating its effectiveness in detecting brain tumors from MRI images.

## Conclusion
This project demonstrates the use of deep learning techniques for brain tumor detection. The developed model can assist medical professionals in diagnosing brain tumors more accurately and efficiently.

## License
This project is licensed under the MIT License.
