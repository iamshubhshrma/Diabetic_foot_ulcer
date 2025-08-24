# Diabetic Foot Ulcer Detection using Thermography & Deep Learning

This project implements a deep learning model for the early detection of Diabetic Foot Ulcers (DFU) using non-invasive thermographic images. It leverages transfer learning with a pre-trained ResNet50 model to achieve high accuracy in classifying feet as either 'Diabetic' or 'Normal'.



## 📋 About The Project

Diabetic foot ulceration is a severe complication of diabetes that can lead to infection, hospitalization, and even amputation. Early detection is crucial for effective management and prevention of severe outcomes. This project provides a non-invasive, automated solution using thermal imaging and a Convolutional Neural Network (CNN) to identify temperature anomalies that may indicate ulceration risk.

The model is built using TensorFlow and Keras, fine-tuning a ResNet50 architecture pre-trained on the ImageNet dataset.

---

## 💾 Dataset

The model was trained on a public dataset of thermographic images of diabetic and normal feet. The dataset is available on Kaggle.

- **Dataset Link:** [Thermography Images of Diabetic Foot on Kaggle](https://www.kaggle.com/datasets/vuppalaadithyasairam/thermography-images-of-diabetic-foot)

---

## 🛠️ Model Architecture & Technology

The model uses a `Sequential` architecture built on top of a frozen **ResNet50** base. The classification head consists of:
- **GlobalAveragePooling2D:** To reduce feature map dimensions.
- **Dense Layer (256 units, ReLU):** For feature interpretation.
- **BatchNormalization & GaussianNoise:** To improve training stability and robustness.
- **Dropout (25%):** To prevent overfitting.
- **Dense Layer (1 unit, Sigmoid):** The output layer for binary classification.

### **Tech Stack**
- Python 3
- TensorFlow & Keras
- NumPy
- Pandas
- Matplotlib

---

## 📊 Results

The model was trained for 15 epochs and achieved strong performance on the validation dataset. The use of data augmentation (rotation, shear, zoom, flips) helped create a more robust model.

- **Best Validation Accuracy: 93.3%**
- **Validation AUC: 95.9%**
- **Validation Precision: 97.7%**

The final trained model is saved as `model.keras`.


---

## 🚀 Getting Started

To get a local copy up and running, follow these simple steps.

### **Prerequisites**

You'll need Python and pip installed.

### **Installation**

1. Clone the repo:
   ```sh
   git clone https://github.com/iamshubhshrma/Diabetic_foot_ulcer.git
