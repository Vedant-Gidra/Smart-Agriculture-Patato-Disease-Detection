# 🌱 Smart Agriculture: CNN-Based Potato Disease Detection System

## Introduction

Potatoes are one of the world’s most significant crops, but their production is threatened by various diseases, leading to yield losses and economic damage for farmers. The major diseases affecting potatoes include:

- **Early Blight**  
  Caused by *Alternaria solani*, early blight results in dark spots on leaves and stems, reducing photosynthesis and leading to lower yields.

- **Late Blight**  
  Caused by *Phytophthora infestans*, late blight is the disease responsible for the historical Irish potato famine. It spreads rapidly, causing dark lesions on leaves, stems, and tubers, eventually rotting the entire plant.

- **Healthy**  
  Indicates leaves or plants free from visible symptoms of disease.

> **Dataset:**  
> The dataset for this project can be found here: [Potato Leaf Disease Dataset on Kaggle](https://www.kaggle.com/datasets/aarishasifkhan/plantvillage-potato-disease-dataset)

---

## Problem Statement

Early detection of potato diseases is crucial to prevent large-scale crop losses. However, manual inspection is time-consuming and prone to human error. An automated image classification system can help farmers identify diseases quickly and accurately.

---

## Solution Overview

This project proposes a **deep learning model** using a **Dual-Branch Convolutional Neural Network (CNN)** with an **attention mechanism** to detect potato leaf diseases from images.

Key benefits:

- High accuracy in disease classification
- Faster than manual inspection
- Scalable to large agricultural fields

---

## Model Architecture

The proposed architecture:

- Dual-Branch CNN:
  - Branch 1: Standard feature extraction
  - Branch 2: Attention mechanism for focusing on important regions
- Merged features from both branches
- Dense layers for final classification

---

## Results
  - The model achieved *99%* training accuracy and *97%* validation accuracy, demonstrating strong generalization and robustness.
  - ![image](https://github.com/user-attachments/assets/8effb36a-84d3-4956-967d-dacc003d1a69)
  
  - Confusion matrix shows that the model was able to genealize well and able to identify the diseased(Late bright or early bright) and the healthy.
  - ![image](https://github.com/user-attachments/assets/789f2914-4fe0-470e-a746-48504659372c)

  - **Accuracy:** Achieved over *98%* accuracy on test images.
  - ![image](https://github.com/user-attachments/assets/2fcb718b-6f37-449b-9fea-68977899bc81)


---

## Requirements

- Python 3.8+
- TensorFlow / Keras
- NumPy
- Matplotlib
- Jupyter Notebook


