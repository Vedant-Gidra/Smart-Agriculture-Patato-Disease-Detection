# Smart Agriculture: CNN-Based Potato Disease Detection System

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

This project proposes a **deep learning model** using a **Dual-Branch Convolutional Neural Network (CNN)** with an **attention mechanism**, deployed through a **Flask-based web server** for real-time disease detection.

### Key Benefits

- High accuracy in disease classification  
- Real-time image upload and prediction through Flask backend  
- Simple web interface for farmers and researchers  
- Faster and more scalable than manual inspection  

---

## How It Works

The system operates through a **Flask-based backend** that handles image uploads, preprocessing, and prediction. It follows a two-step pipeline to classify potato leaf diseases efficiently.

1. **Image Upload:**  
   The user uploads a leaf image through the web interface. The image is received by the Flask backend via the `/preprocess` endpoint.

2. **Preprocessing and Temporary Storage:**  
   The uploaded image is resized to **(224×224)** pixels, normalized to a **[0, 1]** range, and converted into a NumPy array for model compatibility.  
   It is then **saved locally** as `"temp_image.npy"` for later use during prediction.

3. **Model Prediction:**  
   The preprocessed image file (`temp_image.npy`) is loaded by the `/predict` endpoint, which passes it to the trained CNN model for classification.

4. **Response Generation:**  
   The backend returns the **predicted disease class** (Healthy, Early Blight, or Late Blight) along with the **confidence score** as a JSON response.

---

### API Endpoints

| Endpoint | Method | Description |
|-----------|---------|-------------|
| `/` | GET | Displays the web interface for potato disease detection. |
| `/preprocess` | POST | Accepts an uploaded image, preprocesses it, and saves it locally as `temp_image.npy`. |
| `/predict` | POST | Loads `temp_image.npy`, runs model inference, and returns the predicted disease with confidence. |


---

## Model Architecture

The proposed architecture includes:

- **Dual-Branch CNN:**
  - Branch 1: Standard convolutional feature extraction
  - Branch 2: Attention-based feature enhancement  
- **Feature Fusion Layer:** Merges both branches for better context representation  
- **Dense Layers:** Perform final classification into three categories

---

## Results
- The model achieved *99%* training accuracy and *97%* validation accuracy, demonstrating strong generalization and robustness.
<img width="984" height="328" alt="image" src="https://github.com/user-attachments/assets/635f239e-6d13-41d0-bb65-522731721ed2" />

- Confusion matrix shows that the model was able to genealize well and able to identify the diseased(Late bright or early bright) and the healthy.
<img width="746" height="663" alt="image" src="https://github.com/user-attachments/assets/c01cc099-ae69-4b5d-ab64-1fd8896243a7" />

- **Accuracy:** Achieved over *98%* accuracy on test images.
<img width="573" height="214" alt="image" src="https://github.com/user-attachments/assets/d9dc4d2d-a66e-4242-b3c6-a5eccefd440e" />


---

## Interface
<img width="1057" height="750" alt="image" src="https://github.com/user-attachments/assets/8e6e9cf6-af87-413f-ab87-ce8f048fb3ca" />
<img width="1027" height="442" alt="image" src="https://github.com/user-attachments/assets/8371ab3b-1e9b-4f13-994b-46e69af1dd7d" />

---

## Requirements

- Python 3.8+
- Flask
- TensorFlow
- NumPy
- Pandas
