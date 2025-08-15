# Breast Cancer Prediction using Deep Learning 🩺💻

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)

---

## Overview
This project predicts whether a breast tumor is **malignant** or **benign** using **deep learning** techniques.  
We experimented with **CNN**, **LSTM**, and **transfer learning** using pre-trained models: MobileNetV3, EfficientNetB1, VGG16, ResNet50V2.  

The goal is to create an accurate model for **early detection** and **diagnosis** of breast cancer.

---

## Dataset
- **Source:** [BreakHis Breast Cancer Histopathological Dataset](https://www.kaggle.com/datasets/tathagatbanerjee/breakhis-breast-cancer-histopathological/data)  
- **Samples:** 7,909 microscopic images of breast tumor tissue  
- **Features:** 224x224 resized images, binary labels (Malignant, Benign)  
- **Magnification Factors:** 40X, 100X, 200X, 400X  

### Sample Images
**Malignant Tissue:**  
![Malignant Sample](images/malignant_sample.jpg)  

**Benign Tissue:**  
![Benign Sample](images/benign_sample.jpg)  

**Optional: Prediction GIF**  
![Model Prediction Demo](images/prediction_demo.gif)  

### Data Cleaning & Augmentation
- Removed **250 exact duplicates** and **7 nearly duplicate images**  
- Standardized all images  
- **Augmentation techniques:** random brightness, flips, rotations  

**Goal:** Improve model **generalization** and **accuracy**

---

## Methodology

### Algorithms Used
- **CNN:** Extracts hierarchical features from images  
- **LSTM:** Sequential model for recurrence prediction  
- **Transfer Learning Models:** Pre-trained architectures for better performance  
  - MobileNetV3  
  - EfficientNetB1  
  - VGG16  
  - ResNet50V2  

### Model Training
- Dataset split into **training** and **validation** sets  
- Trained with **augmented images**  
- Evaluated using **ROC-AUC**, **Accuracy**, and **Loss**

---

## Results

| Model          | Accuracy | ROC-AUC | Loss   |
|----------------|----------|---------|--------|
| CNN            | 82.77%   | 0.8257  | 0.5518 |
| LSTM           | 70.84%   | 0.5000  | 0.6074 |
| MobileNetV3    | 72.67%   | 0.7424  | 0.5521 |
| EfficientNetB1 | 79.63%   | 0.8767  | 0.4426 |
| VGG16          | 71.02%   | 0.7592  | 0.5624 |
| ResNet50V2     | 78.42%   | 0.8230  | 0.4556 |

**Best Performer:** **EfficientNetB1** – Accuracy: 87%, ROC-AUC: 0.8767  

### Accuracy Graph
![Model Accuracy Graph](images/accuracy_graph.png)  

### Loss Graph
![Model Loss Graph](images/loss_graph.png)  

---

## Technologies Used
- **Python**  
- **TensorFlow / Keras**  
- **NumPy, Pandas**  
- **Matplotlib, Seaborn**  
- **Jupyter Notebook**

---

## Future Work
- Hyperparameter tuning for all models  
- Explore additional pre-trained architectures  
- Deploy model for **real-time breast cancer detection**  
- Multi-class classification (different subtypes of breast cancer)  

---

## How to Run
```bash
git clone https://github.com/your-username/breast-cancer-prediction.git
cd breast-cancer-prediction
jupyter notebook "Breast Cancer Prediction.ipynb"
