# Breast Cancer Prediction using Deep Learning

## Overview
This project focuses on predicting whether a breast tumor is **malignant** or **benign** using deep learning techniques.  
We experimented with several algorithms including **CNN**, **LSTM**, and pre-trained models via **transfer learning** (MobileNetV3, EfficientNetB1, VGG16, ResNet50V2).  
The goal is to develop an accurate model to assist in **early detection** and **diagnosis** of breast cancer.

---

## Dataset
- **Source:** [BreakHis Breast Cancer Histopathological Dataset](https://www.kaggle.com/datasets/tathagatbanerjee/breakhis-breast-cancer-histopathological/data)  
- **Samples:** 7,909 microscopic images of breast tumor tissue  
- **Features:** Images resized to 224x224 pixels, with binary labels (Malignant, Benign)  
- **Magnification Factors:** 40X, 100X, 200X, 400X  

### Data Cleaning
- Removed **250 exact duplicate images** and **7 nearly duplicate images** to prevent model bias.  
- Ensured all images were **resized and standardized** for model training.

### Data Augmentation
- Random brightness adjustment  
- Random flips  
- Random rotations  

**Goal:** Improve generalization and model accuracy

---

## Methodology

### Algorithms Used
- **CNN:** Hierarchical deep learning model for feature extraction from images  
- **LSTM:** Sequential model used to predict breast cancer recurrence  
- **Transfer Learning Models:** Leveraging pre-trained architectures for better performance  
  - MobileNetV3  
  - EfficientNetB1  
  - VGG16  
  - ResNet50V2  

### Model Training
- Images were split into **training** and **validation** sets  
- Models were trained with **augmented images** to improve robustness  
- Evaluated using **ROC-AUC**, **Accuracy**, and **Loss** metrics  

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

**Best Performer:** EfficientNetB1 with **87% accuracy** and **ROC-AUC of 0.8767**

---

## Technologies Used
- Python  
- TensorFlow / Keras  
- NumPy, Pandas  
- Matplotlib, Seaborn  
- Jupyter Notebook  

---

## Future Work
- Hyperparameter tuning for all models  
- Explore additional pre-trained architectures for improved accuracy  
- Deploy models for **real-time breast cancer detection** applications  
- Multi-class classification (e.g., different subtypes of breast cancer)  

---

## How to Run
```bash
git clone https://github.com/your-username/breast-cancer-prediction.git
cd breast-cancer-prediction
jupyter notebook "Breast Cancer Prediction.ipynb"
