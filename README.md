# 🐱🐶 Cat vs Dog Classifier using Transfer Learning

## 📌 Project Overview
This project implements a **Deep Learning-based image classifier** to distinguish between cats and dogs using **transfer learning**.  
A pre-trained Convolutional Neural Network (CNN) (such as VGG16, ResNet50, or MobileNet) is fine-tuned on the dataset to achieve high accuracy with reduced training time and computational cost.

---

## 🚀 Features
- Utilizes **transfer learning** for efficient model training.  
- Classifies input images as **Cat** or **Dog**.  
- Achieves faster convergence with pre-trained feature extractors.  
- Can be extended to other binary/multi-class image classification tasks.  

---

## 📂 Dataset
- Source: [Kaggle Dogs vs Cats Dataset](https://www.kaggle.com/datasets/jangedoo/utkface-new)
- Training images: Cats and Dogs (balanced dataset).  
- Preprocessing includes resizing, normalization, and data augmentation.  

---

## 🛠️ Tech Stack
- **Python**
- **TensorFlow / Keras**
- **NumPy, Pandas, Matplotlib**

---

## 🧠 Model Workflow
1. Load pre-trained CNN (e.g., VGG16/ResNet50) without top layers.  
2. Add custom dense layers for binary classification.  
3. Apply dropout and regularization to prevent overfitting.  
4. Train on Cats vs Dogs dataset using **transfer learning**.  
5. Evaluate performance using accuracy and confusion matrix.  

---

## 📊 Results
- Achieved high accuracy (>90%) on validation dataset.  
- Generalizes well on unseen test images.  

---

## ▶️ How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/imamankishore/Cat-Dog-Classifier-Using-Transfer-Learning.git
   cd Cat-Dog-Classifier-Using-Transfer-Learning
