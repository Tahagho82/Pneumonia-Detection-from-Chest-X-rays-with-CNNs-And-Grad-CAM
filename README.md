# 🩺 Pneumonia Detection from Chest X-rays with Transfer Learning & Grad-CAM

## 📌 Overview
This project applies **deep learning (CNN + Transfer Learning)** to classify **pneumonia vs normal** from **chest X-ray images**.  
We used **ResNet50 pretrained on ImageNet** as the backbone, fine-tuned a classification head, and evaluated the model with standard medical imaging metrics.  
To improve interpretability, we applied **Grad-CAM** to highlight lung regions influencing the model’s predictions.  

---

## 🚀 Key Features
- Dataset: [Chest X-ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
- Framework: **TensorFlow / Keras** (Colab Notebook implementation)  
- Preprocessing: Grayscale-to-RGB conversion, normalization, augmentation with Keras layers  
- Model: Transfer learning with **ResNet50**, custom head (`GlobalAveragePooling + Dropout + Dense`)  
- Training: Adam optimizer, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint  
- Evaluation metrics: Accuracy, ROC-AUC, PR-AUC, Confusion Matrix, Classification Report  
- Explainability: **Grad-CAM heatmaps** to visualize model attention  

---

## 📊 Results
**Test set performance (n=624):**
- **Accuracy:** 91.8%  
- **ROC-AUC:** 0.961  
- **PR-AUC:** 0.966  

**Confusion Matrix:**
```
[[204   30]   → True Negatives / False Positives
 [ 21  369]]  → False Negatives / True Positives
```

**Classification Report:**
| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Normal (0) | 0.907 | 0.872 | 0.889 | 234 |
| Pneumonia (1) | 0.925 | 0.946 | 0.935 | 390 |
| **Overall** | **0.918** | **0.918** | **0.918** | 624 |

📌 Interpretation:
- **High recall for pneumonia (0.946):** Model rarely misses sick patients.  
- **Good specificity (0.872):** Relatively few false alarms.  
- **AUC scores >0.96:** Strong discriminative power, robust across thresholds.  

---

## 🔎 Explainability
Grad-CAM visualizations confirm the model focuses on **lung regions** when detecting pneumonia.  

*(Add sample Grad-CAM heatmap images here)*  

---

## ⚙️ How to Run
1. Open the Colab notebook:   

2. Requirements:  
   - Google Colab environment  
   - Kaggle API key (`kaggle.json`) to download dataset  

3. Steps inside the notebook:  
   - Load dataset  
   - Train the model  
   - Evaluate metrics  
   - Generate Grad-CAM visualizations  

---

## 📌 Future Work
- Fine-tune deeper layers of ResNet50 with a lower learning rate  
- Test on larger external datasets (e.g., NIH ChestX-ray14)  
- Perform calibration for more reliable probability outputs  
- Explore Focal Loss or class weights for better handling of imbalanced data  
