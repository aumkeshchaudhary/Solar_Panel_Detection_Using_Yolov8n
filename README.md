# 🌞 Solar Panel Detection using YOLO

Welcome to the **Solar Panel Detection** project! This repository contains the implementation of a deep learning-based object detection model using **YOLO (You Only Look Once)** to identify solar panels in satellite imagery.

---

## 🗂 Dataset Overview

### 🔗 Dataset Links
- 📸 [Images (Native Resolution)](https://drive.google.com/drive/folders/13QfMQ-7OdWKw-LR8DmypKwSHtI0Hk2wh?usp=sharing)
- 🏷️ [Labels & README](https://drive.google.com/drive/folders/13QfMQ-7OdWKw-LR8DmypKwSHtI0Hk2wh?usp=sharing)
- 📁 [Label Description](https://figshare.com/articles/dataset/Solar_Panel_Object_)

### 📚 Dataset Format
- **Annotation Format:** COCO JSON
- **Resolution:** 416x416 pixels (31 cm native resolution) and HD resolution (15.5 cm per pixel)
- **Bounding Box Type:** Horizontal Bounding Boxes (HBB)

---

## 💻 Model Overview
This project focuses on training a deep learning model for accurate solar panel detection. The model is trained using advanced techniques to improve precision and recall while minimizing loss functions.

### ✅ Features
- **YOLO-based object detection** for solar panels in satellite images
- **MS-COCO dataset format** for annotations
- **High-resolution images** for accurate detection
- **Optimized loss functions** for better performance

---

## 📊 Training Results
The training process was monitored over 10 epochs, capturing key performance metrics at each stage. Below is a summary of how the model evolved throughout training.

### 🌟 Performance Metrics

| Epoch | Train Box Loss | Train Class Loss | Train DFL Loss | Precision (B) | Recall (B) | mAP50 (B) | mAP50-95 (B) | Val Box Loss | Val Class Loss | Val DFL Loss | Learning Rate |
|-------|---------------|------------------|----------------|---------------|------------|-----------|--------------|--------------|---------------|--------------|---------------|
| 1 | 1.5330 | 1.8970 | 1.2994 | 0.77613 | 0.65645 | 0.76159 | 0.46168 | 1.3809 | 1.5466 | 1.1447 | 0.00066087 |
| 2 | 1.3423 | 1.1809 | 1.1610 | 0.75898 | 0.76493 | 0.80262 | 0.49780 | 1.3030 | 1.4904 | 1.1501 | 0.0011961 |
| 3 | 1.3064 | 1.0409 | 1.1446 | 0.77738 | 0.78911 | 0.84442 | 0.55126 | 1.2991 | 1.0789 | 1.1123 | 0.0015994 |
| 4 | 1.2606 | 0.9294 | 1.1248 | 0.83062 | 0.82983 | 0.88585 | 0.60235 | 1.2200 | 0.96988 | 1.1008 | 0.001406 |
| 5 | 1.2035 | 0.84672 | 1.1000 | 0.86901 | 0.85682 | 0.91622 | 0.64468 | 1.1487 | 0.8344 | 1.0579 | 0.001406 |
| 6 | 1.1654 | 0.79697 | 1.0745 | 0.89593 | 0.84584 | 0.92682 | 0.65240 | 1.1258 | 0.81344 | 1.0481 | 0.001208 |
| 7 | 1.1401 | 0.73832 | 1.0538 | 0.89034 | 0.88060 | 0.93442 | 0.66317 | 1.1227 | 0.73454 | 1.0298 | 0.00101 |
| 8 | 1.0822 | 0.69635 | 1.0283 | 0.92599 | 0.89291 | 0.95554 | 0.69690 | 1.0449 | 0.66122 | 1.0051 | 0.000812 |
| 9 | 1.0381 | 0.64833 | 1.0120 | 0.93221 | 0.91211 | 0.96097 | 0.72211 | 1.0130 | 0.62484 | 0.99192 | 0.000614 |
| 10 | 0.99095 | 0.61340 | 0.99433 | 0.94267 | 0.91769 | 0.96839 | 0.73714 | 0.9753 | 0.59519 | 0.97217 | 0.000416 |

### 📈 Insights from Training

#### 1. Loss Function Improvements
- **Box Loss (Localization):** Decreased from **1.533** to **0.99095**, indicating better solar panel localization.
- **Classification Loss:** Improved significantly from **1.897** to **0.6134**, showing strong feature learning.
- **DFL Loss (Distribution Focal Loss):** Reduced from **1.2994** to **0.99433**, ensuring sharper boundary predictions.

#### 2. Detection Performance
- **Precision:** Increased from **77.61%** to **94.27%**, demonstrating fewer false positives.
- **Recall:** Improved from **65.65%** to **91.77%**, meaning more accurate detections.
- **mAP50:** Climbed from **0.76159** to **0.96839**, showing excellent detection confidence.
- **mAP50-95:** Increased from **0.46168** to **0.73714**, proving strong performance at varying thresholds.

#### 3. Learning Rate Adaptation
- Peaked at **0.0015994** (Epoch 3) and gradually decreased to **0.000416**.
- Balanced rapid learning in early stages and fine-tuned stability in later epochs.

#### 4. Validation vs Training Performance
- Minimal gap between validation and training losses confirms strong generalization and minimal overfitting.

---

## 🎯 Conclusion
This deep learning model has demonstrated significant improvements in detecting solar panels with high accuracy. Future enhancements may include:
- **Data Augmentation**: To further improve generalization.
- **Hyperparameter Optimization**: Fine-tuning learning rates and batch sizes.
- **Transfer Learning**: Experimenting with pre-trained models for enhanced performance.

---

### 🚀 Stay Updated
Want to contribute or stay updated? Feel free to open issues or submit PRs!

📌 **Repository Link**: [GitHub](#)  
📢 **Follow for Updates**: [Twitter](#)  
📩 **Contact**: your.email@example.com

