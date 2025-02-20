# 🌞 Solar Panel Detection using YOLO

[![Hugging Face Space](https://img.shields.io/badge/🤖%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/Aumkeshchy2003/Solar_Panel_detection)

Welcome to the **Solar Panel Detection** project! This repository contains the implementation of a deep learning-based object detection model using **YOLO (You Only Look Once)** to identify solar panels in satellite imagery.

📌 **Dataset Format:** MS-COCO with Horizontal Bounding Boxes (HBB)  
📌 **Resolution:** 416x416 pixels (31 cm native resolution) and HD resolution (15.5 cm per pixel)  
📌 **Annotation Format:** COCO JSON  

---

## 📂 Dataset Overview

### 🔗 Dataset Links
- 📸 [Images (Native Resolution)](https://drive.google.com/drive/folders/13QfMQ-7OdWKw-LR8DmypKwSHtI0Hk2wh?usp=sharing)
- 🏷️ [Labels & README](https://drive.google.com/drive/folders/13QfMQ-7OdWKw-LR8DmypKwSHtI0Hk2wh?usp=sharing)
- 📚 [Label Description](https://figshare.com/articles/dataset/Solar_Panel_Object_Labels/22081091)

---

## 📊 Data Exploration & Understanding

### 1. Dataset Statistics

- **Total Solar Panel Instances:** **29,625**
- **Labels Per Image Distribution:**  
  - 81 images have **1** label  
  - 168 images have **2** labels  
  - 224 images have **3** labels  
  - Full distribution is available in the notebook.

*Histogram visualizations show the frequency of solar panel occurrences per image.*

### 2. Solar Panel Area Statistics

**Methodology:**  
- For native resolution, each pixel corresponds to **31 cm**.  
- For HD resolution, each pixel corresponds to **15.5 cm**.  
- The area (in m²) is computed as:  
  \[ \text{Area} = \text{width} \times \text{height} \times (\text{scale factor})^2 \]

- **Mean Area (Native Resolution):** **191.52 m²**  
- **Standard Deviation (Native Resolution):** **630.70 m²**
- **Mean Area (HD Resolution):** **47.879 m²**  
- **Standard Deviation (HD Resolution):** **157.675 m²**

*Observation: The dataset exhibits a wide range of solar panel sizes, as seen in the area histogram.*

---

## 👔 Intersection over Union (IoU) Calculation

**Methodology:**
- The IoU measures the overlap between two bounding boxes.
- Implemented using the **Shapely** library for geometric operations.

**IoU Computation Example:**
- Box 1: [0.5, 0.5, 0.2, 0.2] (YOLO format)
- Box 2: [0.5, 0.5, 0.3, 0.3]
- **IoU Result:** **0.44**

---

## 📊 Average Precision (AP) Calculation

### Implemented AP Calculation Methods:
- **Pascal VOC 11-Point Interpolation**
- **COCO 101-Point Interpolation**
- **Area Under Precision-Recall Curve (AP AUC)**

### Sample AP50 (IoU = 0.5) Results:
- **AP (VOC 11-Point):** **0.3797**
- **AP (COCO 101-Point):** **0.3625**
- **AP (PR AUC):** **0.3661**

---

## 💻 Model Overview
This project focuses on training a deep learning model for accurate solar panel detection. The model is trained using advanced techniques to improve precision and recall while minimizing loss functions.

### Model Training & Evaluation
- **Framework:** YOLO (Ultralytics)
- **Epochs:** 10
- **Loss Convergence:** Confirmed through gradual loss reduction

---

## 📊 Training Results
The training process was monitored over 10 epochs, capturing key performance metrics at each stage. Below is a summary of how the model evolved throughout training.

### 🌟 Performance Metrics

| Epoch | Train Box Loss | Train Class Loss | Train DFL Loss | Precision (B) | Recall (B) | mAP50 (B) | mAP50-95 (B) | Val Box Loss | Val Class Loss | Val DFL Loss | Learning Rate |
|-------|---------------|------------------|----------------|---------------|------------|-----------|--------------|--------------|---------------|--------------|---------------|
| 1 | 1.5330 | 1.8970 | 1.2994 | 0.77613 | 0.65645 | 0.76159 | 0.46168 | 1.3809 | 1.5466 | 1.1447 | 0.00066087 |
| 2 | 1.3423 | 1.1809 | 1.1610 | 0.75898 | 0.76493 | 0.80262 | 0.49780 | 1.3030 | 1.4904 | 1.1501 | 0.0011961 |
| 3 | 1.3064 | 1.0409 | 1.1446 | 0.77738 | 0.78911 | 0.84442 | 0.55126 | 1.2991 | 1.0789 | 1.1123 | 0.0015994 |
| 10 | 0.99095 | 0.61340 | 0.99433 | 0.94267 | 0.91769 | 0.96839 | 0.73714 | 0.9753 | 0.59519 | 0.97217 | 0.000416 |

### 📈 Insights from Training

#### 1. Loss Function Improvements
- **Box Loss:** Decreased from **1.533** to **0.99095**, indicating better solar panel localization.
- **Classification Loss:** Improved significantly from **1.897** to **0.6134**.
- **DFL Loss:** Reduced from **1.2994** to **0.99433**, ensuring sharper boundary predictions.

#### 2. Detection Performance
- **Precision:** Increased from **77.61%** to **94.27%**.
- **Recall:** Improved from **65.65%** to **91.77%**.
- **mAP50:** Climbed from **0.76159** to **0.96839**.
- **mAP50-95:** Increased from **0.46168** to **0.73714**.

---

## 🎯 Conclusion
This deep learning model has demonstrated significant improvements in detecting solar panels with high accuracy. Future enhancements may include:
- **Data Augmentation**
- **Hyperparameter Optimization**
- **Transfer Learning**

---

