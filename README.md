# Crop Disease Classification

Deep learning project for multi-class classification of crop diseases using leaf images. Built as a capstone project under the NgaoLabs Data Science Training Program.

---

## Executive Summary

| Attribute | Specification |
|-----------|-------------|
| **Problem Type** | Multi-class Image Classification (10 classes) |
| **Input Data** | Leaf images (RGB) |
| **Models** | Custom CNN, MobileNetV2, EfficientNetB0 |
| **Primary Metric** | Weighted F1-score (class imbalance sensitivity) |
| **Deployment** | Streamlit web application |

---

## Results & Insights

- ✅ **Transfer learning outperformed custom CNN** due to pre-trained ImageNet feature extraction
- ⚠️ **Generalization challenges** observed from controlled dataset conditions (lab imagery)
- 📊 **Model interpretability** via confidence visualization and top-5 prediction charts
- 🚀 **MobileNetV2 selected** as optimal accuracy-efficiency trade-off (88-93% validation accuracy)

---

## Problem Context

Crop diseases reduce agricultural productivity and are often diagnosed late due to limited access to expertise. In Kenya, **94% of farmers** are affected by pests and diseases, with extension officer-to-farmer ratios exceeding **1:1,500**.

Manual inspection is:
- ❌ Slow and inconsistent
- ❌ Expertise-dependent
- ❌ Reactive (2-3 week detection delays)

**Solution:** Automated image-based classification system for faster, scalable diagnosis.

---

## Approach

### Data Preparation

| Step | Operation | Parameters |
|------|-----------|------------|
| Resizing | Bicubic interpolation | 224×224 pixels |
| Normalization | Pixel scaling | [0, 1] range |
| Splitting | Stratified | 70/15/15 (train/val/test) |
| Augmentation | Rotation, shift, flip, zoom, brightness | Training set only |

**Augmentation Config:**
- Rotation: ±20°
- Width/Height shift: ±20%
- Horizontal flip: 50% probability
- Zoom: [0.8, 1.2]
- Brightness: [0.8, 1.2]

### Exploratory Data Analysis

- Class distribution assessment (imbalance ratio: 1.4:1)
- Sample visualization per class
- Image quality verification (dimensions, corruption detection)
- Duplicate detection (MD5 hash-based)

### Feature Engineering

- **Implicit feature extraction** via CNN architectures
- No manual feature engineering required
- Pre-trained ImageNet weights for transfer learning models

---

## Modeling

### Models Evaluated

| Model | Type | Parameters | Expected Accuracy |
|-------|------|------------|-------------------|
| Custom CNN | Baseline | ~1.2M | 75-85% |
| MobileNetV2 | Transfer Learning | 3.5M (~0.5M trainable) | 88-93% |
| EfficientNetB0 | Transfer Learning | 5.3M (~0.6M trainable) | 90-95% |

### Selection Rationale

- **Custom CNN:** Establishes baseline performance
- **Transfer Learning:** Leverages ImageNet features for improved generalization and training efficiency
- **MobileNetV2 Selected:** Optimal balance of accuracy, speed, and model size for deployment

### Training Strategy

| Parameter | Setting |
|-----------|---------|
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Initial Learning Rate | 0.0001 (frozen), 0.00001 (fine-tuning) |
| Epochs | 30 (10 frozen + 20 fine-tuning) |
| Batch Size | 32 |
| Early Stopping | Patience=5, monitor=val_loss |
| LR Reduction | ReduceLROnPlateau (factor=0.2, patience=3) |
| Class Weights | Computed from training distribution |

---

## Evaluation

### Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| Accuracy | >90% | 91.2% |
| Precision (macro) | >85% | 89.4% |
| Recall (macro) | >85% | 88.7% |
| F1-score (macro) | >85% | 89.0% |
| **F1-score (weighted)** | **>87%** | **90.8%** |
| Cohen's Kappa | >0.80 | 0.84 |

### Validation

- Hold-out test set (15% of data)
- Stratified sampling to preserve class distribution
- Confusion matrix analysis per class

---

## Dataset

| Attribute | Specification |
|-----------|-------------|
| **Source** | [Kaggle – Five Crop Diseases Dataset](https://www.kaggle.com/datasets/shubham2703/five-crop-diseases-dataset) |
| **Original Size** | 13,324 images |
| **Original Classes** | 17 (5 crops) |
| **Filtered Classes** | **10** (3 crops: Corn, Potato, Wheat) |
| **Format** | RGB JPEG |
| **License** | CC0: Public Domain |

### Class Distribution

| Crop | Classes | Images |
|------|---------|--------|
| Corn | Common Rust, Gray Leaf Spot, Northern Leaf Blight, Healthy | 4,252 |
| Potato | Early Blight, Late Blight, Healthy | 2,152 |
| Wheat | Brown Rust, Yellow Rust, Healthy | 2,942 |

*Note: Rice and Sugarcane excluded per project scope (Phase 2 expansion)*

---

## Deployment

**Platform:** Streamlit

### Features

| Feature | Description |
|---------|-------------|
| Image Upload | Drag-and-drop or file selector (JPG, PNG, JPEG) |
| Prediction Display | Predicted class + confidence percentage |
| Confidence Visualization | Top-5 predictions with bar chart |
| Advisory System | Rule-based treatment and prevention recommendations |
| History Log | Session-based scan tracking |
| Theme Toggle | Dark/Light mode |

### Live Demo

```
streamlit run app.py
```

Access at `http://localhost:8501`

---

## Project Structure

```
crop_disease_prediction_CNN/
├── 📁 assets/                    # Project assets
│   ├── augmentation_examples.png
│   ├── background.jpg
│   ├── eda_class_distribution.png
│   ├── eda_image_properties.png
│   ├── eda_sample_images.png
│   ├── image.png
│   └── logo.png
│
├── 📁 Crop Diseases Dataset/     # Raw dataset
│   └── Info.txt
│
├── 📁 documentation/             # Project documentation
│   ├── CAPSTONE NEURALNEST.docx
│   ├── NeuralNest Presentation.pptx
│   ├── NeuralNest Report.docx
│   └── NeuralNest Report.pdf
│
├── 📁 logs/                      # Training logs
│   ├── mobilenetv2_fine/
│   └── mobilenetv2_phase1/
│
├── 📁 models/                    # All model files
│   ├── 📁 deployment/            # Production-ready models
│   │   ├── NeuralNest_MobileNetV2_savedmodel/
│   │   ├── advisory_rules.json
│   │   ├── class_names.json
│   │   ├── label_encoder.pkl
│   │   ├── mobilenetv2_best.h5
│   │   ├── model_info.json
│   │   ├── NeuralNest_MobileNetV2.h5
│   │   ├── NeuralNest_MobileNetV2.keras
│   │   └── NeuralNest_MobileNetV2.tflite
│   │
│   ├── mobilenetv2_best.h5       # Training checkpoint (best)
│   ├── mobilenetv2_best.keras    # Keras format (best)
│   ├── mobilenetv2_final.h5      # Final training checkpoint
│   ├── mobilenetv2_final.keras   # Final Keras format
│   ├── mobilenetv2_phase1_best.h5     # Phase 1 best checkpoint
│   └── mobilenetv2_phase1_best.keras # Phase 1 Keras format
│
├── 📁 notebooks/                 # Jupyter notebooks
│   ├── data_preparation.ipynb
│   └── .ipynb_checkpoints/
│
├── 📁 src/                       # Source code
│   ├── 📁 .ipynb_checkpoints/
│   ├── confusion_matrix_mobilenetv2.png
│   ├── model_training.py        # Training script
│   └── mobilenetv2_report.csv   # Training report
│ 
├── app.py                   # Streamlit application
├── 📄 README.md                 # Project documentation
├── 📄 requirements.txt           # Python dependencies
├── 📄 requirements-streamlit.txt # Streamlit-specific requirements
├── 📄 setup.bat                # Windows setup script
├── 📄 runtime.txt                # Runtime configuration
└── 📄 kaggle.json                # Kaggle API credentials
```

---

## Reproducibility

### Setup

```bash
# 1. Clone the repository
git clone git@github.com:karanja-dave/crop_disease_prediction_CNN.git

# 2. Navigate to project folder
cd crop_disease_prediction_CNN

# 3. Create virtual environment
python -m venv .venv

# 4. Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 5. Install dependencies
pip install -r requirements.txt

# 6. Run application
streamlit run app.py
```

### Requirements

```
streamlit>=1.28.0
tensorflow==2.21.0
pillow>=9.0.0
numpy>=1.21.0
pandas>=1.3.0
plotly>=5.0.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
```

---
#   Deployment Requirements 
streamlit
tensorflow==2.21.0
tf_keras==2.21.0
pillow
numpy
pandas
plotly
opencv-python
protobuf

## Team

**Ngao Labs Cohort II - Capstone Project**

| Name | Role |
|------|------|
| Dave Karanja | ML Engineer |
| Jedidiah | Data Scientist |
| Pauline | UI/UX Developer |

**Duration:** March 2026 – April 2026

---

## License

- **Code:** MIT License
- **Dataset:** CC0 1.0 Universal (Public Domain)

---

## Acknowledgments

- [Five Crop Diseases Dataset](https://www.kaggle.com/datasets/shubham2703/five-crop-diseases-dataset) by Shubham Sharma
- Ngao Labs for capstone program support
- Google Colab for GPU resources
```
