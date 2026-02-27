# 🩺 AI Medical Imaging Assistant

A comprehensive deep learning system for multi-organ medical image analysis with **Streamlit** interface and **Meditron-7B** powered clinical summaries. Detects diseases across chest X-rays, brain MRI, brain CT scans, and ovarian cancer histopathology with high accuracy.

---

## 🚀 Features

### 📊 Multi-Modal Disease Detection
- **Chest X-ray Analysis**: 14 diseases + Normal (DenseNet121)
- **Brain MRI Classification**: Tumor types - Glioma, Meningioma, Pituitary, No Tumor (ResNet18)
- **Brain CT Stroke Detection**: Normal vs Stroke with ensemble voting (Custom CNN + EfficientNetB0)
- **Ovarian Cancer Classification**: Clear Cell, Endometrial, Mucinous, Non-Cancerous (MobileNet)

### 🤖 Clinical Intelligence
- Local inference (no external API required)
- Patient-friendly medical report generation using **Meditron-7B**
- Multi-model ensemble for robust predictions (Brain CT)
- Detailed confidence scores and probability distributions

### 💻 User Interface & Deployment
- Interactive **Streamlit** dashboard for easy medical image uploads
- Web-based interface with real-time predictions
- GPU acceleration support (CUDA compatible)
- Responsive design for multiple medical imaging modalities
- AI Server for programmatic access

---

## 🧠 Datasets

| Modality | Classes | Source |
|----------|---------|--------|
| **Chest X-ray** | 14 diseases + Normal | [Kaggle Dataset](https://www.kaggle.com/datasets/animeshshedge/chest-x-rays-of-14-common-disease) + NIH |
| **Brain MRI** | 4 tumor types | Internal dataset |
| **Brain CT** | 2 (Normal/Stroke) | Internal CT dataset |
| **Ovarian Cancer** | 4 histology types | Pathology dataset |

---

## 📦 Models

| Modality | Architecture | Weights | Accuracy |
|----------|--------------|---------|----------|
| Chest X-ray | DenseNet121 | `densenet121_multilabel_balanced.pth` | Multi-label optimized |
| Brain MRI | ResNet18 | `brain_mri_resnet18.pth` | Tumor classification |
| Brain CT (Custom) | Custom CNN | `brain_ct_custom_cnn_best.pth` | Ensemble model |
| Brain CT (Efficient) | EfficientNetB0 | `brain_ct_efficientnetb0.pth` | Ensemble model |
| Ovarian Cancer | MobileNet | `mobilenet_ovarian.pth` | Histology classification |

---

## ⚙️ Setup

```bash
# Clone repository
git clone https://github.com/<your-username>/ChestXray-Project.git
cd ChestXray-Project

# Create virtual environment
python -m venv mlenv
.\mlenv\Scripts\activate  # Windows
# or
source mlenv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download Meditron-7B model (required for summaries)
# The model will be automatically downloaded on first use (7GB)
```

---

## 🎯 Usage

### Streamlit Dashboard (Recommended)
```bash
streamlit run app_streamlit.py
```
- Upload medical images
- Select imaging modality (automatic detection available)
- View predictions with confidence scores
- Generate clinical summaries using Meditron-7B

### Python API
```python
from predict_and_meditron_local import (
    load_chest_model,
    load_brain_model,
    load_brain_ct_custom_model,
    load_ovarian_model,
    predict_image,
    biomistral_summary
)

# Load models
chest_model = load_chest_model("models/densenet121_multilabel_balanced.pth")

# Make predictions
predictions = predict_image(image_path, chest_model)

# Generate summary
summary = biomistral_summary(predictions)
```

### AI Server
```bash
python ai_server.py
```
REST API for programmatic access to predictions.

---

## 📁 Project Structure

```
ChestXray-Project/
├── app_streamlit.py              # Main Streamlit application
├── ai_server.py                  # REST API server
├── predict_and_meditron_local.py # Inference + Meditron integration
├── utils.py                      # Utility functions
│
├── Training Scripts:
├── train_chest_model.py          # Chest X-ray model training
├── train_brain_model.py          # Brain MRI model training
├── train_ovarian_model.py        # Ovarian cancer model training
├── brain_ctscan_custom_cnn.py    # Custom CNN for Brain CT
├── brain_ctscan_efficient.py     # EfficientNet for Brain CT
│
├── Inference Scripts:
├── brain_infer_ctscan.py         # Brain CT inference
├── evaluate_brain_mri_model.py   # Brain MRI evaluation
│
├── Data Processing:
├── split_brainct.py
├── split_normal_data.py
├── split_ovary_data.py
├── generate_csv_multilabel.py
├── dataset_multilabel.py
├── unzip_data.py
│
├── models/                       # Trained model weights
├── meditron-7b/                  # Meditron-7B model cache
├── data/                         # Chest X-ray dataset
├── brain-mri\ data/              # Brain MRI dataset
├── BrainCT_Split/                # Brain CT dataset
├── OvarianCancerData/            # Ovarian cancer dataset
├── Brain_Data_Organised/         # Organized brain data
├── Dataset_Extracted/            # Extracted datasets
└── requirements.txt
```

---

## 🏥 Supported Medical Conditions

### Chest X-ray (14 diseases)
Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax, + Normal

### Brain MRI
- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

### Brain CT
- Normal
- Stroke

### Ovarian Cancer Histology
- Clear Cell
- Endometrial
- Mucinous
- Non-Cancerous

---

## 💾 System Requirements

- **RAM**: 8GB minimum (16GB recommended for Meditron-7B)
- **Storage**: 20GB for models + datasets
- **GPU**: NVIDIA CUDA-capable GPU (optional, uses CPU fallback)
- **Python**: 3.8+

---

## 📊 Model Performance

- Multi-label chest diseases detected with high precision
- Brain tumor classification for clinical decision support
- Brain CT stroke detection with ensemble voting
- Ovarian cancer histology for pathology labs

---

## 🔬 Technologies Used

- **PyTorch**: Deep learning framework
- **Transformers**: Meditron-7B language model
- **Streamlit**: Web interface
- **TorchVision**: Computer vision utilities
- **scikit-learn**: ML utilities
- **Pillow**: Image processing

---

## 📝 License & Attribution

This project uses medical datasets from Kaggle and NIH sources. Meditron-7B is provided by [Asclepius](https://github.com/asclepius-research/meditron).

---

## 🚀 Future Enhancements

- [ ] Real-time video analysis pipeline
- [ ] Mobile app deployment
- [ ] Additional modalities (Ultrasound, MRI sequences)
- [ ] Integration with hospital PACS systems
- [ ] Explainability with GradCAM visualizations
- [ ] Database logging for patient records

---

## 📧 Contact & Support

For issues, feature requests, or contributions, please open an issue on GitHub or contact the development team.
