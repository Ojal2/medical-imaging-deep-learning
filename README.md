# 🩺 Chest X-ray Multi-label Diagnosis + BioMistral Summary

A Streamlit-based deep learning app for detecting multiple chest diseases using **DenseNet121** and generating patient-friendly summaries using **BioMistral** locally.

---

## 🚀 Features
- Multi-label disease detection (14 classes + Normal)
- Local inference (no Hugging Face API required)
- Patient-friendly report generation using BioMistral
- Streamlit front-end interface
- GPU acceleration (optional)

---

## 🧠 Dataset
Uses [Chest X-rays of 14 Common Diseases](https://www.kaggle.com/datasets/animeshshedge/chest-x-rays-of-14-common-disease)  
and normal class images merged from NIH Chest X-ray dataset.

---

## ⚙️ Setup
```bash
# Clone repo
git clone https://github.com/<your-username>/ChestXray-Project.git
cd ChestXray-Project

# Create virtual environment
python -m venv mlenv
.\mlenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
