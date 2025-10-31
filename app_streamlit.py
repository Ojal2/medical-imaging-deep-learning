import streamlit as st
from predict_and_biomistral import load_model, predict_image, generate_patient_summary
from PIL import Image
import tempfile, os

st.set_page_config(page_title="AI Chest X-ray Diagnosis", page_icon="🩻", layout="wide")
st.title("🩺 Chest X-ray Multi-label Diagnosis + BioMistral Summary")
st.write("Upload an X-ray image and get multi-label probabilities plus a patient-friendly summary.")

model_path = "models/densenet121_multilabel_balanced.pth"
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    st.warning('Trained model not found at models/densenet121_multilabel_balanced.pth. Place your model file there after training.')
    model = None

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg","jpeg","png"])
if uploaded_file and model is not None:
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, uploaded_file.name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image(Image.open(tmp_path), caption="Uploaded X-ray", use_column_width=True)
    with st.spinner("Running model..."):
        preds, predicted = predict_image(model, tmp_path, threshold=0.25)
    st.subheader("Predicted probabilities")
    for k, v in preds.items():
        st.write(f"**{k}**: {v:.3f}")
    st.subheader("Patient-friendly summary (BioMistral)")
    summary = generate_patient_summary(preds, threshold=0.25)
    st.write(summary)
