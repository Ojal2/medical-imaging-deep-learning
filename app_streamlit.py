import os
import tempfile
import streamlit as st
from PIL import Image

# Import backend functions
from predict_and_meditron_local import (
    load_chest_model,
    load_brain_model,
    load_ovarian_model,
    load_biomistral_local,
    load_biomistral_local,
    load_brain_ct_custom_model,
    load_brain_ct_efficient_model,
    predict_image,
    predict_brain_ct,
    CHEST_LABELS,
    BRAIN_LABELS,
    OVARIAN_LABELS,
    biomistral_summary
)

# ------------------------------------------------------
# ✅ STREAMLIT PAGE SETTINGS
# ------------------------------------------------------
st.set_page_config(
    page_title="AI Medical Imaging Assistant",
    page_icon="🧠",
    layout="wide"
)

st.title("🩺 AI Medical Imaging Assistant")
st.markdown("""
This system allows you to upload different types of medical scans:

✅ Chest X-ray  
✅ Brain MRI  
✅ Brain CT (Voting System)
✅ Ovarian Cancer Histopathology  
✅ Automatic model selection  
✅ Patient-friendly explanation using Meditron-7B
""")

# ------------------------------------------------------
# ✅ LOAD ALL MODELS (Cached)
# ------------------------------------------------------
@st.cache_resource
def load_all_models():
    try:
        chest_model = load_chest_model("models/densenet121_multilabel_balanced.pth")
    except:
        chest_model = None
        st.error("❌ Chest X-ray model missing.")

    try:
        brain_model = load_brain_model("models/brain_mri_resnet18.pth")
    except:
        brain_model = None
        st.error("❌ Brain MRI model missing.")

    try:
        ovarian_model = load_ovarian_model("models/mobilenet_ovarian.pth")
    except:
        ovarian_model = None
        st.error("❌ Ovarian Cancer MobileNet model missing.")

    # ✅ Load Brain CT Models
    try:
        brain_ct_custom, ct_labels_1 = load_brain_ct_custom_model("models/brain_ct_custom_cnn_best.pth")
    except:
        brain_ct_custom, ct_labels_1 = None, None
        st.error("❌ Brain CT Custom CNN model missing.")

    try:
        brain_ct_efficient, ct_labels_2 = load_brain_ct_efficient_model("models/brain_ct_efficientnetb0.pth")
    except:
        brain_ct_efficient, ct_labels_2 = None, None
        st.error("❌ Brain CT EfficientNet model missing.")


    # ✅ Load Meditron (local)
    tokenizer, meditron = load_biomistral_local("meditron-7b")

    return chest_model, brain_model, ovarian_model, brain_ct_custom, brain_ct_efficient, tokenizer, meditron



with st.spinner("Loading all models..."):
    chest_model, brain_model, ovarian_model, brain_ct_custom, brain_ct_efficient, tokenizer, meditron = load_all_models()



# ------------------------------------------------------
# ✅ SCAN TYPE DROPDOWN
# ------------------------------------------------------
scan_type = st.selectbox(
    "Select the type of scan you want to upload:",
    ["Chest X-ray", "Brain MRI", "Brain CT", "Ovarian Cancer (Histopathology)"]
)


if scan_type == "Chest X-ray":
    model = chest_model
    LABELS = CHEST_LABELS
    scan_key = "chest_xray"

elif scan_type == "Brain MRI":
    model = brain_model
    LABELS = BRAIN_LABELS
    scan_key = "brain_mri"

elif scan_type == "Brain CT":
    # Special handling for voting
    model = None # Placeholder, we use 2 models
    LABELS = [] # Not used directly
    scan_key = "brain_ct"

else:
    model = ovarian_model
    LABELS = OVARIAN_LABELS
    scan_key = "ovarian_cancer"


if scan_type != "Brain CT" and model is None:
    st.error(f"❌ The model for **{scan_type}** is not loaded.")
    st.stop()



# ------------------------------------------------------
# ✅ IMAGE UPLOADER
# ------------------------------------------------------
uploaded_file = st.file_uploader("Upload medical scan image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save temporarily
    tmp_dir = tempfile.mkdtemp()
    img_path = os.path.join(tmp_dir, uploaded_file.name)

    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Show image
    st.image(Image.open(img_path), caption="Uploaded Scan", width="stretch")

    # ------------------------------------------------------
    # ✅ RUN MODEL PREDICTION
    # ------------------------------------------------------
    # ------------------------------------------------------
    # ✅ RUN MODEL PREDICTION
    # ------------------------------------------------------
    with st.spinner("Running model prediction..."):
        
        if scan_type == "Brain CT":
            # ---------------------------
            # VOTING SYSTEM FOR BRAIN CT
            # ---------------------------
            if brain_ct_custom is None or brain_ct_efficient is None:
                st.error("❌ One or both Brain CT models are missing. Cannot perform voting.")
                st.stop()
            
            # Model 1: Custom CNN
            res1, pred1 = predict_brain_ct(
                brain_ct_custom, img_path, labels=None, is_grayscale_1ch=True
            )
            
            # Model 2: EfficientNet
            res2, pred2 = predict_brain_ct(
                brain_ct_efficient, img_path, labels=None, is_grayscale_1ch=False
            )
            
            # Display Results Side-by-Side
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("🤖 **Model 1: Custom CNN**")
                st.write(f"**Prediction:** `{pred1}`")
                st.json(res1)
                
            with col2:
                st.info("🚀 **Model 2: EfficientNet**")
                st.write(f"**Prediction:** `{pred2}`")
                st.json(res2)
                
            # Voting Logic
            st.subheader("⚖️ Voting Consensus")
            if pred1 == pred2:
                st.success(f"✅ **Agreement:** Both models predict **{pred1}**.")
                predicted = pred1
                preds = res2 # Use EfficientNet probs for Meditron summary as it's generally stronger
            else:
                st.warning(f"⚠️ **Disagreement:** Model 1 says **{pred1}**, Model 2 says **{pred2}**.")
                st.write("Please consult a radiologist for manual verification.")
                predicted = f"Ambiguous ({pred1} vs {pred2})"
                preds = {f"CNN_{k}":v for k,v in res1.items()}
                preds.update({f"EffNet_{k}":v for k,v in res2.items()})

        else:
            # Standard Single Model Logic
            preds, predicted = predict_image(model, img_path, LABELS)

            st.subheader("✅ Model Prediction")
            st.write(f"**Predicted Class:** `{predicted}`")
            st.json(preds)
    
    # ------------------------------------------------------
    # ✅ GENERATE MEDITRON SUMMARY
    # ------------------------------------------------------
    st.subheader("🧠 Patient-Friendly Explanation")

    if meditron is None:
        st.error("❌ Meditron model could not be loaded.")
    else:
        with st.spinner("Generating patient summary using Meditron..."):
            summary = biomistral_summary(tokenizer, meditron, preds, scan_key)
        st.success(summary)

