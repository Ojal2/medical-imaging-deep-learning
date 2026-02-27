import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from llama_cpp import Llama
import timm  # ✅ Needed for ovarian model compatibility

# ------------------ CONFIG ------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATHS = {
    "Brain MRI": r"C:\Users\Lokesh Pagare\Desktop\ml assignment\CP\brain_mri_resnet18.pth",
    "Ovary": r"C:\Users\Lokesh Pagare\Desktop\ml assignment\CP\mobilenet_ovarian.pth",
    "Skin": r"C:\Users\Lokesh Pagare\Desktop\ml assignment\CP\mobilenet_skin_model.pth",
    "Chest X-ray": r"C:\Users\Lokesh Pagare\Desktop\ml assignment\CP\densenet121_multilabel_balanced.pth"
}

CLASS_LABELS = {
    "Brain MRI": ["no tumor", "pituitary", "meningioma", "glioma"],
    "Ovary": ["benign", "malignant"],
    "Skin": ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'],
    "Chest X-ray": [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
        'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
        'Pleural_Thickening', 'Hernia'
    ]
}

LLM_PATH = r"C:\Users\Lokesh Pagare\Documents\floatchat\llama-2-7b-chat.Q4_K_M.gguf"
IMG_SIZE = 224
USE_LLM = True
# -----------------------------------------------------

st.title("🩺 Multi-Disease AI Diagnosis + LLaMA Explanation")
st.caption("Educational AI prototype — not for medical use.")

# Step 1️⃣: Organ selection
organ = st.radio(
    "Which organ/body part do you want to analyze?",
    ["Brain MRI", "Ovary", "Skin", "Chest X-ray"],
    horizontal=True
)
st.write(f"🧬 Selected: **{organ}**")

# Step 2️⃣: Upload image
file = st.file_uploader(f"Upload {organ} image", type=["jpg", "jpeg", "png"])

# ---------------- Model Loader ----------------
@st.cache_resource
def load_model(organ_name):
    """Load the specific CNN model based on organ, using correct architecture."""
    if organ_name == "Skin":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(1280, len(CLASS_LABELS["Skin"]))

        state_dict = torch.load(MODEL_PATHS["Skin"], map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)

    elif organ_name == "Ovary":
        # 🩷 Fix: use timm MobileNetV2 (matches your ovarian model)
        model = timm.create_model("mobilenetv2_100", pretrained=False, num_classes=len(CLASS_LABELS["Ovary"]))
        state_dict = torch.load(MODEL_PATHS["Ovary"], map_location=DEVICE)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            st.warning(f"⚠️ Non-strict load for Ovary model. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    elif organ_name == "Brain MRI":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_LABELS["Brain MRI"]))
        state_dict = torch.load(MODEL_PATHS["Brain MRI"], map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)

    elif organ_name == "Chest X-ray":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, len(CLASS_LABELS["Chest X-ray"]))
        state_dict = torch.load(MODEL_PATHS["Chest X-ray"], map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)

    else:
        raise ValueError("Unknown organ selected!")

    model.to(DEVICE)
    model.eval()
    return model

# ---------------- LLaMA Loader ----------------
@st.cache_resource
def load_llm():
    """Loads local LLaMA model."""
    try:
        return Llama(
            model_path=LLM_PATH,
            n_ctx=2048,
            n_threads=6,
            verbose=False
        )
    except Exception as e:
        st.error(f"LLM failed to load: {e}")
        return None

# ---------------- Image Transform ----------------
def preprocess_image(img):
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return tf(img).unsqueeze(0)

# ---------------- Main App Logic ----------------
if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption=f"Uploaded {organ} Image", use_column_width=True)

    model = load_model(organ)
    x = preprocess_image(image).to(DEVICE)

    with torch.no_grad():
        preds = model(x)
        probs = torch.softmax(preds, dim=1).cpu().numpy()[0]

    classes = CLASS_LABELS[organ]
    st.subheader(f"📊 Prediction Probabilities for {organ}")
    for cls, pr in zip(classes, probs):
        st.write(f"**{cls}** : {pr:.4f}")

    fig, ax = plt.subplots()
    ax.bar(classes, probs)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    st.pyplot(fig)

    top_class = classes[np.argmax(probs)]
    st.success(f"Top prediction: {top_class}")

    # ---------------- LLaMA Explanation ----------------
    if USE_LLM:
        st.subheader("🤖 AI Medical Explanation & First Aid")

        llm = load_llm()
        if llm:
            prompt = f"""
You are a kind and knowledgeable medical assistant.
Explain the condition **{top_class}** (found in a {organ} image)
in simple English for a general audience.
Describe briefly what it is, possible symptoms, and safe self-care tips.
End by reminding to visit a doctor.
Keep the explanation short and easy to understand (about 6–8 sentences).
"""

            with st.spinner("🧠 Generating AI explanation..."):
                try:
                    output = llm.create_completion(
                        prompt=prompt,
                        max_tokens=300,
                        temperature=0.9,
                        top_p=0.95,
                        repeat_penalty=1.1
                    )

                    text = ""
                    if "choices" in output and len(output["choices"]) > 0:
                        text = output["choices"][0].get("text", "").strip()

                    text = text.replace("Assistant:", "").replace("User:", "").strip()

                    if not text or len(text) < 30:
                        text = "⚠️ The model returned an incomplete explanation. Try re-uploading or adjusting temperature."

                    st.markdown("### 🩺 AI Explanation")
                    st.write(text)

                except Exception as e:
                    st.error(f"❌ LLaMA generation failed: {e}")
        else:
            st.warning("LLaMA model not loaded.")

    st.info("⚠️ Educational use only — not for real medical diagnosis.")
