# ===============================================
# ai_server.py  ✅ MULTI-MODEL STABLE VERSION
# -----------------------------------------------
# Flask AI Server for Mobile Connection
# CNN + LLaMA Medical Explanation API
# ===============================================

import os
import time
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from llama_cpp import Llama
import timm  # for Ovary model compatibility

# ------------------ CONFIG ------------------
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

# Disable ANSI colors for cleaner logs
os.environ["TERM"] = "none"
os.environ["CLICOLOR"] = "0"

# ------------------ INIT ------------------
app = Flask(__name__)

print("✅ Initializing AI Server...")
print(f"📌 Using device: {DEVICE}")

# ------------------ MODEL SELECTION ------------------
print("\n==============================")
print("🧠 Available Models:")
for i, name in enumerate(MODEL_PATHS.keys(), start=1):
    print(f"{i}. {name}")
print("==============================")

try:
    choice = int(input("👉 Enter the number of the model you want to load: "))
    if choice < 1 or choice > len(MODEL_PATHS):
        raise ValueError
    organ = list(MODEL_PATHS.keys())[choice - 1]
except Exception:
    print("⚠️ Invalid choice. Defaulting to Skin model.")
    organ = "Skin"

print(f"✅ Selected Model: {organ}")
MODEL_PATH = MODEL_PATHS[organ]
CLASSES = CLASS_LABELS[organ]

# ------------------ LOAD CNN MODEL ------------------
print("🧠 Loading CNN model...")

if organ == "Skin":
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(1280, len(CLASSES))

elif organ == "Ovary":
    model = timm.create_model("mobilenetv2_100", pretrained=False, num_classes=len(CLASSES))

elif organ == "Brain MRI":
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

elif organ == "Chest X-ray":
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(CLASSES))

else:
    raise ValueError("❌ Invalid organ selected!")

# Load weights
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
if missing or unexpected:
    print(f"⚠️ Non-strict load: {len(missing)} missing, {len(unexpected)} unexpected keys.")

model.to(DEVICE)
model.eval()
print(f"✅ {organ} model loaded successfully.")

# ------------------ LOAD LLaMA ------------------
print("🦙 Loading LLaMA model...")
try:
    llm = Llama(
        model_path=LLM_PATH,
        n_ctx=2048,
        n_threads=4,
        verbose=False
    )
    print("✅ LLaMA model loaded successfully.")
except Exception as e:
    print(f"❌ LLaMA failed to load: {e}")
    llm = None

# ------------------ IMAGE TRANSFORM ------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ------------------ ROUTES ------------------
@app.route('/')
def home():
    """Health check route."""
    return jsonify({
        "status": "running",
        "model": organ,
        "classes": CLASSES,
        "message": "🚀 AI Server online",
        "endpoint": "/predict"
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction route."""
    start_time = time.time()
    print("\n📥 Received request from client")

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded. Use key 'image'"}), 400

    file = request.files['image']
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image file: {e}"}), 400

    # CNN Prediction
    x = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        preds = model(x)
        probs = torch.softmax(preds, dim=1).cpu().numpy()[0]

    top_idx = int(np.argmax(probs))
    top_class = CLASSES[top_idx]

    print(f"🧩 Prediction: {top_class}")

    # LLaMA Explanation
    explanation = "LLaMA not available."
    if llm:
        try:
            prob_dict = {CLASSES[i]: round(float(probs[i]), 4) for i in range(len(CLASSES))}
            prompt = f"""
You are a kind medical assistant.
Below are probability results for a {organ} image classifier:

Probabilities:
{prob_dict}

Top predicted condition: {top_class}

Explain in simple, clear English what this condition means,
its possible symptoms, and safe self-care suggestions.
End by reminding to visit a doctor. Keep it 6–8 sentences long.
"""
            print("💬 Generating LLaMA explanation...")
            output = llm.create_completion(
                prompt=prompt,
                max_tokens=300,
                temperature=0.8,
                top_p=0.9
            )

            if "choices" in output and len(output["choices"]) > 0:
                explanation = output["choices"][0]["text"].strip()
            else:
                explanation = str(output)
            print("✅ Explanation done.")

        except Exception as e:
            explanation = f"LLM error: {e}"
            print("❌ LLaMA Error:", e)

    duration = round(time.time() - start_time, 2)
    print(f"⏱️ Done in {duration}s\n")

    return jsonify({
        "model": organ,
        "prediction": top_class,
        "probabilities": {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
        "explanation": explanation,
        "processing_time_sec": duration
    })


# ------------------ MAIN ------------------
if __name__ == "__main__":
    print("\n🌐 Starting server on port 5000 ...")
    print("📱 Use: http://<your_pc_ip>:5000/predict from Android.")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
