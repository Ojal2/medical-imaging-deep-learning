import os, json
import torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from huggingface_hub import InferenceApi

LABELS = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema",
    "Emphysema","Effusion","Fibrosis","Hernia",
    "Infiltration","Mass","Nodule","Pleural_Thickening",
    "Pneumothorax","Normal"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path="models/densenet121_multilabel_balanced.pth"):
    num_classes = len(LABELS)
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

TRANSFORM = transforms.Compose([            transforms.Resize((224,224)),            transforms.ToTensor(),            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])        ])

def predict_image(model, img_path, threshold=0.5):
    img = Image.open(img_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    results = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    predicted = [LABELS[i] for i, p in enumerate(probs) if p >= threshold]
    return results, predicted

# Use Hugging Face Inference API for BioMistral summaries
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", None)
BIOMISTRAL_ID = "BioMistral/BioMistral-7B"

from huggingface_hub import InferenceApi

api = InferenceApi(repo_id="BioMistral/BioMistral-7B", token="YOUR_HF_TOKEN")

def biomistral_via_api(summary_prompt):
    try:
        # Request model output as raw text
        out = api(inputs=summary_prompt, params={"max_new_tokens":150}, raw_response=True)
        summary_text = out.text  # Extract the plain text
        return summary_text.strip()
    except Exception as e:
        print("Error in biomistral_via_api:", e)
        return "Error generating summary."


def generate_patient_summary(predictions, threshold=0.5):
    likely = {k: v for k, v in predictions.items() if v >= threshold and k != "Normal"}
    if not likely:
        prompt = "The chest X-ray appears normal. Produce a short, patient-friendly summary (2-3 sentences) explaining no suspicious findings and recommended next steps if symptomatic."
    else:
        items = ", ".join([f"{k} ({predictions[k]:.2f})" for k in likely])
        prompt = f"The AI model detected the following possible findings on a chest x-ray: {items}. Write a short, compassionate, clinician-style explanation (2-4 sentences) for a patient describing what this might mean and recommended next steps."
    summary = biomistral_via_api(prompt)
    return summary
print("✅ Script finished successfully.")
