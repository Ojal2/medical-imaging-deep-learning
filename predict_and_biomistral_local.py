import os, json, torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------------------------------------------
# LABELS for chest X-ray disease classification
# ------------------------------------------------------
LABELS = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema",
    "Emphysema","Effusion","Fibrosis","Hernia",
    "Infiltration","Mass","Nodule","Pleural_Thickening",
    "Pneumothorax","Normal"
]

# ------------------------------------------------------
# DEVICE
# ------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------
# LOAD LOCAL DENSENET MODEL
# ------------------------------------------------------
def load_model(path="models/densenet121_multilabel_balanced.pth"):
    num_classes = len(LABELS)
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ------------------------------------------------------
# IMAGE TRANSFORM
# ------------------------------------------------------
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ------------------------------------------------------
# PREDICTION FUNCTION
# ------------------------------------------------------
def predict_image(model, img_path, threshold=0.5):
    img = Image.open(img_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    results = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    predicted = [LABELS[i] for i, p in enumerate(probs) if p >= threshold]
    return results, predicted

# ------------------------------------------------------
# LOCAL BIOMISTRAL MODEL (No API)
# ------------------------------------------------------
def load_biomistral_local(model_path="BioMistral/BioMistral-7B-AWQ"):
    print("🔹 Loading local BioMistral model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    print("✅ BioMistral model loaded locally.")
    return tokenizer, model

# ------------------------------------------------------
# TEXT GENERATION FUNCTION
# ------------------------------------------------------
def biomistral_local_summary(tokenizer, model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=150)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary.strip()

# ------------------------------------------------------
# PATIENT SUMMARY LOGIC
# ------------------------------------------------------
def generate_patient_summary(predictions, tokenizer, model, threshold=0.5):
    likely = {k: v for k, v in predictions.items() if v >= threshold and k != "Normal"}
    if not likely:
        prompt = (
            "The chest X-ray appears normal. Produce a short, patient-friendly summary "
            "explaining that no suspicious findings were detected and next steps if symptoms persist."
        )
    else:
        items = ", ".join([f"{k} ({predictions[k]:.2f})" for k in likely])
        prompt = (
            f"The AI model detected possible findings on a chest X-ray: {items}. "
            "Write a short, compassionate explanation for a patient describing what this might mean "
            "and suggested next steps."
        )
    return biomistral_local_summary(tokenizer, model, prompt)

# ------------------------------------------------------
# TESTING BLOCK (optional)
# ------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Initializing...")
    model = load_model()
    tokenizer, biomistral = load_biomistral_local()

    # Example usage:
    img_path = "test_images/sample_xray.jpg"
    results, predicted = predict_image(model, img_path)
    print("Predicted Labels:", predicted)

    summary = generate_patient_summary(results, tokenizer, biomistral)
    print("\n=== Patient Summary ===\n", summary)
