import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------------------------------------------
# Label sets for different models
# ------------------------------------------------------

CHEST_LABELS = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema",
    "Emphysema","Effusion","Fibrosis","Hernia",
    "Infiltration","Mass","Nodule","Pleural_Thickening",
    "Pneumothorax","Normal"
]

BRAIN_LABELS = [
    "glioma", "meningioma", "notumor", "pituitary"
]

OVARIAN_LABELS = [
    "Clear_Cell", "Endometri", "Mucinous", "Non_Cancerous"
]

# Placeholder, will be overwritten by model loading if available
BRAIN_CT_LABELS = ["Tumor", "Normal"]  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------
# ✅ LOAD DENSENET MODEL (Chest X-ray)
# ------------------------------------------------------
def load_chest_model(path="models/densenet121_multilabel_balanced.pth"):
    print("🔹 Loading Chest X-ray model...")

    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(CHEST_LABELS))

    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()

    print("✅ Chest model loaded")
    return model


# ------------------------------------------------------
# ✅ LOAD RESNET MODEL (Brain MRI)
# ------------------------------------------------------
def load_brain_model(path="models/brain_mri_resnet18.pth"):
    print("🔹 Loading Brain MRI model...")

    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, len(BRAIN_LABELS))

    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()

    print("✅ Brain MRI model loaded")
    return model


# ------------------------------------------------------
# ✅ LOAD MOBILENET MODEL (Ovarian Cancer)
# ------------------------------------------------------
def load_ovarian_model(path):
    print("🔍 Loading Ovarian Cancer MobileNetV3-LARGE model...")

    model = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)

    state = torch.load(path, map_location=device)
    model.load_state_dict(state)

    model.to(device)
    model.eval()

    print("✅ Ovarian Cancer MobileNetV3-LARGE model loaded successfully!")
    return model


# ------------------------------------------------------
# ✅ BRAIN CT (CUSTOM CNN) ARCHITECTURE
# ------------------------------------------------------
class BrainCTCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128,256,3,padding=2,dilation=2), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256,256,3,padding=2,dilation=2), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(256,128), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(128,2)
        )
    def forward(self,x):
        x=self.conv1(x); x=self.conv2(x); x=self.conv3(x); x=self.conv4(x)
        x=x.view(x.size(0),-1)
        return self.head(x)

# ------------------------------------------------------
# ✅ LOAD BRAIN CT (CUSTOM CNN)
# ------------------------------------------------------
def load_brain_ct_custom_model(path):
    print(f"🔹 Loading Brain CT Custom CNN from {path}...")
    try:
        ckpt = torch.load(path, map_location=device)
        # Checkpoint contains {'model': state_dict, 'classes': [...]}
        classes = ckpt.get("classes", ["Tumor", "Normal"])
        state_dict = ckpt.get("model", ckpt) # Fallback if just state_dict
        
        model = BrainCTCNN(num_classes=len(classes))
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        print(f"✅ Brain CT Custom CNN loaded. Classes: {classes}")
        return model, classes
    except Exception as e:
        print(f"❌ Error loading Brain CT Custom CNN: {e}")
        return None, None

# ------------------------------------------------------
# ✅ LOAD BRAIN CT (EFFICIENTNET)
# ------------------------------------------------------
def load_brain_ct_efficient_model(path):
    print(f"🔹 Loading Brain CT EfficientNet from {path}...")
    try:
        ckpt = torch.load(path, map_location=device)
        classes = ckpt.get("classes", ["Tumor", "Normal"])
        state_dict = ckpt.get("model", ckpt)

        # Rebuild architecture: efficientnet_b0 with modified classifier
        model = models.efficientnet_b0(weights=None)
        # Original code: model.classifier[1] = nn.Linear(..., 2)
        # We need to match the saved state dict shapes
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, len(classes))
        
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        print(f"✅ Brain CT EfficientNet loaded. Classes: {classes}")
        return model, classes
    except Exception as e:
        print(f"❌ Error loading Brain CT EfficientNet: {e}")
        return None, None



# ------------------------------------------------------
# ✅ IMAGE TRANSFORM (shared)
# ------------------------------------------------------
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


# ------------------------------------------------------
# ✅ PREDICT FUNCTION (Generic)
# ------------------------------------------------------
def predict_image(model, img_path, labels, threshold=0.25):
    img = Image.open(img_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)

    # Multi-class (softmax)
    if logits.shape[1] > 1:
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    # Multi-label (sigmoid)
    else:
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    results = {labels[i]: float(probs[i]) for i in range(len(labels))}
    predicted = labels[int(probs.argmax())]

    return results, predicted
    

# ------------------------------------------------------
# ✅ SPECIAL PREDICT FOR BRAIN CT (Different Transform)
# ------------------------------------------------------
def predict_brain_ct(model, img_path, labels, is_grayscale_1ch=False):
    """
    Custom prediction specifically for Brain CT models which might use
    1-channel grayscale (Custom CNN) or 3-channel (EfficientNet).
    """
    img = Image.open(img_path).convert("L" if is_grayscale_1ch else "RGB")
    
    if is_grayscale_1ch:
        # Transform for Custom CNN: Grayscale(1), Resize(224), Normalize([0.5])
        tfm = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    else:
        # Transform for EfficientNet: Grayscale(3) -> RGB effectively, Resize, Norm
        # The training script used Grayscale(num_output_channels=3)
        tfm = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])

    tensor = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # Map generic labels if not provided
    if not labels:
        labels = [f"Class_{i}" for i in range(len(probs))]

    # specific fix if labels len != probs len
    if len(labels) != len(probs):
         labels = [f"Class_{i}" for i in range(len(probs))]

    results = {labels[i]: float(probs[i]) for i in range(len(probs))}
    predicted = labels[probs.argmax()]

    return results, predicted



# ------------------------------------------------------
# ✅ FIXED — LOAD MEDITRON MODEL (NO ACCELERATE REQUIRED)
# ------------------------------------------------------
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, os

def load_biomistral_local(model_path="meditron-7b"):
    """
    Loads EPFL Meditron-7B from a local folder using the *safetensors* shards
    in 4-bit NF4 with bitsandbytes. Fits on ~6–7 GB VRAM GPUs (e.g., RTX 4050).
    """
    try:
        print(f"🧠 Loading Meditron (safetensors, 4-bit) from: {model_path}")

        # Free any stale CUDA cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Force safetensors (your .bin shards are incomplete)
        os.environ["SAFETENSORS_FAST_GPU"] = "1"

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            use_fast=True
        )

        # 4-bit quantized load (bitsandbytes)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            use_safetensors=True,           # <- force safetensors
            load_in_4bit=True,              # <- 4-bit quantization
            device_map="auto",              # <- place layers automatically
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        # Some Meditron repos miss a pad token; this avoids generate() warnings
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("✅ Meditron loaded (4-bit, safetensors).")
        return tokenizer, model

    except Exception as e:
        print(f"❌ Error loading Meditron (4-bit): {e}")
        print("↪ Falling back to CPU (VERY slow), safetensors full-precision...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                use_fast=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                use_safetensors=True,
                device_map={"": "cpu"},
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            print("✅ Meditron loaded on CPU.")
            return tokenizer, model
        except Exception as e2:
            print(f"❌ CPU fallback also failed: {e2}")
            return None, None




# ------------------------------------------------------
# ✅ GENERATE PATIENT SUMMARY
# ------------------------------------------------------
def biomistral_summary(tokenizer, model, predictions, scan_type):
    if model is None or tokenizer is None:
        return "❌ Meditron model not loaded."

    findings = [f"{k}: {v:.2f}" for k, v in predictions.items()]
    findings_text = ", ".join(findings)

    prompt = (
        f"You are Meditron, a medical expert AI.\n"
        f"Explain these {scan_type} results in simple, patient-friendly language.\n\n"
        f"Findings: {findings_text}\n\n"
        f"Write a short summary under 120 words. Do NOT repeat the instructions or headings. "
        f"Start directly with the explanation.\n"
        f"Answer:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=160,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # ✅ Extract ONLY the answer after "Answer:"
    if "Answer:" in decoded:
        decoded = decoded.split("Answer:", 1)[1].strip()

    # ✅ Remove any repeated prompt parts
    for keyword in ["You are Meditron", "Findings", "Constraints", "Summary"]:
        if keyword in decoded[:80]:  # only look at the start
            decoded = decoded.split(keyword)[-1].strip()

    return decoded.strip()
