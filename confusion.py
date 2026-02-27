import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# 1. Common Setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("results", exist_ok=True)

def plot_and_save_cm(y_true, y_pred, classes, model_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"{model_name} — Confusion Matrix")
    plt.tight_layout()
    save_path = f"results/{model_name}_confusion_matrix.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved confusion matrix: {save_path}")

# -------------------------------
# 2. Evaluation Function
# -------------------------------
def evaluate_model(model, test_loader, classes, model_name):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    plot_and_save_cm(y_true, y_pred, classes, model_name)


# -------------------------------
# 3. Test Transforms
# -------------------------------
standard_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

gray_tfms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# -------------------------------
# 4. CHEST X-RAY (DenseNet121)
# -------------------------------
try:
    chest_dir = "brain-mri data/test"  # ⚠️ replace with your actual chest test directory
    chest_data = datasets.ImageFolder(chest_dir, transform=standard_tfms)
    chest_loader = torch.utils.data.DataLoader(chest_data, batch_size=32, shuffle=False)

    chest_model = models.densenet121(weights=None)
    chest_model.classifier = nn.Linear(chest_model.classifier.in_features, len(chest_data.classes))
    chest_model.load_state_dict(torch.load("models/densenet121_multilabel_balanced.pth", map_location=device))
    chest_model.to(device)
    evaluate_model(chest_model, chest_loader, chest_data.classes, "ChestXray_DenseNet121")
except Exception as e:
    print(f"❌ Chest model error: {e}")


# -------------------------------
# 5. BRAIN MRI (ResNet18)
# -------------------------------
try:
    brain_dir = "brain-mri data/test"
    brain_data = datasets.ImageFolder(brain_dir, transform=standard_tfms)
    brain_loader = torch.utils.data.DataLoader(brain_data, batch_size=32, shuffle=False)

    brain_model = models.resnet18(weights="IMAGENET1K_V1")
    brain_model.fc = nn.Linear(brain_model.fc.in_features, len(brain_data.classes))
    brain_model.load_state_dict(torch.load("models/brain_mri_resnet18.pth", map_location=device))
    brain_model.to(device)
    evaluate_model(brain_model, brain_loader, brain_data.classes, "BrainMRI_ResNet18")
except Exception as e:
    print(f"❌ Brain MRI model error: {e}")


# -------------------------------
# 6. OVARIAN CANCER (MobileNetV3)
# -------------------------------
try:
    ovarian_dir = "OvarianCancerData/test"
    ovarian_data = datasets.ImageFolder(ovarian_dir, transform=standard_tfms)
    ovarian_loader = torch.utils.data.DataLoader(ovarian_data, batch_size=32, shuffle=False)

    ovarian_model = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
    ovarian_model.classifier[3] = nn.Linear(ovarian_model.classifier[3].in_features, len(ovarian_data.classes))
    ovarian_model.load_state_dict(torch.load("models/mobilenet_ovarian.pth", map_location=device))
    ovarian_model.to(device)
    evaluate_model(ovarian_model, ovarian_loader, ovarian_data.classes, "OvarianCancer_MobileNetV3")
except Exception as e:
    print(f"❌ Ovarian model error: {e}")


# -------------------------------
# 7. BRAIN CT (Custom CNN)
# -------------------------------
try:
    from brain_ctscan_custom_cnn import BrainCTCNN  # Ensure this file is in same folder

    brainct_dir = "BrainCT_Split/test"
    brainct_data = datasets.ImageFolder(brainct_dir, transform=gray_tfms)
    brainct_loader = torch.utils.data.DataLoader(brainct_data, batch_size=32, shuffle=False)

    ckpt = torch.load("models/brain_ct_custom_cnn_best.pth", map_location=device)
    brainct_model = BrainCTCNN(num_classes=len(brainct_data.classes))
    brainct_model.load_state_dict(ckpt["model"])
    brainct_model.to(device)
    evaluate_model(brainct_model, brainct_loader, brainct_data.classes, "BrainCT_CustomCNN")
except Exception as e:
    print(f"❌ Brain CT model error: {e}")


print("\n🎉 All available confusion matrices generated and saved in /results/")
