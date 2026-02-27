import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# -----------------------------
# ⚙️ Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

test_dir = "brain-mri data/test"

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# -----------------------------
# 🧩 Load Trained Model
# -----------------------------
model_path = "models/brain_mri_resnet18.pth"


model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, len(test_dataset.classes))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

print(f"✅ Model loaded successfully from {model_path}")

# -----------------------------
# 🧠 Evaluation
# -----------------------------
criterion = nn.CrossEntropyLoss()

val_loss = 0.0
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

val_loss /= len(test_loader)
val_acc = 100 * correct / total

print(f"\n📊 Validation Loss: {val_loss:.4f}")
print(f"🎯 Validation Accuracy: {val_acc:.2f}%")

# -----------------------------
# 📉 Confusion Matrix
# -----------------------------
cm = confusion_matrix(all_labels, all_preds)
classes = test_dataset.classes

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("🧠 Confusion Matrix - Brain MRI Classification")
plt.tight_layout()
plt.show()

# -----------------------------
# 📋 Classification Report
# -----------------------------
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))
