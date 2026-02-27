import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report
import numpy as np
import os

# --- Configuration ---
num_classes = 4  # Clear_Cell, Endometri, Mucinous, Non_Cancerous
batch_size = 16
epochs = 15
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data paths ---
train_dir = "OvarianCancerData/train"
test_dir = "OvarianCancerData/test"

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Load training data ---
train_data = ImageFolder(train_dir, transform=transform)
test_data = ImageFolder(test_dir, transform=transform)

# --- Split training set into train + val (80/20) ---
val_size = int(0.2 * len(train_data))
train_size = len(train_data) - val_size
train_subset, val_subset = random_split(train_data, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

# --- Model ---
model = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
for param in model.features.parameters():
    param.requires_grad = False  # freeze feature extractor

# Replace final classifier layer
model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
model = model.to(device)

# --- Training setup ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# --- Training loop ---
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    val_loss, val_correct, val_total = 0.0, 0, 0
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {running_loss/len(train_loader):.4f} | "
          f"Val Loss: {val_loss/len(val_loader):.4f} | "
          f"Val Acc: {val_correct/val_total:.4f}")

# --- Save trained model ---
torch.save(model.state_dict(), "mobilenet_ovarian.pth")

# --- Evaluation ---
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=train_data.classes))
# --- Save trained model ---
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/mobilenet_ovarian.pth")
print("✅ Model saved as models/mobilenet_ovarian.pth")
