import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to dataset
train_dir = "brain-mri data/train"
test_dir = "brain-mri data/test"

# Data transforms
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# Custom loader that looks inside the "images" subfolder
def make_dataset(path, transform):
    dataset = datasets.ImageFolder(
        root=path,
        transform=transform,
        loader=lambda x: Image.open(x).convert("RGB")
    )
    # Fix path pattern if extra 'images/' folder exists
    dataset.samples = [(os.path.join(root, file), cls)
                       for cls, (root, _, files) in enumerate(os.walk(path))
                       for file in files if file.endswith(('.jpg', '.png', '.jpeg'))]
    return dataset

# Alternatively, if that causes path errors, just use:
# train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms["train"])
# test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms["test"])

train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms["train"])
test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms["test"])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f}")

# Save trained model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/brain_mri_resnet18.pth")
print("✅ Brain MRI model saved at models/brain_mri_resnet18.pth")
