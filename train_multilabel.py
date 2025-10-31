import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from PIL import Image
import pandas as pd

# ------------------------------
# CUSTOM DATASET
# ------------------------------
class ChestXrayMultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_paths = self.df['image_path'].values
        self.labels = self.df.iloc[:, 1:].values.astype(np.float32)
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")   # <-- force RGB
        label = self.labels[idx]

        if self.transform:
            image = self.transform(np.array(image))  # convert to array for ToPILImage if needed

        return image, torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.image_paths)


# ------------------------------
# MAIN TRAINING FUNCTION
# ------------------------------
def main():
    # CONFIG
    train_csv = "data/train_multilabel.csv"
    val_csv = "data/test_multilabel.csv"
    batch_size = 16
    epochs = 10
    lr = 1e-4
    img_size = 224
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DATA TRANSFORMS
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # DATASETS
    train_dataset = ChestXrayMultiLabelDataset(train_csv, transform=train_transform)
    val_dataset = ChestXrayMultiLabelDataset(val_csv, transform=val_transform)
    num_classes = train_dataset.labels.shape[1]

    # BALANCED SAMPLER
    label_sums = np.sum(train_dataset.labels, axis=0)
    class_weights = 1.0 / (label_sums + 1e-6)
    sample_weights = np.array([np.mean(train_dataset.labels[i] * class_weights) for i in range(len(train_dataset))])
    sample_weights = torch.from_numpy(sample_weights).float()
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # MODEL
    model = models.densenet121(weights="IMAGENET1K_V1")
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # TRAINING LOOP
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: train loss = {total_loss/len(train_loader):.4f}")

    # VALIDATION
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = torch.sigmoid(model(imgs))
            preds.append(out.cpu())
            trues.append(labels.cpu())
    preds = torch.cat(preds)
    trues = torch.cat(trues)
    aucs = [roc_auc_score(trues[:, i], preds[:, i]) for i in range(num_classes)]
    print("AUCs:", aucs)

    # SAVE MODEL
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/densenet121_multilabel_balanced.pth")
    print("✅ Model saved to models/densenet121_multilabel_balanced.pth")


# ------------------------------
# WINDOWS SAFE ENTRY POINT
# ------------------------------
if __name__ == "__main__":
    main()
