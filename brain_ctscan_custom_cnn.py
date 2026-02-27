import os, math, random, time
from pathlib import Path

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np

# ------------------------------
# Config
# ------------------------------
TRAIN_DIR = "BrainCT_Split/train"
VAL_DIR   = "BrainCT_Split/test"
IMG_SIZE  = 224
BATCH     = 32
EPOCHS    = 25
LR        = 3e-4
WD        = 1e-4
NUM_WORKERS = 0   # WINDOWS FIX
OUT_DIR   = "models"
CKPT_PATH = os.path.join(OUT_DIR, "brain_ct_custom_cnn_best.pth")
SEED      = 42
USE_AMP   = True

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------
# Dataset & Transforms (grayscale)
# ------------------------------
train_tfms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

val_tfms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
val_ds   = datasets.ImageFolder(VAL_DIR,   transform=val_tfms)

class_names = train_ds.classes
num_classes = len(class_names)
assert num_classes == 2

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# ------------------------------
# Custom CNN
# ------------------------------
class BrainCTCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=2, dilation=2),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x); x = self.conv4(x)
        x = x.view(x.size(0), -1)
        return self.head(x)

model = BrainCTCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and device.type == "cuda")

best_val_auc = -1.0

def run_epoch(loader, training=True):
    model.train(training)
    total_loss, total_correct, total = 0.0, 0, 0
    all_probs = []; all_labels = []

    pbar = tqdm(loader, desc="Train" if training else "Val", ncols=100)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.cuda.amp.autocast(enabled=USE_AMP and device.type == "cuda"):
            logits = model(imgs)
            loss = criterion(logits, labels)

        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()

        total_loss += loss.item()*imgs.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total += imgs.size(0)

        probs = F.softmax(logits, dim=1)[:,1].detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.detach().cpu().numpy())

        pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{100*total_correct/total:.2f}%")

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    auc = roc_auc_score(all_labels, all_probs)
    return total_loss/total, total_correct/total, auc


if __name__ == "__main__":  # WINDOWS FIX
    for epoch in range(1, EPOCHS+1):
        print(f"\nEpoch {epoch}/{EPOCHS} — LR: {scheduler.get_last_lr()[0]:.6f}")

        train_loss, train_acc, train_auc = run_epoch(train_loader, training=True)
        val_loss,   val_acc,   val_auc   = run_epoch(val_loader,   training=False)

        scheduler.step()

        print(f"Train  — loss:{train_loss:.4f} acc:{train_acc*100:.2f}% auc:{train_auc:.4f}")
        print(f"Val    — loss:{val_loss:.4f} acc:{val_acc*100:.2f}% auc:{val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({"model": model.state_dict(),
                        "classes": class_names,
                        "img_size": IMG_SIZE}, CKPT_PATH)
            print(f"✅ Saved best checkpoint to {CKPT_PATH} (AUC={val_auc:.4f})")
