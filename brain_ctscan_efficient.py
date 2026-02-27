import os, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# --------------------
# Paths
# --------------------
TRAIN_DIR = "BrainCT_Split/train"
VAL_DIR   = "BrainCT_Split/test"
OUT_PATH  = "models/brain_ct_efficientnetb0.pth"

IMG_SIZE = 224
BATCH = 32
EPOCHS = 20
LR = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# Transforms
# --------------------
train_tfms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

val_tfms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
val_ds   = datasets.ImageFolder(VAL_DIR, transform=val_tfms)
class_names = train_ds.classes

train_loader = DataLoader(train_ds,batch_size=BATCH,shuffle=True)
val_loader   = DataLoader(val_ds,batch_size=BATCH,shuffle=False)

# --------------------
# EfficientNet B0
# --------------------
model = models.efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --------------------
# Train function
# --------------------
def run_epoch(loader, training=True):
    model.train(training)
    total_loss, correct, total = 0,0,0
    probs_all, labels_all = [],[]

    for x,y in tqdm(loader, desc="Train" if training else "Val"):
        x,y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits,y)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()*x.size(0)
        pred = logits.argmax(1)
        correct += (pred==y).sum().item()
        total += x.size(0)

        p = torch.softmax(logits, dim=1)[:,1].detach().cpu().numpy()
        probs_all.append(p)
        labels_all.append(y.detach().cpu().numpy())

    probs_all=np.concatenate(probs_all)
    labels_all=np.concatenate(labels_all)
    auc = roc_auc_score(labels_all, probs_all)

    return total_loss/total, correct/total, auc

best_auc = -1

for ep in range(1,EPOCHS+1):
    print(f"\nEpoch {ep}/{EPOCHS}")
    trL,trA,trU = run_epoch(train_loader,True)
    vlL,vlA,vlU = run_epoch(val_loader,False)

    print(f"Train Loss:{trL:.4f} Acc:{trA*100:.2f}% AUC:{trU:.4f}")
    print(f"Val   Loss:{vlL:.4f} Acc:{vlA*100:.2f}% AUC:{vlU:.4f}")

    if vlU > best_auc:
        best_auc = vlU
        torch.save({"model":model.state_dict(),"classes":class_names}, OUT_PATH)
        print(f"✅ Saved best → {OUT_PATH}")
