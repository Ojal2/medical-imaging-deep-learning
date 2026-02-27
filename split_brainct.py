import os
import shutil
import random

SOURCE_DIR = "Brain_Data_Organised"            # Put ALL images inside BrainCT/Normal and BrainCT/Stroke before running
DEST_DIR = "BrainCT_Split"
train_ratio = 0.8

classes = ["Normal", "Stroke"]

for split in ["train", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(DEST_DIR, split, cls), exist_ok=True)

for cls in classes:
    class_path = os.path.join(SOURCE_DIR, cls)
    images = os.listdir(class_path)

    random.shuffle(images)

    train_count = int(len(images) * train_ratio)

    train_files = images[:train_count]
    test_files  = images[train_count:]

    for f in train_files:
        shutil.copy(os.path.join(class_path, f), 
                    os.path.join(DEST_DIR, "train", cls, f))
    
    for f in test_files:
        shutil.copy(os.path.join(class_path, f), 
                    os.path.join(DEST_DIR, "test", cls, f))

print("✅ Done splitting BrainCT dataset into TRAIN & TEST")
print("Train/Test ratio =", train_ratio)
print("Train & test folders created inside ->", DEST_DIR)
