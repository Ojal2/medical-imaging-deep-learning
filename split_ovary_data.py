import os
import random
import shutil

# Paths
source_root = "Dataset_Extracted"
dest_root = "OvarianCancerData"
train_ratio = 0.8

os.makedirs(dest_root, exist_ok=True)

for class_name in os.listdir(source_root):
    outer_path = os.path.join(source_root, class_name)
    if not os.path.isdir(outer_path):
        continue

    # Go one level deeper if nested (e.g. Clear_Cell/Clear_Cell)
    inner_folders = [os.path.join(outer_path, f) for f in os.listdir(outer_path) if os.path.isdir(os.path.join(outer_path, f))]
    if len(inner_folders) == 1:
        class_path = inner_folders[0]
    else:
        class_path = outer_path

    # Collect images
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(images)

    train_count = int(len(images) * train_ratio)
    train_images = images[:train_count]
    test_images = images[train_count:]

    # Create destination folders
    train_dir = os.path.join(dest_root, "train", class_name)
    test_dir = os.path.join(dest_root, "test", class_name)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Copy images
    for img in train_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, img))
    for img in test_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(test_dir, img))

    print(f"✅ {class_name}: {len(train_images)} train, {len(test_images)} test")

print("\n🎯 Dataset split completed successfully!")
print("📁 Final structure at:", os.path.abspath(dest_root))
