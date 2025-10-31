import os, random, shutil

# EDIT: set this to the folder where you downloaded Normal chest X-rays
source_folder = r"C:\Users\Ojal\Downloads\normal"

train_folder = "data/train/Normal"
test_folder = "data/test/Normal"
train_ratio = 0.8

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

all_images = [f for f in os.listdir(source_folder)
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
random.shuffle(all_images)

split_idx = int(len(all_images) * train_ratio)
train_imgs = all_images[:split_idx]
test_imgs = all_images[split_idx:]

for img in train_imgs:
    shutil.copy(os.path.join(source_folder, img), os.path.join(train_folder, img))
for img in test_imgs:
    shutil.copy(os.path.join(source_folder, img), os.path.join(test_folder, img))

print(f"✅ {len(train_imgs)} → {train_folder}")
print(f"✅ {len(test_imgs)} → {test_folder}")
