import zipfile
import os

dataset_dir = "Dataset"
extract_dir = "Dataset_Extracted"

os.makedirs(extract_dir, exist_ok=True)

# Loop through all zip files in the Dataset folder
for file in os.listdir(dataset_dir):
    if file.endswith(".zip"):
        zip_path = os.path.join(dataset_dir, file)
        folder_name = os.path.splitext(file)[0]  # remove .zip extension
        dest_path = os.path.join(extract_dir, folder_name)
        os.makedirs(dest_path, exist_ok=True)
        
        print(f"🔍 Extracting {file} ...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_path)
        print(f"✅ Done: {folder_name}")

print("\n🎯 All ovarian dataset folders extracted to:", os.path.abspath(extract_dir))
