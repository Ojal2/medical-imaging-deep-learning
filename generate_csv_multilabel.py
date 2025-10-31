import os, pandas as pd

LABELS = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema",
    "Emphysema","Effusion","Fibrosis","Hernia",
    "Infiltration","Mass","Nodule","Pleural_Thickening",
    "Pneumothorax","Normal"
]

def generate_csv(data_dir, output_csv):
    rows = []
    for label in LABELS:
        folder = os.path.join(data_dir, label)
        if not os.path.exists(folder):
            continue
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            rows.append({"Image": img_path, "Label": label})
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"No images found under {data_dir}. Check folder structure.")
        return
    grouped = df.groupby("Image")["Label"].apply(list).reset_index()
    for lbl in LABELS:
        grouped[lbl] = grouped["Label"].apply(lambda x: 1 if lbl in x else 0)
    grouped.drop(columns=["Label"], inplace=True)
    grouped.to_csv(output_csv, index=False)
    print(f"✅ CSV saved: {output_csv}")

if __name__ == "__main__":
    generate_csv("data/train", "data/train_multilabel.csv")
    generate_csv("data/test",  "data/test_multilabel.csv")
