import pandas as pd
import shutil
from pathlib import Path

metadata = pd.read_csv("ham10000/HAM10000_metadata.csv")

src = Path("ham10000/all_images")
dest = Path("ham10000/sorted")
dest.mkdir(parents=True, exist_ok=True)

for _, row in metadata.iterrows():
    img = row["image_id"] + ".jpg"
    label = row["dx"]  # diagnosis label

    label_folder = dest / label
    label_folder.mkdir(exist_ok=True)

    src_img = src / img
    if src_img.exists():
        shutil.copy(src_img, label_folder / img)

print("Sorting complete. Check ham10000/sorted/")