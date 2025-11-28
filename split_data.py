"""
split_data.py
Split a dataset into multiple client folders for FL simulation.

Usage examples:
# If your source has class subfolders (recommended):
python split_data.py --source /path/to/ham10000 --out ./data --clients 3 --by_class

# If your source is a flat folder with all images:
python split_data.py --source /path/to/all_images --out ./data --clients 3
"""
import argparse
import random
import shutil
from pathlib import Path

def split_by_class(source, out, clients=3):
    source = Path(source)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    classes = [d for d in source.iterdir() if d.is_dir()]
    if not classes:
        raise RuntimeError("Source must have class subfolders when --by_class is used.")
    for i in range(clients):
        (out / f"client_{i+1}").mkdir(parents=True, exist_ok=True)
    for cls in classes:
        images = list(cls.iterdir())
        random.shuffle(images)
        for idx, img in enumerate(images):
            target_client = (idx % clients) + 1
            target_dir = out / f"client_{target_client}" / cls.name
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(img, target_dir / img.name)

def split_flat(source, out, clients=3):
    source = Path(source)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    images = [p for p in source.iterdir() if p.is_file() and p.suffix.lower() in ['.jpg','.jpeg','.png']]
    random.shuffle(images)
    for i in range(clients):
        (out / f"client_{i+1}").mkdir(parents=True, exist_ok=True)
    for idx, img in enumerate(images):
        target_client = (idx % clients) + 1
        # place all into a default class folder
        target_dir = out / f"client_{target_client}" / "class_0"
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(img, target_dir / img.name)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="Source folder (class subfolders recommended)")
    p.add_argument("--out", default="./data", help="Output folder for clients")
    p.add_argument("--clients", type=int, default=3)
    p.add_argument("--by_class", action="store_true", help="If source contains class subfolders")
    args = p.parse_args()

    if args.by_class:
        split_by_class(args.source, args.out, args.clients)
    else:
        split_flat(args.source, args.out, args.clients)

    print("Split complete. Output folder:", args.out)