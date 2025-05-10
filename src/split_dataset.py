import shutil
import random
from pathlib import Path

SRC_DIR = Path("data/raw")
DST_DIR = Path("data/split")
SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train, val, test

def split_data():
    random.seed(42)
    for class_dir in SRC_DIR.iterdir():
        if not class_dir.is_dir():
            continue
        images = list(class_dir.glob("*"))
        random.shuffle(images)

        n = len(images)
        n_train = int(n * SPLIT_RATIOS[0])
        n_val = int(n * SPLIT_RATIOS[1])
        
        split_sets = {
            "train": images[:n_train],
            "val": images[n_train:n_train+n_val],
            "test": images[n_train+n_val:]
        }

        for split, split_imgs in split_sets.items():
            out_dir = DST_DIR / split / class_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for img in split_imgs:
                shutil.copy(img, out_dir / img.name)

    print("✅ Dataset successfully split.")

if __name__ == "__main__":
    split_data()
