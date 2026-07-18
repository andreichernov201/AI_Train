import hashlib
import os
from pathlib import Path

def hash_file(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

ROOT_DIR = Path(__file__).resolve().parents[2]
train_dir = ROOT_DIR / "dataset" / "detection" / "images" / "train"
val_dir = ROOT_DIR / "dataset" / "detection" / "images" / "val"

train_hashes = {hash_file(train_dir / f): f for f in os.listdir(train_dir)}
val_hashes = {hash_file(val_dir / f): f for f in os.listdir(val_dir)}

dups = set(train_hashes.keys()) & set(val_hashes.keys())

print("Найдено дубликатов:", len(dups))
for h in dups:
    print(train_hashes[h], val_hashes[h])
