"""
Diagnostic only -- tests whether loading .tif files crashes specifically
when torch/torchvision are also imported (a known Windows DLL-conflict
pattern between different libraries' bundled image codecs).
Run from the project root with:
    python -u -m scripts.diagnose_torch_tiff_conflict
"""
import sys
print("Importing torch and torchvision first...")
sys.stdout.flush()
import torch
import torchvision
print("torch/torchvision imported OK. Now trying to decode UC Merced TIFFs...")
sys.stdout.flush()

from pathlib import Path
from PIL import Image

root = Path("data/raw/ucmerced")
classes = sorted([d.name for d in root.iterdir() if d.is_dir()])

count = 0
for c in classes:
    for f in sorted((root / c).glob("*")):
        if f.suffix.lower() not in (".tif", ".tiff"):
            continue
        count += 1
        print(f"[{count}] Loading: {f}")
        sys.stdout.flush()
        img = Image.open(f).convert("RGB")
        img.load()

print(f"\nDone. Loaded {count} TIFFs successfully even with torch/torchvision imported.")
