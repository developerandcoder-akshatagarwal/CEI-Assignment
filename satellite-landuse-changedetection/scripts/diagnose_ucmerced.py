"""
Diagnostic only -- loads every UC Merced image one by one, printing
progress as it goes. If the program crashes with no Python traceback,
whatever filename printed last (before the crash) is the problem file.
Run from the project root with:
    python -u -m scripts.diagnose_ucmerced
"""
import sys
from pathlib import Path
from PIL import Image

root = Path("data/raw/ucmerced")
classes = sorted([d.name for d in root.iterdir() if d.is_dir()])

count = 0
for c in classes:
    for f in sorted((root / c).glob("*")):
        if f.suffix.lower() not in (".tif", ".tiff", ".jpg", ".png"):
            continue
        count += 1
        print(f"[{count}] Loading: {f}")
        sys.stdout.flush()  # force this line to print immediately, before any crash
        img = Image.open(f).convert("RGB")
        img.load()  # force full decode now, not lazy -- this is where a bad file would actually crash

print(f"\nDone. Successfully loaded all {count} images with no crash.")
