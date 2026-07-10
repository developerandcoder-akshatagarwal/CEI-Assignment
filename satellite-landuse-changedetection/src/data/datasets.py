"""
Dataset wrappers for EuroSAT (auto-downloaded via torchvision) and
UC Merced Land Use (manual download required — see README).
"""
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image


class EuroSATDataset(Dataset):
    """Thin wrapper around torchvision's EuroSAT so we control the
    transform and can attach a filename (needed later for spatial-block
    hashing in splits.py).
    """

    def __init__(self, root, transform=None, download=True):
        self.base = datasets.EuroSAT(root=root, download=download)
        self.transform = transform
        self.classes = self.base.classes

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        # torchvision keeps the underlying file path in .samples
        filepath = self.base.samples[idx][0]
        if self.transform:
            img = self.transform(img)
        return img, label, filepath


class UCMercedDataset(Dataset):
    """UC Merced Land Use dataset. Expects the standard folder-per-class
    layout after manual download:

        data/raw/ucmerced/<classname>/<image>.tif

    Download from the original UC Merced Vision Group page and extract
    into data/raw/ucmerced/ before running any UC Merced notebook.
    """

    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.classes = sorted(
            [d.name for d in self.root.iterdir() if d.is_dir()]
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        for c in self.classes:
            class_dir = self.root / c
            for f in class_dir.glob("*"):
                if f.suffix.lower() in (".tif", ".tiff", ".jpg", ".png"):
                    self.samples.append((str(f), self.class_to_idx[c]))
        if not self.samples:
            raise RuntimeError(
                f"No images found under {self.root}. Did you download and "
                f"extract UC Merced there? See README.md."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, path
