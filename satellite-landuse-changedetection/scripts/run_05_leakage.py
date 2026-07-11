"""
Spatial leakage experiment: random split vs spatial block split, same
architecture and hyperparameters. Run from the project root with:
    python -m scripts.run_05_leakage
"""
import yaml
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from src.data.datasets import EuroSATDataset
from src.data.transforms import get_train_transform, get_eval_transform
from src.data.splits import random_split, spatial_block_split
from src.models.resnet_classifier import build_resnet18, freeze_backbone
from src.training.engine import train_one_epoch, evaluate

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = cfg["data"]["eurosat_classes"]


def run_split_experiment(split_fn, name, epochs=3):
    train_ds_full = EuroSATDataset(cfg["paths"]["data_raw"], transform=get_train_transform(cfg["data"]["image_size"]), download=False)
    eval_ds_full = EuroSATDataset(cfg["paths"]["data_raw"], transform=get_eval_transform(cfg["data"]["image_size"]), download=False)
    splits = split_fn(train_ds_full, val_fraction=cfg["data"]["val_fraction"], test_fraction=cfg["data"]["test_fraction"], seed=cfg["seed"])

    train_loader = DataLoader(Subset(train_ds_full, splits["train"]), batch_size=cfg["data"]["batch_size"], shuffle=True)
    val_loader = DataLoader(Subset(eval_ds_full, splits["val"]), batch_size=cfg["data"]["batch_size"])

    model = build_resnet18(num_classes=len(classes), pretrained=True)
    model = freeze_backbone(model).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, device)
    result = evaluate(model, val_loader, criterion, device)
    print(f"{name}: val macro-F1 = {result['macro_f1']:.4f}")
    return result["macro_f1"]


def main():
    f1_random = run_split_experiment(random_split, "Random split")
    f1_block = run_split_experiment(spatial_block_split, "Spatial block split")

    print(f"\nLeakage gap: {f1_random - f1_block:.4f} (random split score minus block split score)")
    print("Interpretation: a positive gap here suggests random split inflates accuracy via near-duplicate/adjacent tiles leaking across train/val.")


if __name__ == "__main__":
    main()
