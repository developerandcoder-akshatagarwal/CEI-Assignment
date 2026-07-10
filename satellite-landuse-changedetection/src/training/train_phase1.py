"""
Phase 1: freeze the ResNet18 backbone, train only the new classifier
head for a few epochs. Run train_baseline.py's data-loading pattern —
duplicated here intentionally so each script is standalone and runnable
independently (useful if a run crashes and you need to resume just this
phase).

Usage:
    python -m src.training.train_phase1
"""
import yaml
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.datasets import EuroSATDataset
from src.data.transforms import get_train_transform, get_eval_transform
from src.data.splits import spatial_block_split
from src.models.resnet_classifier import build_resnet18, freeze_backbone
from src.training.engine import train_one_epoch, evaluate, save_checkpoint


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_train = EuroSATDataset(cfg["paths"]["data_raw"], transform=get_train_transform(cfg["data"]["image_size"]))
    full_eval = EuroSATDataset(cfg["paths"]["data_raw"], transform=get_eval_transform(cfg["data"]["image_size"]), download=False)

    splits = spatial_block_split(
        full_train,
        val_fraction=cfg["data"]["val_fraction"],
        test_fraction=cfg["data"]["test_fraction"],
        seed=cfg["seed"],
    )
    train_ds = Subset(full_train, splits["train"])
    val_ds = Subset(full_eval, splits["val"])

    train_loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"])

    model = build_resnet18(num_classes=len(cfg["data"]["eurosat_classes"]), pretrained=True)
    model = freeze_backbone(model).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["phase1"]["lr"],
    )

    best_f1 = 0.0
    for epoch in tqdm(range(cfg["training"]["phase1"]["epochs"]), desc="phase1 epochs"):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}")
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            save_checkpoint(model, f"{cfg['paths']['checkpoints']}/resnet18_phase1.pt")

    print(f"Best Phase 1 val macro-F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
