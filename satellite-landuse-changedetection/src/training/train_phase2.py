"""
Phase 2: load the Phase 1 checkpoint, unfreeze layer3+layer4, train
with a 10x-lower LR on the backbone than the head (discriminative LR).
Watch val_loss each epoch — if it spikes upward, that's catastrophic
forgetting; revert to the Phase 1 checkpoint and lower lr_backbone
further in config.yaml.

Usage:
    python -m src.training.train_phase2
"""
import yaml
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.datasets import EuroSATDataset
from src.data.transforms import get_train_transform, get_eval_transform
from src.data.splits import spatial_block_split
from src.models.resnet_classifier import build_resnet18, unfreeze_layers, get_param_groups
from src.training.engine import train_one_epoch, evaluate, save_checkpoint, load_checkpoint


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
    model = load_checkpoint(model, f"{cfg['paths']['checkpoints']}/resnet18_phase1.pt", device)
    unfrozen = cfg["training"]["phase2"]["unfreeze_layers"]
    model = unfreeze_layers(model, layer_names=unfrozen).to(device)

    criterion = nn.CrossEntropyLoss()
    param_groups = get_param_groups(
        model,
        lr_head=cfg["training"]["phase2"]["lr_head"],
        lr_backbone=cfg["training"]["phase2"]["lr_backbone"],
        unfrozen_layers=unfrozen,
    )
    optimizer = optim.Adam(param_groups)

    best_f1 = 0.0
    prev_val_loss = None
    for epoch in tqdm(range(cfg["training"]["phase2"]["epochs"]), desc="phase2 epochs"):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}")

        if prev_val_loss is not None and val_metrics["loss"] > prev_val_loss * 1.5:
            print("WARNING: val_loss spiked >50% vs previous epoch — possible catastrophic "
                  "forgetting. Consider stopping and lowering lr_backbone in config.yaml.")
        prev_val_loss = val_metrics["loss"]

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            save_checkpoint(model, f"{cfg['paths']['checkpoints']}/resnet18_eurosat_finetuned.pt")

    print(f"Best Phase 2 val macro-F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
