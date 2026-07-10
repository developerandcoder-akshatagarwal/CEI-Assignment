"""
Trains the baseline scratch CNN. Time-boxed deliberately: this exists
only to show transfer learning is better. Run it once, log the metrics,
move on — don't tune it further.

Usage:
    python -m src.training.train_baseline
"""
import yaml
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.datasets import EuroSATDataset
from src.data.transforms import get_train_transform, get_eval_transform
from src.data.splits import spatial_block_split
from src.models.baseline_cnn import BaselineCNN
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

    model = BaselineCNN(num_classes=len(cfg["data"]["eurosat_classes"]), image_size=cfg["data"]["image_size"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["baseline"]["lr"])

    best_f1 = 0.0
    for epoch in tqdm(range(cfg["training"]["baseline"]["epochs"]), desc="baseline epochs"):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}")
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            save_checkpoint(model, f"{cfg['paths']['checkpoints']}/baseline_cnn.pt")

    print(f"Best baseline val macro-F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
