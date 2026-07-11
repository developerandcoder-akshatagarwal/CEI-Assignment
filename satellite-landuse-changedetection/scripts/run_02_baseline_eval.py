"""
Baseline CNN evaluation: per-class F1, macro-F1, confusion matrix.
Run from the project root with:
    python -m scripts.run_02_baseline_eval
"""
import os
import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from src.data.datasets import EuroSATDataset
from src.data.transforms import get_eval_transform
from src.data.splits import spatial_block_split
from src.models.baseline_cnn import BaselineCNN
from src.training.engine import evaluate, load_checkpoint
from src.evaluation.metrics import per_class_f1, plot_confusion_matrix


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = cfg["data"]["eurosat_classes"]

    os.makedirs("outputs/confusion_matrices", exist_ok=True)

    ds = EuroSATDataset(cfg["paths"]["data_raw"], transform=get_eval_transform(cfg["data"]["image_size"]), download=False)
    splits = spatial_block_split(ds, val_fraction=cfg["data"]["val_fraction"], test_fraction=cfg["data"]["test_fraction"], seed=cfg["seed"])
    val_loader = DataLoader(Subset(ds, splits["val"]), batch_size=cfg["data"]["batch_size"])

    model = BaselineCNN(num_classes=len(classes), image_size=cfg["data"]["image_size"])
    model = load_checkpoint(model, f"{cfg['paths']['checkpoints']}/baseline_cnn.pt", device).to(device)

    metrics = evaluate(model, val_loader, nn.CrossEntropyLoss(), device)
    print("Baseline macro-F1:", metrics["macro_f1"])
    print("Per-class F1:", per_class_f1(metrics["labels"], metrics["preds"], classes))

    plot_confusion_matrix(metrics["labels"], metrics["preds"], classes,
                           save_path="outputs/confusion_matrices/baseline_cnn_cm.png",
                           title="Baseline CNN Confusion Matrix")
    print("Saved: outputs/confusion_matrices/baseline_cnn_cm.png")


if __name__ == "__main__":
    main()
