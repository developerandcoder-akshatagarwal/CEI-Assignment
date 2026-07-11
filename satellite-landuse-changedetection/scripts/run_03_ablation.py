"""
Ablation table: baseline vs phase1 (frozen) vs phase2 (fine-tuned), plus
the fine-tuned model's confusion matrix. Run from the project root with:
    python -m scripts.run_03_ablation
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
from src.models.resnet_classifier import build_resnet18
from src.training.engine import evaluate, load_checkpoint
from src.evaluation.metrics import plot_confusion_matrix


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = cfg["data"]["eurosat_classes"]

    os.makedirs("outputs/confusion_matrices", exist_ok=True)

    ds = EuroSATDataset(cfg["paths"]["data_raw"], transform=get_eval_transform(cfg["data"]["image_size"]), download=False)
    splits = spatial_block_split(ds, val_fraction=cfg["data"]["val_fraction"], test_fraction=cfg["data"]["test_fraction"], seed=cfg["seed"])
    val_loader = DataLoader(Subset(ds, splits["val"]), batch_size=cfg["data"]["batch_size"])
    criterion = nn.CrossEntropyLoss()

    results = {}
    for name, ckpt, builder in [
        ("baseline_cnn", "baseline_cnn.pt", lambda: BaselineCNN(num_classes=len(classes), image_size=cfg["data"]["image_size"])),
        ("phase1_frozen", "resnet18_phase1.pt", lambda: build_resnet18(num_classes=len(classes), pretrained=False)),
        ("phase2_finetuned", "resnet18_eurosat_finetuned.pt", lambda: build_resnet18(num_classes=len(classes), pretrained=False)),
    ]:
        model = builder()
        model = load_checkpoint(model, f"{cfg['paths']['checkpoints']}/{ckpt}", device).to(device)
        m = evaluate(model, val_loader, criterion, device)
        results[name] = m["macro_f1"]
        print(f"{name}: macro-F1 = {m['macro_f1']:.4f}")
        if name == "phase2_finetuned":
            plot_confusion_matrix(m["labels"], m["preds"], classes,
                                   save_path="outputs/confusion_matrices/phase2_cm.png",
                                   title="Phase 2 (Fine-tuned) Confusion Matrix")
            print("Saved: outputs/confusion_matrices/phase2_cm.png")

    print("\nAblation table:", results)


if __name__ == "__main__":
    main()
