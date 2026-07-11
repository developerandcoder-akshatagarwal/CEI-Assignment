"""
Top-5 confidently-wrong predictions. Run from the project root with:
    python -m scripts.run_08_error_analysis
"""
import os
import yaml
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from PIL import Image
from src.data.datasets import EuroSATDataset
from src.data.transforms import get_eval_transform
from src.data.splits import spatial_block_split
from src.models.resnet_classifier import build_resnet18
from src.evaluation.error_analysis import top_k_failures, format_failure_report
from src.training.engine import load_checkpoint


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = cfg["data"]["eurosat_classes"]

    os.makedirs("outputs/figures", exist_ok=True)

    ds = EuroSATDataset(cfg["paths"]["data_raw"], transform=get_eval_transform(cfg["data"]["image_size"]), download=False)
    splits = spatial_block_split(ds, val_fraction=cfg["data"]["val_fraction"], test_fraction=cfg["data"]["test_fraction"], seed=cfg["seed"])
    val_loader = DataLoader(Subset(ds, splits["val"]), batch_size=cfg["data"]["batch_size"])

    model = build_resnet18(num_classes=len(classes), pretrained=False)
    model = load_checkpoint(model, f"{cfg['paths']['checkpoints']}/resnet18_eurosat_finetuned.pt", device).to(device)

    failures = top_k_failures(model, val_loader, device, k=5)
    print(format_failure_report(failures, classes))

    for i, (loss, path, true_label, pred_label, conf) in enumerate(failures):
        img = Image.open(path)
        plt.imshow(img)
        plt.title(f"True: {classes[true_label]} | Pred: {classes[pred_label]} ({conf:.1%}) | loss={loss:.2f}")
        plt.axis("off")
        plt.savefig(f"outputs/figures/error_{i}.png", dpi=150)
        plt.close()
        # TODO once real failures exist: write one hypothesis per example
        # (e.g. visually similar textures, seasonal appearance shift, ambiguous boundary)

    print("Saved 5 error examples to outputs/figures/")


if __name__ == "__main__":
    main()
