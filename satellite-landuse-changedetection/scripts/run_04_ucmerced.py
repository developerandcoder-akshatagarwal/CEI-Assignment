"""
UC Merced holdout evaluation. Run from the project root with:
    python -m scripts.run_04_ucmerced
"""
import yaml
import torch
from torch.utils.data import DataLoader
from src.data.datasets import UCMercedDataset
from src.data.transforms import get_eval_transform
from src.models.resnet_classifier import build_resnet18
from src.training.engine import load_checkpoint
from src.evaluation.class_mapping import (
    build_ucm_to_eurosat_label_map,
    evaluate_mapped_subset,
    average_confidence_on_unmapped,
)


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = cfg["data"]["eurosat_classes"]

    ucm_ds = UCMercedDataset(f"{cfg['paths']['data_raw']}/ucmerced", transform=get_eval_transform(cfg["data"]["image_size"]))
    ucm_loader = DataLoader(ucm_ds, batch_size=cfg["data"]["batch_size"])

    model = build_resnet18(num_classes=len(classes), pretrained=False)
    model = load_checkpoint(model, f"{cfg['paths']['checkpoints']}/resnet18_eurosat_finetuned.pt", device).to(device)

    label_map = build_ucm_to_eurosat_label_map(classes, ucm_ds.classes)
    print("Mapped UC Merced classes:", len(label_map), "of", len(ucm_ds.classes))

    # MINIMUM TIER -- always run this
    result = evaluate_mapped_subset(model, ucm_loader, label_map, device)
    print("Mapped-subset accuracy:", result["mapped_subset_accuracy"], "| n =", result["n_evaluated"])
    print("Report line for the write-up: 'Due to class taxonomy mismatch, only semantically aligned classes were evaluated.'")

    # OPTIONAL TIER -- only if you're on schedule (cheap: minutes of runtime)
    conf_result = average_confidence_on_unmapped(model, ucm_loader, label_map, device)
    print("Avg confidence on unmapped classes:", conf_result["avg_confidence_unmapped"], "| n =", conf_result["n_unmapped"])


if __name__ == "__main__":
    main()
