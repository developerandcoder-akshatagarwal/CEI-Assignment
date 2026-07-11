"""
GradCAM on 3+ representative examples. Run from the project root with:
    python -m scripts.run_07_gradcam
"""
import os
import yaml
import torch
import matplotlib.pyplot as plt
from src.data.datasets import EuroSATDataset
from src.data.transforms import get_eval_transform, IMAGENET_MEAN, IMAGENET_STD
from src.models.resnet_classifier import build_resnet18
from src.explainability.gradcam import build_gradcam, generate_gradcam_overlay
from src.training.engine import load_checkpoint


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = cfg["data"]["eurosat_classes"]

    os.makedirs("outputs/gradcam", exist_ok=True)

    ds = EuroSATDataset(cfg["paths"]["data_raw"], transform=get_eval_transform(cfg["data"]["image_size"]), download=False)
    model = build_resnet18(num_classes=len(classes), pretrained=False)
    model = load_checkpoint(model, f"{cfg['paths']['checkpoints']}/resnet18_eurosat_finetuned.pt", device).to(device)
    model.eval()
    gradcam = build_gradcam(model)

    # Change these indices to specific examples of interest once you've
    # looked at your data -- these three are just a starting sample.
    sample_indices = [0, 500, 1200]
    for idx in sample_indices:
        img, label, path = ds[idx]
        with torch.no_grad():
            pred_idx = model(img.unsqueeze(0).to(device)).argmax(dim=1).item()
        overlay = generate_gradcam_overlay(gradcam, img, pred_idx, IMAGENET_MEAN, IMAGENET_STD)
        plt.imshow(overlay)
        plt.title(f"True: {classes[label]} | Pred: {classes[pred_idx]}")
        plt.axis("off")
        plt.savefig(f"outputs/gradcam/example_{idx}.png", dpi=150)
        plt.close()
        print(f"Saved outputs/gradcam/example_{idx}.png -- True: {classes[label]}, Pred: {classes[pred_idx]}")
        # TODO once real results exist: write 1-2 sentences on which spatial
        # regions drove the prediction and whether that matches intuition.


if __name__ == "__main__":
    main()
