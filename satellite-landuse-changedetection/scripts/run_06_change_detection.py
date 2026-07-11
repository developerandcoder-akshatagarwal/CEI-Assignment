"""
Change detection: embeddings -> cosine similarity -> ROC -> threshold ->
heatmaps. Run from the project root with:
    python -m scripts.run_06_change_detection
"""
import os
import yaml
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
from src.data.datasets import EuroSATDataset
from src.data.transforms import get_eval_transform
from src.data.region_simulator import build_pairs
from src.models.resnet_classifier import build_resnet18
from src.models.embedding_extractor import EmbeddingExtractor
from src.change_detection.similarity import cosine_similarity_single
from src.change_detection.roc_threshold import compute_roc, select_threshold_youden, plot_roc
from src.change_detection.heatmap import compute_patch_similarity_grid, plot_heatmap
from src.training.engine import load_checkpoint


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = cfg["data"]["eurosat_classes"]

    os.makedirs("outputs/roc_curves", exist_ok=True)
    os.makedirs("outputs/change_heatmaps", exist_ok=True)

    ds = EuroSATDataset(cfg["paths"]["data_raw"], transform=get_eval_transform(cfg["data"]["image_size"]), download=False)
    model = build_resnet18(num_classes=len(classes), pretrained=False)
    model = load_checkpoint(model, f"{cfg['paths']['checkpoints']}/resnet18_eurosat_finetuned.pt", device).to(device)
    embedder = EmbeddingExtractor(model).to(device)

    indices_by_class = defaultdict(list)
    for i in range(len(ds)):
        _, label, _ = ds[i]
        indices_by_class[label].append(i)

    pairs = build_pairs(indices_by_class, num_pairs=cfg["change_detection"]["num_regions"] * 15,
                         change_fraction=cfg["change_detection"]["change_fraction"], seed=cfg["seed"])
    print("Built", len(pairs), "T1/T2 pairs")

    similarities, changed_labels = [], []
    with torch.no_grad():
        for p in pairs:
            img1, _, _ = ds[p["t1_idx"]]
            img2, _, _ = ds[p["t2_idx"]]
            e1 = embedder(img1.unsqueeze(0).to(device))
            e2 = embedder(img2.unsqueeze(0).to(device))
            similarities.append(cosine_similarity_single(e1, e2))
            changed_labels.append(int(p["changed"]))

    roc_result = compute_roc(similarities, changed_labels)
    threshold = select_threshold_youden(roc_result)
    print(f"AUC = {roc_result['auc']:.4f} | Selected similarity threshold = {threshold:.4f}")
    plot_roc(roc_result, save_path="outputs/roc_curves/change_detection_roc.png")
    print("Saved: outputs/roc_curves/change_detection_roc.png")

    # Generate >=5 heatmaps for sample pairs
    sample_pairs = pairs[:5]
    for i, p in enumerate(sample_pairs):
        img1, _, _ = ds[p["t1_idx"]]
        img2, _, _ = ds[p["t2_idx"]]
        sim_grid = compute_patch_similarity_grid(embedder, img1, img2, device, grid_size=8)
        plot_heatmap(sim_grid, save_path=f"outputs/change_heatmaps/pair_{i}.png",
                     title=f"Pair {i} (ground truth changed={p['changed']})")
        plt.close()
    print("Saved 5 heatmaps to outputs/change_heatmaps/")


if __name__ == "__main__":
    main()
