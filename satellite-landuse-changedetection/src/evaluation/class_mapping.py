"""
EuroSAT (10 classes) and UC Merced (21 classes) don't map 1:1. This
module holds the explicit mapping and a tiered evaluation function —
always run the minimum tier; add the confidence-distribution tier only
if you're on schedule (see roadmap Day 4).
"""
import torch
import torch.nn.functional as F

# EuroSAT class name -> list of equivalent UC Merced class names
EUROSAT_TO_UCM = {
    "Forest": ["forest"],
    "Residential": ["denseresidential", "mediumresidential", "sparseresidential"],
    "Highway": ["freeway"],
    "River": ["river"],
    "AnnualCrop": ["agricultural"],
    "PermanentCrop": ["agricultural"],
    # No confident UC Merced equivalent for these — left unmapped on purpose:
    # Industrial, Pasture, HerbaceousVegetation, SeaLake
}


def build_ucm_to_eurosat_label_map(eurosat_classes, ucm_classes):
    """Returns dict: ucm_class_idx -> eurosat_class_idx, for only the
    UC Merced classes that have a defined EuroSAT equivalent."""
    ucm_to_eurosat = {}
    for eurosat_name, ucm_names in EUROSAT_TO_UCM.items():
        if eurosat_name not in eurosat_classes:
            continue
        eurosat_idx = eurosat_classes.index(eurosat_name)
        for ucm_name in ucm_names:
            if ucm_name in ucm_classes:
                ucm_to_eurosat[ucm_classes.index(ucm_name)] = eurosat_idx
    return ucm_to_eurosat


def evaluate_mapped_subset(model, loader, ucm_to_eurosat_map, device):
    """MINIMUM TIER — always do this. Evaluates only the UC Merced
    samples whose class has a defined EuroSAT equivalent."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            mask = torch.tensor([l.item() in ucm_to_eurosat_map for l in labels])
            if mask.sum() == 0:
                continue
            images_m = images[mask]
            mapped_labels = torch.tensor(
                [ucm_to_eurosat_map[l.item()] for l in labels[mask]]
            ).to(device)
            preds = model(images_m).argmax(dim=1)
            correct += (preds == mapped_labels).sum().item()
            total += mask.sum().item()
    accuracy = correct / total if total else float("nan")
    return {"mapped_subset_accuracy": accuracy, "n_evaluated": total}


def average_confidence_on_unmapped(model, loader, ucm_to_eurosat_map, device):
    """OPTIONAL TIER — only add if Day 4 is on schedule. ~10 lines,
    minutes of runtime. Reports whether the model is appropriately
    uncertain on classes it's never seen, rather than confidently wrong."""
    model.eval()
    confidences = []
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            mask = torch.tensor([l.item() not in ucm_to_eurosat_map for l in labels])
            if mask.sum() == 0:
                continue
            probs = F.softmax(model(images[mask]), dim=1)
            confidences.extend(probs.max(dim=1).values.cpu().tolist())
    if not confidences:
        return {"avg_confidence_unmapped": float("nan"), "n_unmapped": 0}
    return {
        "avg_confidence_unmapped": sum(confidences) / len(confidences),
        "n_unmapped": len(confidences),
    }
