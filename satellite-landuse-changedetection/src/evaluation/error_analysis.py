"""Surfaces the top-K confidently-wrong predictions — ranked by
confidence-weighted loss, not just wrong/right, so you see the failures
that matter most (confidently wrong beats barely-wrong for a report)."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def top_k_failures(model, loader, device, k=5):
    model.eval()
    records = []  # (loss, filepath, true_label, pred_label, confidence)
    for images, labels, filepaths in loader:
        images_d, labels_d = images.to(device), labels.to(device)
        logits = model(images_d)
        losses = F.cross_entropy(logits, labels_d, reduction="none")
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        confidences = probs.max(dim=1).values

        for i in range(len(labels)):
            if preds[i] != labels_d[i]:
                records.append((
                    losses[i].item(),
                    filepaths[i],
                    labels[i].item(),
                    preds[i].item(),
                    confidences[i].item(),
                ))

    records.sort(key=lambda r: r[0], reverse=True)
    return records[:k]


def format_failure_report(records, class_names):
    lines = []
    for loss, path, true_label, pred_label, conf in records:
        lines.append(
            f"file={path} | true={class_names[true_label]} | "
            f"predicted={class_names[pred_label]} (confidence={conf:.2f}) | loss={loss:.3f}"
        )
    return "\n".join(lines)
