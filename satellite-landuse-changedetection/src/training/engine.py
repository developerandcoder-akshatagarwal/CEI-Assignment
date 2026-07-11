"""Shared train/val loop used by all three training scripts (baseline,
phase1, phase2), so behavior is identical across the ablation table."""
import numpy as np
import torch


def _macro_f1_numpy(labels, preds):
    """Manual macro-F1, no scikit-learn -- avoids a reproducible crash
    seen combining torch + scikit-learn on some Windows setups. Computes
    per-class precision/recall/F1 from a confusion matrix, then averages."""
    labels = np.array(labels)
    preds = np.array(preds)
    num_classes = int(max(labels.max(), preds.max())) + 1

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[t, p] += 1

    f1s = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        images, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch in loader:
        images, labels = batch[0].to(device), batch[1].to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    macro_f1 = _macro_f1_numpy(all_labels, all_preds)
    return {
        "loss": total_loss / len(loader.dataset),
        "macro_f1": macro_f1,
        "preds": all_preds,
        "labels": all_labels,
    }


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model
