"""Shared train/val loop used by all three training scripts (baseline,
phase1, phase2), so behavior is identical across the ablation table."""
import torch
from sklearn.metrics import f1_score


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

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
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
