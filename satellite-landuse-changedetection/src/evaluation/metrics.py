"""Per-class F1, macro-F1, and confusion matrix plotting -- computed
manually with numpy, no scikit-learn (avoids a reproducible crash seen
combining torch + scikit-learn on some Windows setups)."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def _confusion_matrix(labels, preds, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


def per_class_f1(labels, preds, class_names):
    num_classes = len(class_names)
    cm = _confusion_matrix(labels, preds, num_classes)
    scores = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        scores.append(f1)
    return dict(zip(class_names, scores))


def macro_f1(labels, preds):
    labels = np.array(labels)
    preds = np.array(preds)
    num_classes = int(max(labels.max(), preds.max())) + 1
    cm = _confusion_matrix(labels, preds, num_classes)
    scores = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        scores.append(f1)
    return float(np.mean(scores))


def plot_confusion_matrix(labels, preds, class_names, save_path=None, title="Confusion Matrix"):
    cm = _confusion_matrix(labels, preds, len(class_names))
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig
