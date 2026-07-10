"""Per-class F1, macro-F1, and confusion matrix plotting."""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix


def per_class_f1(labels, preds, class_names):
    scores = f1_score(labels, preds, average=None)
    return dict(zip(class_names, scores))


def macro_f1(labels, preds):
    return f1_score(labels, preds, average="macro")


def plot_confusion_matrix(labels, preds, class_names, save_path=None, title="Confusion Matrix"):
    cm = confusion_matrix(labels, preds)
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
