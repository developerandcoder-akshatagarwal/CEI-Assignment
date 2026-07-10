"""
ROC curve + Youden's J threshold selection for change detection.

Note the label convention: "changed" pairs should have LOW similarity,
so we feed (1 - similarity) as the score to roc_curve so that a higher
score consistently means "more likely changed" — sklearn's roc_curve
expects that convention.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


def compute_roc(similarities, changed_labels):
    """similarities: list/array of cosine similarities per pair.
    changed_labels: list/array of 1 (changed) / 0 (unchanged), matching order.
    """
    scores = 1 - np.array(similarities)  # higher score = more likely changed
    labels = np.array(changed_labels)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": auc}


def select_threshold_youden(roc_result):
    """Youden's J = TPR - FPR, maximized. Returns the similarity
    threshold (not the 1-similarity score) so it's directly usable
    against raw cosine similarity values elsewhere in the pipeline."""
    j_scores = roc_result["tpr"] - roc_result["fpr"]
    best_idx = np.argmax(j_scores)
    best_score_threshold = roc_result["thresholds"][best_idx]
    similarity_threshold = 1 - best_score_threshold
    return similarity_threshold


def plot_roc(roc_result, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(roc_result["fpr"], roc_result["tpr"], label=f"AUC = {roc_result['auc']:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Change Detection ROC Curve")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig
