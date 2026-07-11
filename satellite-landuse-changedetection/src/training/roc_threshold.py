"""
ROC curve + Youden's J threshold selection for change detection --
computed manually with numpy, no scikit-learn (avoids a reproducible
crash seen combining torch + scikit-learn on some Windows setups).

Note the label convention: "changed" pairs should have LOW similarity,
so we feed (1 - similarity) as the score so that a higher score
consistently means "more likely changed."
"""
import numpy as np
import matplotlib.pyplot as plt


def compute_roc(similarities, changed_labels):
    """similarities: list/array of cosine similarities per pair.
    changed_labels: list/array of 1 (changed) / 0 (unchanged), matching order.
    """
    scores = 1 - np.array(similarities)  # higher score = more likely changed
    labels = np.array(changed_labels)

    order = np.argsort(-scores)  # descending, so we sweep the threshold down
    scores_sorted = scores[order]
    labels_sorted = labels[order]

    num_positive = labels.sum()
    num_negative = len(labels) - num_positive

    tps = np.cumsum(labels_sorted)
    fps = np.cumsum(1 - labels_sorted)

    tpr = tps / num_positive if num_positive > 0 else np.zeros_like(tps, dtype=float)
    fpr = fps / num_negative if num_negative > 0 else np.zeros_like(fps, dtype=float)

    # prepend the (0, 0) point at threshold = +inf, matching standard ROC convention
    tpr = np.concatenate(([0.0], tpr))
    fpr = np.concatenate(([0.0], fpr))
    thresholds = np.concatenate(([scores_sorted[0] + 1.0], scores_sorted))

    # np.trapezoid is the modern name (numpy 2.0+); np.trapz still exists
    # on older numpy but was removed as of 2.0 -- this covers either version.
    trapezoid_fn = getattr(np, "trapezoid", None) or np.trapz
    auc = float(trapezoid_fn(tpr, fpr))  # fpr is non-decreasing along this sweep

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
