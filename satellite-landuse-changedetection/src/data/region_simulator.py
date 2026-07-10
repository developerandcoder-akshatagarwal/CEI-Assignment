"""
Builds T1/T2 image pairs for the change-detection module.

We need labeled "changed"/"unchanged" pairs to draw an ROC curve at all —
EuroSAT has no real before/after imagery, so we simulate it:
  - "unchanged" pair: two different tiles of the SAME class (T1 and T2
    should look similar/embed close together)
  - "changed" pair: two tiles of DIFFERENT classes (T1 and T2 should
    embed far apart)

This is a documented synthetic-label construction, not real temporal
ground truth — say so plainly in the report.
"""
import random


def build_pairs(dataset_indices_by_class, num_pairs=200, change_fraction=0.3, seed=42):
    """
    dataset_indices_by_class: dict[class_idx] -> list of dataset indices

    Returns a list of dicts: {"t1_idx", "t2_idx", "changed": bool}
    """
    rng = random.Random(seed)
    classes = list(dataset_indices_by_class.keys())
    pairs = []

    n_changed = int(num_pairs * change_fraction)
    n_unchanged = num_pairs - n_changed

    # unchanged: same class, two different tiles
    for _ in range(n_unchanged):
        c = rng.choice(classes)
        pool = dataset_indices_by_class[c]
        if len(pool) < 2:
            continue
        t1, t2 = rng.sample(pool, 2)
        pairs.append({"t1_idx": t1, "t2_idx": t2, "changed": False})

    # changed: two different classes
    for _ in range(n_changed):
        c1, c2 = rng.sample(classes, 2)
        t1 = rng.choice(dataset_indices_by_class[c1])
        t2 = rng.choice(dataset_indices_by_class[c2])
        pairs.append({"t1_idx": t1, "t2_idx": t2, "changed": True})

    rng.shuffle(pairs)
    return pairs
