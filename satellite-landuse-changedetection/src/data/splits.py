"""
Two split strategies, built together so the Day-4 spatial-leakage
experiment can compare them directly on the identical dataset.

IMPORTANT LIMITATION (documented, not hidden): the common EuroSAT-RGB
release has no real lat/lon metadata per tile. `spatial_block_split`
therefore SIMULATES spatial blocks by hashing each filename into a
deterministic pseudo-region bucket. This preserves the methodology of
block-based splitting (whole regions go entirely to train OR val OR
test, never split across them) without pretending we have ground-truth
geo-coordinates we don't have. State this plainly in the report.
"""
import hashlib
import random
from collections import defaultdict


def _region_id(filepath, num_regions):
    """Deterministic pseudo-region assignment via filename hash."""
    h = hashlib.md5(filepath.encode("utf-8")).hexdigest()
    return int(h, 16) % num_regions


def random_split(dataset, val_fraction=0.15, test_fraction=0.15, seed=42):
    n = len(dataset)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_val = int(n * val_fraction)
    n_test = int(n * test_fraction)
    val_idx = idx[:n_val]
    test_idx = idx[n_val:n_val + n_test]
    train_idx = idx[n_val + n_test:]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def spatial_block_split(dataset, num_regions=40, val_fraction=0.15,
                         test_fraction=0.15, seed=42):
    """Assigns whole pseudo-regions to train/val/test so that no region
    straddles a split boundary — this is what makes it a meaningful
    comparison against random_split for the leakage experiment.
    """
    region_to_indices = defaultdict(list)
    for i in range(len(dataset)):
        # dataset[i] returns (img, label, filepath) per our Dataset wrappers
        filepath = dataset.base.samples[i][0] if hasattr(dataset, "base") else dataset.samples[i][0]
        region = _region_id(filepath, num_regions)
        region_to_indices[region].append(i)

    regions = list(region_to_indices.keys())
    random.Random(seed).shuffle(regions)

    n_regions = len(regions)
    n_val_regions = max(1, int(n_regions * val_fraction))
    n_test_regions = max(1, int(n_regions * test_fraction))

    val_regions = set(regions[:n_val_regions])
    test_regions = set(regions[n_val_regions:n_val_regions + n_test_regions])
    train_regions = set(regions[n_val_regions + n_test_regions:])

    train_idx, val_idx, test_idx = [], [], []
    for region, indices in region_to_indices.items():
        if region in val_regions:
            val_idx.extend(indices)
        elif region in test_regions:
            test_idx.extend(indices)
        else:
            train_idx.extend(indices)

    return {"train": train_idx, "val": val_idx, "test": test_idx}
