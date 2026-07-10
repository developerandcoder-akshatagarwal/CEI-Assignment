"""Cosine similarity between T1/T2 embeddings — the core change signal."""
import torch
import torch.nn.functional as F


def cosine_similarity_batch(emb_t1, emb_t2):
    """emb_t1, emb_t2: (batch, 512) tensors. Returns (batch,) similarities in [-1, 1]."""
    return F.cosine_similarity(emb_t1, emb_t2, dim=1)


def cosine_similarity_single(emb_t1, emb_t2):
    """Single-pair convenience wrapper for the dashboard (1, 512) tensors."""
    return F.cosine_similarity(emb_t1, emb_t2, dim=1).item()
