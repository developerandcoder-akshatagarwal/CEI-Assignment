"""
Smoke test — verifies every module imports and a forward pass works,
without needing the real datasets downloaded. Run before every commit:
    python -m pytest tests/test_smoke.py -v
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch


def test_baseline_cnn_forward():
    from src.models.baseline_cnn import BaselineCNN
    model = BaselineCNN(num_classes=10, image_size=64)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 10)


def test_resnet18_build_and_freeze():
    from src.models.resnet_classifier import build_resnet18, freeze_backbone, unfreeze_layers
    model = build_resnet18(num_classes=10, pretrained=False)
    model = freeze_backbone(model)
    frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
    assert frozen_count > 0

    model = unfreeze_layers(model, layer_names=("layer3", "layer4"))
    trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
    assert trainable_count > 0

    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 10)


def test_embedding_extractor():
    from src.models.resnet_classifier import build_resnet18
    from src.models.embedding_extractor import EmbeddingExtractor
    model = build_resnet18(num_classes=10, pretrained=False)
    embedder = EmbeddingExtractor(model)
    x = torch.randn(2, 3, 64, 64)
    out = embedder(x)
    assert out.shape == (2, 512)


def test_cosine_similarity():
    from src.change_detection.similarity import cosine_similarity_batch
    a = torch.randn(4, 512)
    b = torch.randn(4, 512)
    sims = cosine_similarity_batch(a, b)
    assert sims.shape == (4,)
    assert (sims >= -1).all() and (sims <= 1).all()


def test_risk_level_and_explanation():
    from src.transition.transition_rules import risk_level, generate_explanation
    assert risk_level(0.95)["level"] == "stable"
    assert risk_level(0.70)["level"] == "moderate"
    assert risk_level(0.30)["level"] == "significant"

    explanation = generate_explanation("Forest", "Residential", 0.4)
    assert explanation["risk_level"] == "significant"
    assert "previous_class" in explanation


def test_class_mapping():
    from src.evaluation.class_mapping import build_ucm_to_eurosat_label_map
    eurosat_classes = ["Forest", "Residential", "Highway"]
    ucm_classes = ["forest", "denseresidential", "freeway", "airplane"]
    mapping = build_ucm_to_eurosat_label_map(eurosat_classes, ucm_classes)
    assert mapping[ucm_classes.index("forest")] == 0
    assert ucm_classes.index("airplane") not in mapping
