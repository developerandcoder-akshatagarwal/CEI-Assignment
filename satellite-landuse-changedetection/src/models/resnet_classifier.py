"""
Pretrained ResNet18 classifier with helpers for the two-phase
fine-tuning scheme: Phase 1 freezes the backbone and trains only the
new head; Phase 2 unfreezes layer3+layer4 with a 10x-lower backbone LR.
"""
import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes=10, pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def freeze_backbone(model):
    """Phase 1: everything frozen except the new fc head."""
    for name, param in model.named_parameters():
        param.requires_grad = "fc" in name
    return model


def unfreeze_layers(model, layer_names=("layer3", "layer4")):
    """Phase 2: unfreeze the named blocks (plus fc, which is already
    trainable). Everything else (conv1, bn1, layer1, layer2) stays frozen.
    """
    for name, param in model.named_parameters():
        if any(name.startswith(ln) for ln in layer_names) or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def get_param_groups(model, lr_head, lr_backbone, unfrozen_layers=("layer3", "layer4")):
    """Two-param-group optimizer setup for Phase 2's discriminative LR:
    head trains faster than the newly-unfrozen backbone layers."""
    head_params, backbone_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "fc" in name:
            head_params.append(param)
        elif any(name.startswith(ln) for ln in unfrozen_layers):
            backbone_params.append(param)
    return [
        {"params": head_params, "lr": lr_head},
        {"params": backbone_params, "lr": lr_backbone},
    ]
