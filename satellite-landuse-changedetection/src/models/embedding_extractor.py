"""
Strips the classifier head off a trained ResNet18 to expose the
512-dimensional embedding used for cosine-similarity change detection.
"""
import copy
import torch.nn as nn


class EmbeddingExtractor(nn.Module):
    def __init__(self, trained_resnet18):
        super().__init__()
        model = copy.deepcopy(trained_resnet18)
        model.fc = nn.Identity()
        self.model = model
        self.model.eval()

    def forward(self, x):
        # returns (batch, 512)
        return self.model(x)
