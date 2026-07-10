"""
Baseline CNN trained from scratch. Its only job is to be a floor metric
that transfer learning clearly beats. Do NOT spend time tuning this —
2-3 hours total including training, per the project roadmap.
"""
import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(self, num_classes=10, image_size=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # /2
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # /4
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # /8
        )
        reduced = image_size // 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * reduced * reduced, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
