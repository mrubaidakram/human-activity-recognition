import torch.nn as nn

class ActivityMLP(nn.Module):
    def __init__(self, features: int = 561, classes: int = 6):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.35),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128, classes),
        )
    def forward(self, x): return self.network(x)
