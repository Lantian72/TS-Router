from typing import Tuple
import torch
import torch.nn as nn


class SoftLabelMLP(nn.Module):
    """Four-head router; each head predicts competence for the released specialist pool."""
    def __init__(self, input_dim: int, hidden_layers: Tuple[int, int, int], num_classes: int, num_metrics: int, dropout: float = 0.2):
        super().__init__()
        self.num_metrics = num_metrics
        self.num_classes = num_classes
        h1, h2, h3 = hidden_layers
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(h3, num_classes * num_metrics),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output: (B, num_classes * num_metrics) -> (B, num_metrics, num_classes)
        logits_flat = self.net(x)  # (B, num_classes * num_metrics)
        logits = logits_flat.view(-1, self.num_metrics, self.num_classes)
        return logits
