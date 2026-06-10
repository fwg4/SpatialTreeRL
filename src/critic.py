# src/critic.py
import torch
import torch.nn as nn
from state import StateBatch

class MetroTreeCritic(nn.Module):
    def __init__(self, feature_dim=15, hidden_dim=64):
        super().__init__()

        self.station_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, state_batch: StateBatch) -> torch.Tensor:
        features = state_batch.features
        masks = state_batch.station_mask

        x = self.station_encoder(features)
        masks = masks.unsqueeze(-1).float()

        denom = masks.sum(dim=-2).clamp(min=1.0)
        pooled = (x * masks).sum(dim=-2) / denom

        return self.value_head(pooled).squeeze(-1)