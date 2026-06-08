# src/critic.py
import torch
import torch.nn as nn

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
        
    def forward(self, features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        # 支援單筆 [N, 15] 或批次 [B, N, 15]
        
        # 1. Station Embedding: [..., N, 15] -> [..., N, H]
        x = self.station_encoder(features)
        
        # 2. Mask 廣播處理: [..., N] -> [..., N, 1]
        masks = masks.unsqueeze(-1).float()
        
        # 3. Mask Pooling (重點：改用 dim=-2 針對 Station 維度聚合)
        denom = masks.sum(dim=-2).clamp(min=1.0)
        pooled = (x * masks).sum(dim=-2) / denom  # 輸出: [..., H]
        
        # 4. 輸出 Value 並降維確保形狀正確: [..., 1] -> [...]
        # 單筆輸出 [] (純量 Tensor)，Batch 輸出 [B]
        return self.value_head(pooled).squeeze(-1)