# src/state.py

from dataclasses import dataclass
import torch


@dataclass
class StateBatch:
    features: torch.Tensor
    station_mask: torch.Tensor
