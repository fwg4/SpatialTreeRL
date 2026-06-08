# src/state.py

from dataclasses import dataclass
import torch


@dataclass
class TensorizedState:

    features: torch.Tensor       # Shape: [max_stations, FEATURE_DIM]
    
    station_mask: torch.Tensor   # Shape: [max_stations], dtype=torch.bool
    # line_mask: torch.Tensor      # Shape: [max_lines], dtype=torch.bool

    station_count: int = 0
    lines_count: int = 0
    score: int = 0

    def reset(self):
        self.features.zero_()
        self.station_mask.zero_()
        # self.line_mask.zero_()
        
        self.station_count = 0
        self.lines_count = 0
        self.score = 0