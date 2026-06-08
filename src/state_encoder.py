# src/state_encoder.py
import math
import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from state import TensorizedState


class StateEncoder:
    FEATURE_DIM = 15
    def __init__(
        self,
        device,
        max_stations: int = 20,
        max_shapes: int = 7,
        map_width: float = 1920.0,
        map_height: float = 1080.0
    ):
        self.device = device
        self.max_shapes = max_shapes
        self.scale = torch.tensor([map_width, map_height], device=device)
        
        self.feature_buffer = torch.zeros((max_stations, self.FEATURE_DIM), device=device)
        self.mask_buffer = torch.zeros(max_stations, dtype=torch.bool, device=device)

    @torch.no_grad()
    def encode(
        self,
        obs: dict
    ):
        pos_np = obs["arrays"]["station_positions"]
        shape_np = obs["arrays"]["station_shape_types"]
        N = pos_np.shape[0]

        pos = torch.as_tensor(pos_np, device=self.device)
        shapes = torch.as_tensor(shape_np, device=self.device)

        # One-hot
        shape_onehot = F.one_hot(shapes, num_classes=self.max_shapes).float()

        # 距離矩陣
        normalized_pos = pos / self.scale
        dist = torch.cdist(normalized_pos, normalized_pos) / math.sqrt(2)

        # TopK 特徵 (N, 4)
        dist_for_min = dist.clone().fill_diagonal_(float('inf'))
        dist_for_max = dist.clone().fill_diagonal_(-1.0)

        # 確保維度正確 (N, 1)
        top1 = torch.topk(dist_for_min, k=1, dim=1, largest=False).values
        top2 = torch.topk(dist_for_min, k=2, dim=1,
                          largest=False).values[:, 1:]
        far1 = torch.topk(dist_for_max, k=1, dim=1, largest=True).values
        far2 = torch.topk(dist_for_max, k=2, dim=1, largest=True).values[:, 1:]

        # Shape diff (N, 2)
        shape_mask = shapes.unsqueeze(1) != shapes.unsqueeze(0)
        diff_min = torch.where(shape_mask, dist, float(
            'inf')).min(dim=1).values.unsqueeze(1)
        diff_max = torch.where(
            shape_mask, dist, -1.0).max(dim=1).values.unsqueeze(1)
        # 修正 inf/-1 為 0
        diff_min = torch.where(diff_min == float('inf'), 0.0, diff_min)
        diff_max = torch.where(diff_max == -1.0, 0.0, diff_max)

        # 組合所有特徵 (N, 8 + shapes)
        # 2(pos) + 4(dist) + 2(diff) = 8
        full_features = torch.cat(
            [normalized_pos, top1, top2, far1, far2, diff_min, diff_max, shape_onehot], dim=1)
        self.feature_buffer.zero_()
        self.mask_buffer.zero_()
        self.feature_buffer[:N] = full_features
        self.mask_buffer[:N] = True
        self.mask_buffer[N:] = False
        
        return TensorizedState(
            features=self.feature_buffer.clone(), 
            station_mask=self.mask_buffer.clone()
        )