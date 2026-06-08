# src/interfaces.py
from typing import Protocol, Optional, Dict
import torch
from enum import Enum, auto

class IGARegistry(Protocol):
    """定義 GA 結構註冊表的標準介面"""
    tree_structure: dict
    def get_node_config(self, node_id: int) -> Optional[dict]: ...
    def get_leaf_config(self, leaf_id: int) -> dict: ...

class ITreePolicy(Protocol):
    """定義決策樹策略網路的標準介面"""
    def get_node_filter_dist(self, node_ids: torch.Tensor): ...
    def get_node_route_dist(self, node_ids: torch.Tensor, scalar_x: torch.Tensor): ...
    def get_leaf_dist(self, leaf_ids: torch.Tensor, active_lines: int): ...
    

class RolloutStatus(Enum):
    # 正常推進
    STEP_OK = auto()
    DECISION_OK = auto()
    
    # 異常中斷 (由 Trainer 決定要扣幾分)
    CRASH_TOPOLOGY_INVALID = auto()  # 模型給出少於 2 個站的廢線
    CRASH_ENV_ERROR = auto()         # 環境 pause/resume/remove 發生預期外錯誤
    
    # 自然結束
    GAME_OVER = auto()               # 乘客塞爆或時間結束