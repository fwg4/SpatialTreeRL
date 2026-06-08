# src/engine.py
import torch
from state import TensorizedState

class SpatialEngine:
    """
    客觀幾何與符號執行引擎。
    包含 Policy 專用的 MFA (聚合) 與 Action 專用的 MFT (拓撲)。
    """

    @staticmethod
    def _get_mask(
        state: TensorizedState,
        feature_idx: int,
        theta_filter: float,
        is_negated: bool
    ) -> torch.Tensor:
        """
        [內部運算元] 純粹的 Filtration
        """
        condition = state.features[:, feature_idx] < theta_filter
        
        if is_negated:
            condition = ~condition
            
        return state.station_mask & condition

    # ==========================================
    # Policy 端：MFA (Mapping -> Filtration -> Aggregation)
    # ==========================================
    @staticmethod
    def compute_routing_scalar(
        state: TensorizedState,
        node_config: dict,
        theta_filter: float,
    ) -> torch.Tensor:
        """
        輸出客觀純量，供 Policy 進行 Bernoulli 路由。
        """
        mask = SpatialEngine._get_mask(
            state, node_config["feature_idx"], theta_filter, node_config["is_negated"])
        valid_targets = state.features[mask, node_config["feature_idx"]]
        
        if valid_targets.numel() == 0:
            return torch.zeros(1, device=state.features.device)

        agg_type = node_config.get("agg_type", "quantile")
        
        if agg_type == "quantile":
            return torch.quantile(
                valid_targets.float(), 
                node_config.get("q_value", 0.5), 
                dim=0, 
                keepdim=True
            )
        elif agg_type == "mean":
            return valid_targets.mean(dim=0, keepdim=True)
        elif agg_type == "std":
            return valid_targets.std(dim=0, keepdim=True) if valid_targets.numel() > 1 else torch.zeros(1, device=state.features.device)
        else:
            raise ValueError(f"Unknown agg_type: {agg_type}")

    # ==========================================
    # Action 端：MFT (Mapping -> Filtration -> Topology)
    # ==========================================
    @staticmethod
    def compute_topology_sequence(
            state: TensorizedState,
            leaf_config: dict,
            theta_filter: float,
            choose_coord: torch.Tensor,
            eps: float = 1e-5,
    ) -> torch.Tensor:
        """
        輸出節點索引序列，供環境進行實體軌道連線。
        choose_coord: 網路抽樣出的中心點座標 [x, y]
        """
        # 1. Filtration (與 Policy 共用底層邏輯)
        mask = SpatialEngine._get_mask(
            state, leaf_config["feature_idx"], theta_filter, leaf_config["is_negated"])

        global_ids = torch.nonzero(mask).squeeze(1)

        if global_ids.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=state.features.device), False
        
        # 取得被選中站點的原始 2D 座標 (假設特徵的前兩維度是 x, y)
        valid_nodes = state.features[global_ids, :2]
        topo = leaf_config.get("topo", "line")
        loop = False
        # 2. Mapping & 3. Ordering
        if topo == "ring":
            loop = True
            # [Mapping: Polar] 算角度
            angles = torch.atan2(valid_nodes[:, 1] - choose_coord[1],
                                 valid_nodes[:, 0] - choose_coord[0])
            # [Ordering: Sort] 取得局部排序索引
            local_sort_idx = torch.argsort(angles)

        elif topo == "line":
            # [Mapping: Projection] 算線性投影
            direction = valid_nodes.mean(dim=0) - choose_coord

            # 邊界防禦：如果重心剛好等於採樣點，賦予預設方向
            if torch.norm(direction) < eps:
                direction = torch.tensor([1.0, 0.0], device=direction.device)

            projection = (valid_nodes - choose_coord) @ direction
            # [Ordering: Sort] 取得局部排序索引
            local_sort_idx = torch.argsort(projection)

        else:
            raise ValueError(f"Unknown topology: {topo}")

        return global_ids[local_sort_idx], loop
