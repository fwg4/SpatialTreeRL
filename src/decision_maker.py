# src/decision_maker.py
import torch
import torch.nn as nn
from engine import SpatialEngine


class GATopoDecisionMaker(nn.Module):
    def __init__(self, ga_registry, device):
        super().__init__()
        self.registry = ga_registry
        self.device = device

        self.id_tensor_cache = {nid: torch.tensor([nid], device=device)
                                for nid in ga_registry.node_configs.keys()}
        self.leaf_tensor_cache = {lid: torch.tensor([lid], device=device)
                                  for lid in ga_registry.leaf_configs.keys()}

    def _sample_dist(self, dist):
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()

    def make_tree_decision(self, state, policy, active_lines):
        """
        將原 runner 中的 _make_tree_decision 邏輯重構至此。
        """
        current_id = 0
        atomic_decisions = []

        # --- A. Node Routing (Tree Traversal) ---
        while self.registry.get_node_config(current_id) is not None:
            config = self.registry.get_node_config(current_id)
            node_id_tensor = self.id_tensor_cache[current_id]

            # Filter 決策
            f_dist = policy.get_atomic_dist("node_filter", node_id_tensor)
            f_act, f_logp = self._sample_dist(f_dist)

            # 物理映射 (使用引擎工具)
            physical_filter = (f_act + 1.0) / 2.0
            scalar_x = SpatialEngine.compute_routing_scalar(
                state, config, physical_filter)

            # Route 決策
            r_dist = policy.get_atomic_dist(
                "node_route", node_id_tensor, {"scalar_x": scalar_x})
            r_act, r_logp = self._sample_dist(r_dist)

            atomic_decisions.extend([
                {"type": "node_filter", "id": current_id,
                    "inputs": {}, "sample": f_act, "log_prob": f_logp},
                {"type": "node_route", "id": current_id, "inputs": {
                    "scalar_x": scalar_x.item()}, "sample": r_act, "log_prob": r_logp}
            ])

            direction = "right" if r_act == 1.0 else "left"
            current_id = self.registry.tree_structure[current_id][direction]

        # --- B. Leaf Action (Topology Computation) ---
        leaf_config = self.registry.get_leaf_config(current_id)
        leaf_id_tensor = self.leaf_tensor_cache[current_id]

        env_action_dict = {}
        for l_idx in range(active_lines):
            # 採樣 Filter 與 Coordinates
            leaf_actions = {}
            for key in ["filter", "x", "y"]:
                d_type = f"leaf_{key}_{l_idx}"
                dist = policy.get_atomic_dist(d_type, leaf_id_tensor)
                act, logp = self._sample_dist(dist)
                leaf_actions[key] = act
                atomic_decisions.append({"type": d_type, "id": current_id, "inputs": {
                }, "sample": act, "log_prob": logp})

            # MFT 運算
            seq, is_loop = SpatialEngine.compute_topology_sequence(
                state, leaf_config[l_idx],
                (leaf_actions["filter"] + 1.0) / 2.0,
                torch.tensor([(leaf_actions["x"]+1)/2,
                             (leaf_actions["y"]+1)/2], device=self.device)
            )
            env_action_dict[l_idx] = {
                "type": "create_path", "stations": seq.tolist(), "loop": is_loop}

        return env_action_dict, atomic_decisions
