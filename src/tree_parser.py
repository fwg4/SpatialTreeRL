import torch
import torch.nn.functional as F

class TreeParser:
    # 嚴格對齊 15 維度
    _FEATURE_NAMES = [
        "pos_x", "pos_y", "dist_top1", "dist_top2", "dist_far1", 
        "dist_far2", "diff_min", "diff_max", "is_shape_0", "is_shape_1", 
        "is_shape_2", "is_shape_3", "is_shape_4", "is_shape_5", "is_shape_6"
    ]

    @classmethod
    def _get_feature_name(cls, idx):
        return cls._FEATURE_NAMES[idx] if idx < len(cls._FEATURE_NAMES) else f"F{idx}"

    @classmethod
    def parse(cls, registry, policy=None, decisions=None) -> dict:
        active_nodes = set()
        decision_map = {}
        
        if decisions:
            active_nodes = set(d["id"] for d in decisions)
            for d in decisions:
                if d["id"] not in decision_map:
                    decision_map[d["id"]] = {}
                decision_map[d["id"]][d["type"]] = d["sample"]

        # 動態推導 Root ID
        all_nodes = set(registry.node_configs.keys()) | set(registry.leaf_configs.keys())
        children_nodes = set(
            v for info in registry.tree_structure.values() for v in (info['left'], info['right'])
        )
        root_candidates = all_nodes - children_nodes
        root_id = list(root_candidates)[0] if root_candidates else 0

        pos_map = {}
        edges = []

        def traverse(node_id, depth, x_offset):
            is_leaf = node_id in registry.leaf_configs
            node_info = registry.tree_structure.get(node_id)

            if is_leaf or not node_info:
                pos_map[node_id] = (x_offset, -depth * 2.0)
                return x_offset + 2.0 # 加大橫向間距，避免文字重疊

            left_id, right_id = node_info['left'], node_info['right']
            edges.append({"parent": node_id, "child": left_id, "dir": "left"})
            edges.append({"parent": node_id, "child": right_id, "dir": "right"})

            x_mid1 = traverse(left_id, depth + 1, x_offset)
            x_mid2 = traverse(right_id, depth + 1, x_mid1)
            
            pos_map[node_id] = ((pos_map[left_id][0] + pos_map[right_id][0]) / 2.0, -depth * 2.0)
            return x_mid2

        traverse(root_id, 0, 0)

        nodes = []
        for n_id, (nx, ny) in pos_map.items():
            is_leaf = n_id in registry.leaf_configs
            is_active = n_id in active_nodes
            info_text = cls._build_info_text(n_id, is_leaf, registry, policy, decision_map)

            nodes.append({
                "id": n_id, "x": nx, "y": ny,
                "is_leaf": is_leaf, "is_active": is_active,
                "text": info_text
            })

        for edge in edges:
            edge["is_active"] = (edge["parent"] in active_nodes) and (edge["child"] in active_nodes)

        return {"nodes": nodes, "edges": edges}

    @classmethod
    def _build_info_text(cls, node_id, is_leaf, registry, policy, decision_map):
        """極簡化資訊面板：直接呈現數值關係，去除冗餘標籤與分佈符號"""
        has_trace = node_id in decision_map

        if is_leaf:
            # ==========================================
            # 葉節點 (Leaf)
            # ==========================================
            cfg = registry.get_leaf_config(node_id)[0]
            f_name = cls._get_feature_name(cfg['feature_idx'])
            operator = "<" if cfg["is_negated"] else ">="
            
            # 1. 解析 Filter 數值
            s_val = "?"
            if has_trace and "leaf_filter_0" in decision_map[node_id]:
                s_val = f"{decision_map[node_id]['leaf_filter_0']:.2f}"
            elif policy:
                try:
                    s_val = f"{torch.tanh(policy.params['leaf_filter_0_mu'][node_id]).item():.2f}"
                except Exception: pass

            # 2. 解析 Topo 座標數值
            pos_val = "?, ?"
            if has_trace and "leaf_x_0" in decision_map[node_id] and "leaf_y_0" in decision_map[node_id]:
                x = decision_map[node_id]['leaf_x_0']
                y = decision_map[node_id]['leaf_y_0']
                pos_val = f"{x:.2f}, {y:.2f}"
            elif policy:
                try:
                    mx = torch.tanh(policy.params["leaf_x_0_mu"][node_id]).item()
                    my = torch.tanh(policy.params["leaf_y_0_mu"][node_id]).item()
                    pos_val = f"{mx:.2f}, {my:.2f}"
                except Exception: pass

            return f"Filter: {f_name} {operator} {s_val}\nTopo: {cfg['topo'].upper()} | ({pos_val})"

        else:
            # ==========================================
            # 內部節點 (Node)
            # ==========================================
            cfg = registry.get_node_config(node_id)
            f_name = cls._get_feature_name(cfg['feature_idx'])
            
            # 1. 解析 Filter 數值
            s_val = "?"
            if has_trace and "node_filter" in decision_map[node_id]:
                s_val = f"{decision_map[node_id]['node_filter']:.2f}"
            elif policy:
                try:
                    s_val = f"{torch.tanh(policy.params['node_filter_mu'][node_id]).item():.2f}"
                except Exception: pass
            
            # 2. 解析 Cond 閾值數值 (Route Phi)
            phi_val = "?"
            if policy:
                try:
                    phi_val = f"{policy.params['node_route_phi'][node_id].item():.2f}"
                except Exception: pass

            # 組合精簡字串
            agg = cfg["agg_type"].upper()
            if agg == "QUANTILE": 
                agg += f"({cfg['q_value']:.2f})"
            
            operator = "<" if cfg["is_negated"] else ">"

            return f"Filter: Subset = {{ {f_name} < {s_val} }}\nCond: {agg}(Subset) {operator} {phi_val}"