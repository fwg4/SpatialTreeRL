# src/ga.py
import random
import math
import json


class DummyGARegistry:
    """
    動態隨機生成的 Dummy 樹狀註冊表。
    使用常態分佈 CDF 控制葉節點深度的平均值 (mean) 與變異程度 (std)。
    """

    def __init__(self, mean_depth: float = 3.0, std_depth: float = 2.0, feature_dim: int = 15, max_safe_depth: int = 50, max_lines: int = 4):
        self.tree_structure = {}
        self.node_configs = {}
        self.leaf_configs = {}
        self.mean_depth = mean_depth
        self.std_depth = std_depth
        self.feature_dim = feature_dim
        self.max_safe_depth = max_safe_depth
        self.max_lines = max_lines

        self._next_id = 0
        self._build_tree(node_id=self._get_next_id(), current_depth=0)

    def _get_next_id(self) -> int:
        curr = self._next_id
        self._next_id += 1
        return curr

    def _get_stop_probability(self, depth: int) -> float:
        """計算在當前深度成為葉節點的機率 (使用常態分佈 CDF)"""
        if self.std_depth <= 0:
            return 1.0 if depth >= self.mean_depth else 0.0

        # Normal Distribution CDF formula using math.erf
        z = (depth - self.mean_depth) / (self.std_depth * math.sqrt(2))
        return 0.5 * (1 + math.erf(z))

    def _build_tree(self, node_id: int, current_depth: int):
        # 決定是否生成葉節點：到達絕對安全深度，或命中累積機率
        if current_depth >= self.max_safe_depth:
            is_leaf = True
        else:
            stop_prob = self._get_stop_probability(current_depth)
            is_leaf = random.random() < stop_prob

        if is_leaf:
            self.leaf_configs[node_id] = [
                {
                    "filter_idx": random.randint(0, self.feature_dim - 1),
                    "is_negated": random.choice([True, False]),
                    "topo": random.choice(["line", "ring"]),
                }
                for _ in range(self.max_lines)
            ]
        else:
            left_id = self._get_next_id()
            right_id = self._get_next_id()
            self.tree_structure[node_id] = {"left": left_id, "right": right_id}
            self.node_configs[node_id] = {
                "filter_idx": random.randint(0, self.feature_dim - 1),
                "agg_idx": random.randint(0, self.feature_dim - 1),
                "is_negated": random.choice([True, False]),
                "agg_type": random.choice(["mean", "quantile", "std"]),
                "q_value": random.uniform(0.0, 1.0)
            }

            # 遞迴往下生成
            self._build_tree(left_id, current_depth + 1)
            self._build_tree(right_id, current_depth + 1)

    def get_node_config(self, node_id: int) -> dict | None:
        return self.node_configs.get(node_id, None)

    def get_leaf_config(self, leaf_id: int) -> dict:
        if leaf_id not in self.leaf_configs:
            raise ValueError(f"Tree Route Error: ID {leaf_id} 不是合法的葉節點")
        return self.leaf_configs[leaf_id]

    def save_to_file(self, filepath: str):
        """將動態生成的樹狀結構與配置序列化為 JSON 儲存"""
        data = {
            "tree_structure": self.tree_structure,
            "node_configs": self.node_configs,
            "leaf_configs": self.leaf_configs,
            "feature_dim": self.feature_dim,
            "max_safe_depth": self.max_safe_depth,
            "max_lines": self.max_lines,
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        # print(f"[Registry] 樹狀結構已儲存至 {filepath}")

    @classmethod
    def load_from_file(cls, filepath: str):
        """從 JSON 檔案還原註冊表實例"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 建立一個空實例 (傳入 0 避免觸發不必要的隨機生成)
        instance = cls(mean_depth=0, std_depth=0)

        # 覆寫內部資料 (注意：JSON 會將字典的 int key 轉為 string，必須轉回來)
        instance.tree_structure = {
            int(k): v for k, v in data["tree_structure"].items()}
        instance.node_configs = {
            int(k): v for k, v in data["node_configs"].items()}
        instance.leaf_configs = {
            int(k): v for k, v in data["leaf_configs"].items()}
        instance.feature_dim = data["feature_dim"]
        instance.max_safe_depth = data.get("max_safe_depth", 50)
        instance.max_lines = data.get("max_lines", 4)

        # print(f"[Registry] 已從 {filepath} 成功還原樹狀結構")
        return instance
