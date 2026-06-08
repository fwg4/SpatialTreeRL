# src/export_tree.py
import torch
from ga import DummyGARegistry
from policy import MetroTreePolicy_GATopo
from tree_parser import TreeParser
from tree_renderer import TreeRenderer

def main():
    device = torch.device("cpu")
    
    # 讀取結構與權重
    ga_registry = DummyGARegistry.load_from_file("checkpoints/ga.json")
    max_id = max(max(ga_registry.node_configs.keys(), default=0), max(ga_registry.leaf_configs.keys(), default=0))
    
    policy = MetroTreePolicy_GATopo(num_nodes=max_id+1, num_leaves=max_id+1).to(device)
    policy.load_state_dict(torch.load("checkpoints/policy_best.pth", map_location=device))
    policy.eval()

    # 解耦工作流：1. 解析資料 -> 2. 渲染匯出
    tree_data = TreeParser.parse(registry=ga_registry, policy=policy, decisions=None)
    TreeRenderer.export_image(tree_data, "outputs/tree_architecture.png")

if __name__ == "__main__":
    main()