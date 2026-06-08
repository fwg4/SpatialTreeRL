# src/main_test.py
import torch
from state_encoder import StateEncoder
from critic import MetroTreeCritic
from agent import MetroTreeAgent
from collector import SMDPCollector
from buffer import PPOBuffer

from ga import DummyGARegistry
from env import MiniMetroEnv
from policy import MetroTreePolicy_GATopo
from decision_maker import GATopoDecisionMaker


def test_pipeline():
    device = torch.device("cpu")
    ga_registry = DummyGARegistry(mean_depth=2.0)

    # 2. 獲取 ID 空間大小 (依照你原本的邏輯)
    max_node_id = max(ga_registry.node_configs.keys()
                      ) if ga_registry.node_configs else 0
    max_leaf_id = max(ga_registry.leaf_configs.keys()
                      ) if ga_registry.leaf_configs else 0
    tensor_capacity = max(max_node_id, max_leaf_id) + 1

    decision_maker = GATopoDecisionMaker(ga_registry, device="cpu")
    # 1. 初始化組件
    encoder = StateEncoder(device=device)
    critic = MetroTreeCritic(feature_dim=15)
    policy = MetroTreePolicy_GATopo(
        num_nodes=tensor_capacity, num_leaves=tensor_capacity).to(device)
    decision_maker = GATopoDecisionMaker(ga_registry, device)

    agent = MetroTreeAgent(encoder, policy, critic, decision_maker)
    collector = SMDPCollector(gamma=0.99, gae_lambda=0.95)
    buffer = PPOBuffer(device=device)
    env = MiniMetroEnv(dt_ms=16)

    # 2. 執行一個 Episode
    print("🚀 開始測試 Pipeline...")

    # 使用 collector 收集
    decision_buf, atomic_buf = collector.collect_episode(env, agent, device)

    # 3. 壓入 Buffer
    buffer.add_episode(decision_buf, atomic_buf)

    # 4. 驗證資料結構
    print(f"✅ 決策點數量: {len(buffer.decision_buffer)}")
    print(f"✅ 原子決策數量: {len(buffer.atomic_buffer)}")

    # 驗證索引對齊
    for atomic in buffer.atomic_buffer:
        idx = atomic["decision_idx"]
        assert idx < len(buffer.decision_buffer), "決策索引越界！"
        # 這裡可以順便印出數據確認格式
        # print(f"原子決策 {atomic['type']} 指向決策點 {idx}")

    print("🎉 Pipeline 測試通過！")

    return buffer, agent, agent.policy, agent.critic


if __name__ == "__main__":
    test_pipeline()
