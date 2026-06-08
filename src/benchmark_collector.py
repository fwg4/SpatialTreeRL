# src/benchmark_collector.py
import os
import torch
import pandas as pd

from env import MiniMetroEnv
from ga import DummyGARegistry
from runner import smdp_generator
from my_utils import set_global_seed

# 引入 Agent 相關模組
from state_encoder import StateEncoder
from policy import MetroTreePolicy_GATopo
from critic import MetroTreeCritic
from decision_maker import GATopoDecisionMaker
from agent import MetroTreeAgent

# 引入 Baseline 邏輯
from naive_baseline import run_dumb_ring_baseline

def run_agent_evaluation(env, agent, seed):
    """執行神經網路單次評估"""
    gen = smdp_generator(env, seed=seed)
    try:
        event = next(gen)
        while True:
            msg_type = event["type"]
            if msg_type == "request_decision":
                with torch.no_grad():
                    result = agent.act(event["obs"], event["active_lines"])
                event = gen.send(result["env_actions"])
            elif msg_type == "game_over":
                obs = event["obs"]
                return obs["structured"]["time_ms"] / 1000.0, obs["structured"]["score"] # 回傳秒數與分數
            elif msg_type == "game_crash":
                return 0.0, 0.0
            else:
                event = next(gen)
    except Exception:
        return 0.0, 0.0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt_ms = 16
    test_seeds = [11, 42, 99, 110, 255]
    epochs_to_test = [1000, 2000, 3000, 4000, 5000]
    
    print("\n" + "="*60)
    print("🚇 [階段 1] 開始執行最終基準測試 (Data Collection)")
    print("="*60)

    results = []
    env = MiniMetroEnv(dt_ms=dt_ms)

    # ---------------------------------------------------------
    # 1. 測試 Baseline (Dumb Ring)
    # ---------------------------------------------------------
    print("⏳ 正在測試 Baseline (Dumb Ring)...")
    for seed in test_seeds:
        set_global_seed(seed)
        time_ms, score, status = run_dumb_ring_baseline(dt_ms, seed)
        results.append({
            "Model": "Baseline",
            "Seed": seed,
            "Survival Time (s)": time_ms / 1000.0 if status == "OK" else 0.0,
            "Score": score if status == "OK" else 0
        })

    # ---------------------------------------------------------
    # 2. 初始化 Agent
    # ---------------------------------------------------------
    SAVE_DIR = "checkpoints"
    ga_path = os.path.join(SAVE_DIR, "ga.json")
    ga_registry = DummyGARegistry.load_from_file(ga_path) if os.path.exists(ga_path) else DummyGARegistry()
    
    max_node_id = max(ga_registry.node_configs.keys()) if ga_registry.node_configs else 0
    max_leaf_id = max(ga_registry.leaf_configs.keys()) if ga_registry.leaf_configs else 0
    tensor_capacity = max(max_node_id, max_leaf_id) + 1

    encoder = StateEncoder(device=device)
    policy = MetroTreePolicy_GATopo(num_nodes=tensor_capacity, num_leaves=tensor_capacity).to(device)
    critic = MetroTreeCritic(feature_dim=StateEncoder.FEATURE_DIM).to(device)
    decision_maker = GATopoDecisionMaker(ga_registry, device=device)
    agent = MetroTreeAgent(encoder, policy, critic, decision_maker).to(device)
    agent.eval()

    # ---------------------------------------------------------
    # 3. 測試各個 Checkpoints
    # ---------------------------------------------------------
    for ep in epochs_to_test:
        pth_path = os.path.join(SAVE_DIR, f"policy_ep_{ep}.pth")
        if not os.path.exists(pth_path):
            print(f"⚠️ 找不到權重檔 {pth_path}，跳過 Epoch {ep}")
            continue
            
        print(f"⏳ 正在測試 Epoch {ep}...")
        agent.policy.load_state_dict(torch.load(pth_path, map_location=device))
        
        for seed in test_seeds:
            set_global_seed(seed)
            time_s, score = run_agent_evaluation(env, agent, seed)
            results.append({
                "Model": f"Ep-{ep}",
                "Seed": seed,
                "Survival Time (s)": time_s,
                "Score": score
            })

    # ---------------------------------------------------------
    # 4. 資料儲存
    # ---------------------------------------------------------
    df = pd.DataFrame(results)
    os.makedirs("outputs", exist_ok=True)
    csv_path = "outputs/benchmark_raw_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ 原始測試數據 (含 Score 與 Seed) 已儲存至 {csv_path}")

if __name__ == "__main__":
    main()