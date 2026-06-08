# src/evaluate_policies.py
import os
import glob
import re
import torch
import numpy as np
import logging

from env import MiniMetroEnv
from ga import DummyGARegistry
from runner import smdp_generator
from my_utils import format_time_ms, set_global_seed

# 引入 Agent 相關模組
from state_encoder import StateEncoder
from policy import MetroTreePolicy_GATopo
from critic import MetroTreeCritic
from decision_maker import GATopoDecisionMaker
from agent import MetroTreeAgent

# 引入你的 baseline 邏輯
from naive_baseline import build_ring_actions

logging.basicConfig(level=logging.INFO, format="%(message)s")

def run_agent_evaluation(env, agent, seed):
    """使用神經網路 Agent 進行測試"""
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
                return (obs["structured"]["time_ms"], obs["structured"]["score"], "OK")

            elif msg_type == "game_crash":
                reason = event["info"].get("crash_reason", "Unknown")
                return (0.0, 0, f"Crash: {reason}")

            else:
                event = next(gen)

    except Exception as e:
        return (0.0, 0, f"Error: {str(e)}")

def run_dumb_ring_evaluation(env, seed):
    """使用你的 Dumb Ring Baseline 進行測試"""
    gen = smdp_generator(env, seed=seed)
    
    try:
        event = next(gen)
        while True:
            msg_type = event["type"]

            if msg_type == "request_decision":
                actions = build_ring_actions(event["obs"], event["active_lines"])
                event = gen.send(actions)

            elif msg_type == "game_over":
                obs = event["obs"]
                return (obs["structured"]["time_ms"], obs["structured"]["score"], "OK")

            elif msg_type == "game_crash":
                return (0.0, 0, "Crash")

            else:
                event = next(gen)

    except Exception as e:
        return (0.0, 0, f"Error: {str(e)}")

def format_stats(data_list, is_time=False):
    """將數據轉換為 Min / Mean / Max 格式"""
    if not data_list:
        return "--- / --- / ---"
    
    min_val = np.min(data_list)
    mean_val = np.mean(data_list)
    max_val = np.max(data_list)
    
    if is_time:
        return f"{format_time_ms(min_val)} / {format_time_ms(mean_val)} / {format_time_ms(max_val)}"
    else:
        return f"{min_val:^5.1f} / {mean_val:^5.1f} / {max_val:^5.1f}"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt_ms = 16
    test_seeds = [11, 42, 99, 110, 255] # 5 組固定的測試環境
    save_dir = "checkpoints"
    
    print("\n" + "="*85)
    print(f"🚇 Mini Metro Policy Benchmark (dt_ms={dt_ms}, {len(test_seeds)} Seeds)")
    print("="*85)
    print(f"{'Model/Epoch':<15} | {'Survival Time (Min / Mean / Max)':<30} | {'Score (Min / Mean / Max)':<25}")
    print("-" * 85)

    env = MiniMetroEnv(dt_ms=dt_ms)

    # ---------------------------------------------------------
    # 1. 執行 Dumb Ring Baseline
    # ---------------------------------------------------------
    times_base, scores_base = [], []
    for seed in test_seeds:
        t, s, status = run_dumb_ring_evaluation(env, seed)
        if status == "OK":
            times_base.append(t)
            scores_base.append(s)
            
    print(f"{'Dumb Ring':<15} | {format_stats(times_base, True):<30} | {format_stats(scores_base, False):<25}")

    # ---------------------------------------------------------
    # 2. 準備 Agent (加載 GA Registry 與架構)
    # ---------------------------------------------------------
    ga_path = os.path.join(save_dir, "ga.json")
    if os.path.exists(ga_path):
        ga_registry = DummyGARegistry.load_from_file(ga_path)
    else:
        ga_registry = DummyGARegistry()

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
    # 3. 搜尋並排序所有的 Checkpoints
    # ---------------------------------------------------------
    pth_files = glob.glob(os.path.join(save_dir, "policy_ep_*.pth"))
    
    # 解析 epoch 數字並排序
    epochs = []
    for f in pth_files:
        match = re.search(r"policy_ep_(\d+)\.pth", f)
        if match:
            epochs.append((int(match.group(1)), f))
    epochs.sort(key=lambda x: x[0]) # 按照 Epoch 從小到大排序

    # ---------------------------------------------------------
    # 4. 逐一載入並測試模型
    # ---------------------------------------------------------
    for ep_num, pth_path in epochs:
        agent.policy.load_state_dict(torch.load(pth_path, map_location=device))
        
        times, scores = [], []
        crashes = 0
        
        for seed in test_seeds:
            t, s, status = run_agent_evaluation(env, agent, seed)
            if status == "OK":
                times.append(t)
                scores.append(s)
            else:
                crashes += 1

        label = f"Epoch {ep_num}"
        if crashes > 0:
            label += f" ({crashes}x 💥)"

        print(f"{label:<15} | {format_stats(times, True):<30} | {format_stats(scores, False):<25}")

    print("="*85 + "\n")

if __name__ == "__main__":
    main()