# src/train.py
import os
import torch
import torch.optim as optim
from collections import deque
import numpy as np
from tqdm import tqdm
import glob
import re

from env import MiniMetroEnv
from ga import DummyGARegistry
from runner import smdp_generator
from my_utils import format_time_ms, set_global_seed
from tree_parser import TreeParser


from state_encoder import StateEncoder
from policy import MetroTreePolicy_GATopo
from critic import MetroTreeCritic
from decision_maker import GATopoDecisionMaker
from agent import MetroTreeAgent
from collector import SMDPCollector
from buffer import PPOBuffer
from trainer import PPOTrainer


def main():
    set_global_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 系統啟動 | 運算核心: {device}\n" + "="*70)

    LOAD_CHECKPOINT = False
    SAVE_DIR = "checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)

    ga_path = os.path.join(SAVE_DIR, "ga.json")
    policy_best_path = os.path.join(SAVE_DIR, "policy_best.pth")
    log_path = os.path.join(SAVE_DIR, "parameter_evolution.log")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"🚀 Training Session Initialized\n")

    # ==========================================
    # 1. 雙軌讀取 (GA 結構)
    # ==========================================
    if LOAD_CHECKPOINT and os.path.exists(ga_path):
        ga_registry = DummyGARegistry.load_from_file(ga_path)
    else:
        ga_registry = DummyGARegistry(mean_depth=1.0)
        ga_registry.save_to_file(ga_path)

    max_node_id = max(ga_registry.node_configs.keys()
                      ) if ga_registry.node_configs else 0
    max_leaf_id = max(ga_registry.leaf_configs.keys()
                      ) if ga_registry.leaf_configs else 0
    tensor_capacity = max(max_node_id, max_leaf_id) + 1

    # ==========================================
    # 2. 初始化核心組件與 Agent 封裝
    # ==========================================
    encoder = StateEncoder(device=device)
    policy = MetroTreePolicy_GATopo(
        num_nodes=tensor_capacity, num_leaves=tensor_capacity).to(device)
    critic = MetroTreeCritic(feature_dim=StateEncoder.FEATURE_DIM).to(device)
    decision_maker = GATopoDecisionMaker(ga_registry, device=device)

    # 將所有組件封裝進 Agent
    agent = MetroTreeAgent(encoder, policy, critic, decision_maker).to(device)

    # 權重讀取 (只對 Policy 操作)
    if LOAD_CHECKPOINT:
        ep_files = glob.glob(os.path.join(SAVE_DIR, "policy_ep_*.pth"))
        if ep_files:
            latest_file = max(ep_files, key=lambda f: int(
                re.search(r"ep_(\d+)", f).group(1)))
            agent.policy.load_state_dict(
                torch.load(latest_file, map_location=device))
            print(f"📂 成功接續歷史訓練: {latest_file}")
        elif os.path.exists(policy_best_path):
            agent.policy.load_state_dict(torch.load(
                policy_best_path, map_location=device))
            print(f"📂 找不到指定回合存檔，退回載入最佳權重: {policy_best_path}")

    # ==========================================
    # 3. 初始化優化器、環境與 Trainer
    # ==========================================
    policy_optimizer = optim.Adam(agent.policy.parameters(), lr=3e-3)
    critic_optimizer = optim.Adam(agent.critic.parameters(), lr=1e-2)

    env = MiniMetroEnv(dt_ms=16)
    collector = SMDPCollector(gamma=0.99, gae_lambda=0.99, time_scale=1000.0)
    buffer = PPOBuffer(device=device)

    # Trainer 現在直接持有 Agent 與雙 Optimizer
    trainer = PPOTrainer(
        agent=agent,
        policy_optimizer=policy_optimizer,
        critic_optimizer=critic_optimizer,
        c_entropy=0.0005
    )

    # --- 訓練超參數 ---
    num_episodes = 5000
    update_interval = 100
    save_interval = 200
    batch_size = 32
    ppo_epochs = 20

    recent_rewards = deque(maxlen=update_interval)
    best_avg_reward = -float('inf')

    latest_vloss = 0.0
    latest_ent = 0.0

    # ==========================================
    # 4. 主訓練迴圈
    # ==========================================
    pbar = tqdm(range(1, num_episodes + 1), desc="🚇 Training", unit="ep")

    for ep in pbar:

        # 透過 Collector 收集單回合資料
        decision_buf, atomic_buf = collector.collect_episode(
            env, agent, device)
        buffer.add_episode(decision_buf, atomic_buf)

        # 從最後一個決策點獲取時間長度
        ep_reward = sum(d["reward"] for d in decision_buf)
        recent_rewards.append(ep_reward)

        # --- PPO 更新與評估 ---
        if ep % update_interval == 0 and len(buffer.atomic_buffer) > 0:
            # 觸發 Trainer 進行分離式更新
            train_stats = trainer.update(
                buffer,
                ppo_epochs=ppo_epochs,
                batch_size=batch_size
            )
            latest_vloss = train_stats.get('value_loss', 0.0)
            latest_ent = train_stats.get('entropy', 0.0)

            avg_reward = np.mean(recent_rewards)

            # 儲存最佳模型
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(agent.policy.state_dict(), policy_best_path)

            pbar.set_postfix({
                'Rwd': f"{avg_reward:.1f}",  # 顯示 Reward
                'V-Loss': f"{train_stats.get('value_loss', 0.0):.2f}",
                'Ent': f"{train_stats.get('entropy', 0.0):.3f}",
                'Smpl': len(buffer.atomic_buffer)
            })

            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(
                        f"🚀 Episode {ep} | Avg Reward: {avg_reward:.2f} | Samples: {len(buffer.atomic_buffer)}\n")
                    f.write(f"{'-'*60}\n")
                    f.write(f"[Actor 健康指標]\n")
                    f.write(
                        f"  • Approx KL : {train_stats.get('approx_kl', 0.0):.2e}  (理想: 0.005 ~ 0.02)\n")
                    f.write(
                        f"  • Clip Frac : {train_stats.get('clip_frac', 0.0):.2e}  (理想: 0.05 ~ 0.2)\n")
                    f.write(
                        f"  • Entropy   : {train_stats.get('entropy', 0.0):.4f}  (過低=探索停止)\n")
                    f.write(
                        f"  • P-Loss    : {train_stats.get('policy_loss', 0.0):.4f}\n")
                    f.write(f"[Critic 健康指標]\n")
                    f.write(
                        f"  • Expl. Var : {train_stats.get('explained_var', 0.0):.4f}  (理想: > 0.5, 接近 1.0 最好)\n")
                    f.write(
                        f"  • V-Loss    : {train_stats.get('value_loss', 0.0):.4f}\n")
                    f.write(f"{'='*60}\n")

            except Exception as e:
                pbar.write(f"\n⚠️ [警告] 文字日誌寫入失敗: {e}")

            buffer.clear()

        # --- 定期存檔與輕量級參數追蹤 ---
        if ep % save_interval == 0:
            checkpoint_path = os.path.join(SAVE_DIR, f"policy_ep_{ep}.pth")
            torch.save(agent.policy.state_dict(), checkpoint_path)


if __name__ == "__main__":
    main()
