# src/test_smoke.py
import torch

def run_smoke_test(trainer, buffer):
    print("\n🔥 開始 Trainer Smoke Test...")
    
    # 1. 記錄更新前參數 (Policy 與 Critic 各挑一層)
    # 若為 ParameterDict，取其中一個鍵；若為常規 Module，取一個 weight
    try:
        policy_before = trainer.agent.policy.params["node_filter_mu"].detach().clone()
    except AttributeError:
        policy_before = next(trainer.agent.policy.parameters()).detach().clone()
        
    critic_before = next(trainer.agent.critic.parameters()).detach().clone()

    # 2. 執行 PPO 更新
    stats = trainer.update(buffer, ppo_epochs=1, batch_size=8)
    print(f"📊 更新統計: {stats}")

    # 3. 提取更新後參數
    try:
        policy_after = trainer.agent.policy.params["node_filter_mu"]
    except AttributeError:
        policy_after = next(trainer.agent.policy.parameters())
        
    critic_after = next(trainer.agent.critic.parameters())

    # 4. 驗證是否發生變化
    p_changed = not torch.equal(policy_before, policy_after)
    c_changed = not torch.equal(critic_before, critic_after)

    print(f"Policy 參數已更新: {'✅' if p_changed else '❌'}")
    print(f"Critic 參數已更新: {'✅' if c_changed else '❌'}")

    assert p_changed, "Policy 參數未更新，計算圖斷裂！"
    assert c_changed, "Critic 參數未更新，計算圖斷裂！"
    print("🎉 Smoke Test 通過！PPO 訓練迴圈已正式打通。")