import pytest
from env import MiniMetroEnv

def test_reward_structure_diagnosis():
    """
    執行診斷測試，透過觀察 Reward 的集合來判定它是 Pulse 還是 Cumulative
    """
    env = MiniMetroEnv(dt_ms=16)
    env.reset(seed=0)
    
    # 先做一個動作觸發遊戲開始
    env.step({"type": "create_path", "stations": [0, 1, 2], "loop": True})
    env.step({"type": "resume"})

    rewards = []
    print("\n--- 開始診斷 Reward 結構 ---")
    
    for i in range(1000):
        _, reward, done, _ = env.step({"type": "noop"})
        if reward != 0:
            rewards.append(reward)
            print(f"Frame {i}: Detected Reward: {reward}")
        
        if done:
            break
            
    # 邏輯判斷
    unique_rewards = set(rewards)
    print(f"\n診斷結果:")
    print(f"出現過的非零 Reward 值: {unique_rewards}")
    
    # 判斷邏輯
    if len(unique_rewards) > 5:
        # 如果出現非常多種不同的數值，且呈現遞增趨勢，極高機率是 Cumulative
        print("【警告】偵測到多種不同的 Reward 值，可能是 Cumulative Score！")
    else:
        print("【結論】偵測到固定數值的 Reward (如 1.0)，極高機率是 Pulse Reward (事件獎勵)。")

if __name__ == "__main__":
    test_reward_structure_diagnosis()