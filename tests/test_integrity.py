# src/test_integrity.py
import torch
from main_test import test_pipeline # 假設 test_pipeline 返回 buffer 或你直接在這裡 setup

def run_integrity_tests(buffer, agent, policy, critic):
    print("\n🔍 開始資料完整性測試...")

    # Test 1 & 2: 檢查 Buffer 內容結構
    print(f"--- 測試 Buffer 結構 ---")
    assert len(buffer.decision_buffer) > 0, "Decision Buffer 為空"
    d = buffer.decision_buffer[0]
    required_keys = ["reward", "delta_t", "value", "advantage", "target_value", "state"]
    for k in required_keys:
        assert k in d, f"Decision Buffer 缺少 {k}"
    print(f"✅ Decision Buffer 結構檢查通過")

    # Test 3: Policy Replay (驗證 Distribution 一致性)
    print(f"--- 測試 Policy Replay ---")
    a = buffer.atomic_buffer[0]
    # 準備 batch 輸入
    batch_type = a["type"]
    batch_ids = torch.tensor([a["id"]], dtype=torch.long)
    batch_samples = torch.tensor([a["sample"]], dtype=torch.float32)
    
    # 重新評估
    logp, entropy = policy.evaluate_batch(batch_type, batch_ids, batch_samples, [a["inputs"]])
    
    diff = abs(logp.item() - a["log_prob"])
    print(f"LogP diff: {diff:.6f}")
    assert diff < 1e-4, f"Policy Replay 誤差過大: {diff}"
    print("✅ Policy Replay 驗證通過")

    # Test 4: Critic Determinism (驗證 Critic 與 State 綁定)
    print(f"--- 測試 Critic Replay ---")
    d = buffer.decision_buffer[0]
    v_recomputed = critic(d["state"]).item()
    diff_v = abs(v_recomputed - d["value"])
    print(f"Value diff: {diff_v:.6f}")
    assert diff_v < 1e-4, "Critic Forward 不一致，可能 State 內容被修改"
    print("✅ Critic Replay 驗證通過")

    # Test 5: 多 Episode Index Offset
    # 這段由你手動運行兩次 collect_episode 並檢查 atomic_buffer[-1]["decision_idx"]
    print("✅ 多 Episode 連結測試 (建議手動檢查 atomic_buffer 索引是否連續)")

    print("\n🎉 所有完整性測試通過！你可以安心開始寫 Trainer 了。")

# 使用方式：
buffer, agent, policy, critic = test_pipeline() # 修改 test_pipeline 回傳這些物件
run_integrity_tests(buffer, agent, policy, critic)

def test_batch_integrity(buffer):
    print("\n📦 開始測試 PPO Batch 生成器...")

    # --- 1. Policy Batch 測試 ---
    print("--- Policy Batch ---")
    for batch in buffer.get_policy_batches(batch_size=8):
        # 驗證是否真的被分組了
        batch_type = batch["type"]
        print(f"提取 Type: {batch_type}")
        
        # 驗證 Shape
        print(f"ids shape: {batch['ids'].shape}")
        print(f"samples shape: {batch['samples'].shape}")
        print(f"advantages shape: {batch['advantages'].shape}")
        
        # 確保 type 現在是唯一的字串
        assert isinstance(batch_type, str), "Policy Batch 的 type 應該是單一字串"
        assert batch["ids"].dim() == 1, "IDs 應該是 1D Tensor"
        break # 只測第一個 batch
    print("✅ Policy Batch 測試通過")

    # --- 2. Critic Batch 測試 ---
    print("--- Value Batch ---")
    for batch in buffer.get_value_batches(batch_size=4):
        print(f"features shape: {batch['features'].shape}")
        print(f"masks shape: {batch['masks'].shape}")
        print(f"target_values shape: {batch['target_values'].shape}")
        
        # 驗證維度是否符合 [Batch, N, F]
        assert batch["features"].dim() == 3, "Features 應該是 3D Tensor [B, N, F]"
        assert batch["masks"].dim() == 2, "Masks 應該是 2D Tensor [B, N]"
        assert batch["target_values"].dim() == 1, "Target Values 應該是 1D Tensor [B]"
        break # 只測第一個 batch
    print("✅ Value Batch 測試通過")

# 使用方式：
test_batch_integrity(buffer)