# src/test_batches.py
import torch
import torch.nn.functional as F
from main_test import test_pipeline

def run_batch_smoke_tests(buffer, policy, critic):
    print("\n🔥 開始 Batch Smoke Test (Critic & Policy Replay) ...")
    
    # ==========================================
    # Step 1 & 2: Critic Forward & Gradient Test
    # ==========================================
    print("\n--- 測試 Critic Batch Forward 與 Gradient ---")
    
    # 清空可能殘留的梯度
    critic.zero_grad()
    
    # 取出一個 batch
    v_batch = next(buffer.get_value_batches(batch_size=4))
    features = v_batch["features"]
    masks = v_batch["masks"]
    targets = v_batch["target_values"]
    
    # Forward Pass
    preds = critic(features, masks)
    print(f"Features shape: {features.shape}")
    print(f"Preds shape: {preds.shape}")
    assert preds.dim() == 1 and preds.size(0) == features.size(0), "Critic 輸出維度錯誤"
    
    # Backward Pass
    loss = F.mse_loss(preds, targets)
    loss.backward()
    
    # 檢查 Gradient 是否成功傳遞
    grad_ok = True
    for name, p in critic.named_parameters():
        if p.grad is None:
            print(f"❌ 警告: {name} 沒有梯度！")
            grad_ok = False
        elif torch.all(p.grad == 0):
            print(f"⚠️ 警告: {name} 梯度全為 0！")
    
    if grad_ok:
        print("✅ Critic Forward 與 Gradient 測試通過 (所有參數皆有梯度)")

    # ==========================================
    # Step 3: Policy Batch Replay Test
    # ==========================================
    print("\n--- 測試 Policy Batch Replay ---")
    
    # 清空梯度
    policy.zero_grad()
    
    # 找一個至少有資料的 batch
    p_batch = next(buffer.get_policy_batches(batch_size=8))
    batch_type = p_batch["type"]
    ids = p_batch["ids"]
    samples = p_batch["samples"]
    inputs = p_batch["inputs"]
    old_log_probs = p_batch["log_probs"]
    
    print(f"測試 Type: {batch_type}")
    print(f"Batch Size: {ids.shape[0]}")
    
    # Replay Forward
    new_logp, entropy = policy.evaluate_batch(batch_type, ids, samples, inputs)
    print(f"New LogProb shape: {new_logp.shape}")
    assert new_logp.dim() == 1 and new_logp.size(0) == ids.size(0), "Policy Replay 輸出維度錯誤"
    
    # 確認 LogProb 是否一致 (測試 Determinism)
    diff = (new_logp - old_log_probs).abs().max().item()
    print(f"Max LogProb Diff: {diff:.6f}")
    if diff < 1e-4:
        print("✅ Policy LogProb 一致性測試通過")
    else:
        print("❌ Policy LogProb 一致性失敗！(可能是 inputs 記錄不全)")
        
    # 順便測一下 Policy 的梯度連通性
    dummy_loss = -new_logp.mean()
    dummy_loss.backward()
    
    p_grad_ok = False
    for p in policy.parameters():
        if p.grad is not None and not torch.all(p.grad == 0):
            p_grad_ok = True
            break
            
    if p_grad_ok:
        print("✅ Policy Gradient 連通性測試通過")
    else:
        print("❌ Policy Gradient 傳遞失敗")

    print("\n🎉 所有 Smoke Tests 結束！")
    
buffer, agent, policy, critic = test_pipeline() # 修改 test_pipeline 回傳這些物件
run_batch_smoke_tests(buffer, policy, critic)
