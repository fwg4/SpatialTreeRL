import os
import torch
import pytest

def test_cuda_availability():
    """
    動態測試 CUDA 掛載狀態，依賴環境變數 EXPECT_GPU。
    未設定時預設為 0 (CPU 模式)。
    """
    # 讀取環境變數，將字串 "1" 轉換為布林值 True
    expect_gpu = os.environ.get("EXPECT_GPU", "0") == "1"
    actual_gpu = torch.cuda.is_available()

    # 核心斷言：實際的 GPU 狀態必須符合環境變數的預期
    assert actual_gpu == expect_gpu, (
        f"🚨 系統硬體狀態不符預期！\n"
        f"環境要求 (EXPECT_GPU): {expect_gpu}\n"
        f"實際偵測 (is_available): {actual_gpu}"
    )

    # 附帶印出當前設備資訊，方便在 pytest -s 時查看
    if actual_gpu:
        device_name = torch.cuda.get_device_name(0)
        print(f"\n✅ 測試通過：成功掛載 GPU ({device_name})")
    else:
        print("\n✅ 測試通過：當前正確運行於純 CPU 模式")