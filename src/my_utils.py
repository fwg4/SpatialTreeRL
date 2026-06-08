# src/my_utils.py
import torch
import numpy as np
import random


def format_time_ms(time_ms: float | int) -> str:
    """將毫秒轉換為最大單位的直觀字串 (內部封裝單位轉換)"""
    if time_ms < 1000:
        return f"{int(time_ms)}ms"

    seconds = time_ms / 1000.0
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, rem = divmod(int(seconds), 3600)
        m, _ = divmod(rem, 60)
        return f"{h}h {m}m"


def set_global_seed(seed=42):
    """鎖定全域隨機性，確保實驗起點一致"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
