import pytest
import torch

@pytest.fixture(scope="session")
def device():
    """全域定義的 device fixture，自動判斷是否使用 GPU"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")