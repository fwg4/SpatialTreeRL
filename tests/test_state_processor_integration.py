# tests/test_state_processor_integration.py
import pytest
import torch
import numpy as np

from env import MiniMetroEnv
from processor import StateProcessor


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def env():
    return MiniMetroEnv(dt_ms=16)


@pytest.fixture
def processor(device):
    return StateProcessor(
        device=device,
        max_stations=20,
        max_shapes=7
    )


def test_processor_with_real_env(env, processor):
    """
    Integration Test

    驗證:
    MiniMetroEnv -> observation -> StateProcessor

    確保:
    - feature shape 正確
    - active mask 正確
    - position normalization 正確
    - one-hot 正確
    - topk distance 正確
    """

    obs = env.reset(seed=42)

    # --------------------------------------------------
    # 真實環境 observation
    # --------------------------------------------------

    features, active_mask = processor.process(obs)

    station_positions = obs["arrays"]["station_positions"]
    station_shapes = obs["arrays"]["station_shape_types"]

    N = station_positions.shape[0]

    # --------------------------------------------------
    # Shape validation
    # --------------------------------------------------

    assert features.shape == (20, 15)
    assert active_mask.shape == (20,)

    # --------------------------------------------------
    # Active mask validation
    # --------------------------------------------------

    for i in range(N):
        assert active_mask[i].item() is True

    for i in range(N, 20):
        assert active_mask[i].item() is False

    # --------------------------------------------------
    # Padding area should be zero
    # --------------------------------------------------

    assert torch.all(features[N:] == 0)

    # --------------------------------------------------
    # Position normalization validation
    # --------------------------------------------------

    scale_x = 1920.0
    scale_y = 1080.0
    eps = 1e-5

    for i in range(N):

        feat = features[i].cpu()

        expected_x = station_positions[i][0] / scale_x
        expected_y = station_positions[i][1] / scale_y

        assert abs(feat[0].item() - expected_x) < eps
        assert abs(feat[1].item() - expected_y) < eps

    # --------------------------------------------------
    # One-hot validation
    # --------------------------------------------------

    for i in range(N):

        feat = features[i].cpu()

        shape_idx = station_shapes[i]

        for cls in range(processor.max_shapes):

            expected = 1.0 if cls == shape_idx else 0.0

            assert feat[8 + cls].item() == expected

    # --------------------------------------------------
    # Distance feature validation
    # --------------------------------------------------

    normalized_positions = np.stack([
        [
            p[0] / scale_x,
            p[1] / scale_y
        ]
        for p in station_positions
    ])

    for i in range(N):

        feat = features[i].cpu()

        dists = []

        for j in range(N):

            if i == j:
                continue

            d = np.linalg.norm(
                normalized_positions[i] - normalized_positions[j]
            )

            dists.append(d)

        dists_sorted = sorted(dists)

        # 至少有 2 個 station 才能驗證
        if len(dists_sorted) >= 2:

            expected_top1 = dists_sorted[0]
            expected_top2 = dists_sorted[1]

            expected_far1 = dists_sorted[-1]
            expected_far2 = dists_sorted[-2]

            assert abs(feat[2].item() - expected_top1) < eps
            assert abs(feat[3].item() - expected_top2) < eps

            assert abs(feat[4].item() - expected_far1) < eps
            assert abs(feat[5].item() - expected_far2) < eps


def test_processor_after_environment_steps(env, processor):
    """
    驗證 environment 動態更新後:
    StateProcessor 仍能正常運作
    """

    obs = env.reset(seed=1)

    action = {"type": "noop"}

    # 推進遊戲時間
    for _ in range(50):
        obs, reward, done, info = env.step(action)

    features, active_mask = processor.process(obs)

    N = obs["arrays"]["station_positions"].shape[0]

    # active mask 應正確
    for i in range(N):
        assert active_mask[i].item() is True

    # feature 不應包含 nan / inf
    assert not torch.isnan(features).any()
    assert not torch.isinf(features).any()

    # 所有有效 station feature 不應全為 0
    assert torch.any(features[:N] != 0)


def test_processor_empty_case(processor):
    """
    驗證空 observation
    """

    obs = {
        "arrays": {
            "station_positions": np.zeros((0, 2), dtype=np.float32),
            "station_shape_types": np.zeros((0,), dtype=np.int64)
        }
    }

    features, active_mask = processor.process(obs)

    assert torch.all(features == 0)
    assert torch.all(active_mask == False)