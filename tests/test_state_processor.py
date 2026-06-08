# tests/test_state_processor.py
import torch
import numpy as np
import pytest

from processor import StateProcessor


@pytest.fixture
def processor(device):
    return StateProcessor(device=device, max_stations=3, max_shapes=7)


def test_state_processor_features(processor):
    """
    測試:
    - position normalization
    - top1/top2
    - far1/far2
    - diff_min/diff_max
    - onehot
    - active mask
    """

    positions = np.array([
        [0.0, 0.0],        # station 0
        [300.0, 400.0],    # station 1
        [0.0, 300.0],      # station 2
    ], dtype=np.float32)

    shapes = np.array([
        0,  # RECT
        1,  # CIRCLE
        1,  # CIRCLE
    ], dtype=np.int64)

    obs = {
        "arrays": {
            "station_positions": positions,
            "station_shape_types": shapes
        }
    }

    features, active_mask = processor.process(obs)

    # --------------------------------------------------
    # 預期距離 (normalized)
    # --------------------------------------------------

    scale_x = 1920.0
    scale_y = 1080.0

    p0 = np.array([0.0 / scale_x, 0.0 / scale_y])
    p1 = np.array([300.0 / scale_x, 400.0 / scale_y])
    p2 = np.array([0.0 / scale_x, 300.0 / scale_y])

    d01 = np.linalg.norm(p0 - p1)
    d02 = np.linalg.norm(p0 - p2)
    d12 = np.linalg.norm(p1 - p2)

    eps = 1e-5

    # --------------------------------------------------
    # Station 0 驗證
    # --------------------------------------------------

    feat_0 = features[0].cpu()

    # pos
    assert abs(feat_0[0].item() - p0[0]) < eps
    assert abs(feat_0[1].item() - p0[1]) < eps

    # top1 = 最近距離 = d02
    assert abs(feat_0[2].item() - d02) < eps

    # top2 = 第二近 = d01
    assert abs(feat_0[3].item() - d01) < eps

    # far1 = 最遠 = d01
    assert abs(feat_0[4].item() - d01) < eps

    # far2 = 第二遠 = d02
    assert abs(feat_0[5].item() - d02) < eps

    # diff_min
    # station0 是 RECT
    # shape 不同的是 station1 / station2
    # 最近是 station2
    assert abs(feat_0[6].item() - d02) < eps

    # diff_max
    # 最遠是 station1
    assert abs(feat_0[7].item() - d01) < eps

    # one-hot
    assert feat_0[8 + 0].item() == 1.0
    assert feat_0[8 + 1].item() == 0.0

    # --------------------------------------------------
    # Station 1 驗證
    # --------------------------------------------------

    feat_1 = features[1].cpu()

    # top1 = 最近 = d12
    assert abs(feat_1[2].item() - d12) < eps

    # top2 = d01
    assert abs(feat_1[3].item() - d01) < eps

    # diff_min
    # station1 是 CIRCLE
    # 不同 shape 只有 station0
    assert abs(feat_1[6].item() - d01) < eps

    # diff_max
    assert abs(feat_1[7].item() - d01) < eps

    # one-hot
    assert feat_1[8 + 0].item() == 0.0
    assert feat_1[8 + 1].item() == 1.0

    # --------------------------------------------------
    # active mask
    # --------------------------------------------------

    assert active_mask[0].item() is True
    assert active_mask[1].item() is True
    assert active_mask[2].item() is True