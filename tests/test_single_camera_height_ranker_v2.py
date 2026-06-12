from __future__ import annotations

import numpy as np
import torch

from tools.train_single_camera_height_ranker_v2 import (
    SingleCameraHeightRankerV2,
    build_v2_sequence_features,
    pair_logits,
)


def test_v2_features_are_fixed_width_and_finite():
    bbox = np.array(
        [
            [0.20, 0.10, -0.10, 0.80, 0.45, 0.02, 0.90],
            [0.24, 0.11, -0.12, 0.82, 0.47, 0.026, 0.80],
        ],
        dtype=np.float32,
    )
    stats = np.array(
        [
            [1.1, 1.2, 1.4, 1.5, 1.8],
            [1.0, 1.3, 1.5, 1.6, 2.0],
        ],
        dtype=np.float32,
    )
    crops = np.zeros((2, 1, 128, 64), dtype=np.float32)
    crops[0, 0, 10:80, 20:40] = 1.0
    crops[1, 0, 12:82, 22:42] = 1.2

    feat = build_v2_sequence_features(bbox, stats, crops, "2d5_0")

    assert feat.ndim == 1
    assert feat.shape[0] > 9
    assert np.isfinite(feat).all()


def test_ranker_pair_logits_are_antisymmetric_for_single_frame_sequences():
    torch.manual_seed(0)
    model = SingleCameraHeightRankerV2(tabular_dim=32, variant="fusion_v2", max_frames=4)
    model.eval()
    tab = torch.randn(2, 32)
    crops = torch.randn(2, 1, 1, 128, 64)

    logits_ab = pair_logits(model, tab[:1], crops[:1], tab[1:], crops[1:])
    logits_ba = pair_logits(model, tab[1:], crops[1:], tab[:1], crops[:1])

    assert torch.allclose(logits_ab, -logits_ba, atol=1e-6)
