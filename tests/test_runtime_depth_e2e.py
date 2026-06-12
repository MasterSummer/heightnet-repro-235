import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from src.heightnet.config import load_config
from src.heightnet.runtime_depth import RuntimeDepthEstimator


class _TinyDepthModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pretrained = torch.nn.Conv2d(3, 3, kernel_size=1)
        self.depth_head = torch.nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.depth_head(self.pretrained(x)).squeeze(1)


class RuntimeDepthE2ETests(unittest.TestCase):
    def test_set_trainable_depth_head_freezes_backbone_only(self):
        estimator = RuntimeDepthEstimator.__new__(RuntimeDepthEstimator)
        estimator.model = _TinyDepthModel()

        estimator.set_trainable("depth_head")

        self.assertFalse(any(p.requires_grad for p in estimator.model.pretrained.parameters()))
        self.assertTrue(all(p.requires_grad for p in estimator.model.depth_head.parameters()))

    def test_forward_batch_keeps_gradient_to_depth_head(self):
        estimator = RuntimeDepthEstimator.__new__(RuntimeDepthEstimator)
        estimator.model = _TinyDepthModel()
        estimator.input_size = 28
        estimator.set_trainable("depth_head")

        images = torch.full((2, 3, 16, 20), 127, dtype=torch.uint8)
        depth = estimator.forward_batch(images)
        loss = depth.mean()
        loss.backward()

        self.assertEqual(tuple(depth.shape), (2, 1, 16, 20))
        self.assertIsNotNone(estimator.model.depth_head.weight.grad)
        self.assertGreater(float(estimator.model.depth_head.weight.grad.abs().sum()), 0.0)

    def test_runtime_depth_config_loads_e2e_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.yaml"
            cfg_path.write_text(
                yaml.safe_dump(
                    {
                        "seed": 42,
                        "device": "cpu",
                        "paths": {"output_dir": "runs/x"},
                        "train": {
                            "epochs": 1,
                            "batch_size": 1,
                            "num_workers": 0,
                            "lr": 1e-4,
                            "weight_decay": 1e-4,
                            "grad_clip_norm": 1.0,
                            "use_amp": False,
                        },
                        "loss": {
                            "lambda_rmse": 1.0,
                            "lambda_rank": 1.0,
                            "lambda_cons": 0.0,
                            "eps": 1e-6,
                            "min_valid_pixels": 1,
                            "min_valid_ratio": 0.0,
                            "pairwise_json": "",
                        },
                        "data": {"image_size": [32, 32], "normalize_rgb": True, "use_pair_consistency": False},
                        "model": {"name": "derived_height_ranker", "base_channels": 8},
                        "eval": {"save_visualizations": False, "vis_limit": 0},
                        "runtime_seg": {
                            "enabled": True,
                            "model_path": "",
                            "conf": 0.25,
                            "iou": 0.7,
                            "imgsz": 640,
                            "strict_native": True,
                        },
                        "runtime_depth": {
                            "enabled": True,
                            "depthanything_root": "",
                            "encoder": "vitl",
                            "checkpoint": "",
                            "input_size": 518,
                            "trainable": True,
                            "train_parts": "depth_head",
                            "depth_lr": 1e-5,
                            "use_depth_cache_during_train": False,
                        },
                    }
                ),
                encoding="utf-8",
            )

            cfg = load_config(str(cfg_path))

        self.assertTrue(cfg.runtime_depth.trainable)
        self.assertEqual(cfg.runtime_depth.train_parts, "depth_head")
        self.assertAlmostEqual(cfg.runtime_depth.depth_lr, 1e-5)
        self.assertFalse(cfg.runtime_depth.use_depth_cache_during_train)


if __name__ == "__main__":
    unittest.main()
