from __future__ import annotations

import contextlib
import io
import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class RuntimeDepthConfig:
    enabled: bool
    depthanything_root: str
    encoder: str
    checkpoint: str
    input_size: int
    assume_inverse: bool = False
    use_ground_anchor: bool = True
    trainable: bool = False
    train_parts: str = "depth_head"
    depth_lr: float = 1e-5
    use_depth_cache_during_train: bool = True


class RuntimeDepthEstimator:
    def __init__(
        self,
        depthanything_root: str,
        encoder: str,
        checkpoint: str,
        input_size: int = 518,
    ) -> None:
        if not depthanything_root or not os.path.exists(depthanything_root):
            raise FileNotFoundError(f"depthanything root not found: {depthanything_root}")
        if not checkpoint or not os.path.exists(checkpoint):
            raise FileNotFoundError(f"depth checkpoint not found: {checkpoint}")

        self.depthanything_root = depthanything_root
        self.encoder = encoder
        self.checkpoint = checkpoint
        self.input_size = int(input_size)

        if depthanything_root not in sys.path:
            sys.path.insert(0, depthanything_root)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from depth_anything_v2.dpt import DepthAnythingV2

        model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
            "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
        }
        if encoder not in model_configs:
            raise ValueError(f"unsupported encoder: {encoder}")

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.model = DepthAnythingV2(**model_configs[encoder])
        state_dict = _torch_load_compat(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)

    def to(self, device: torch.device) -> "RuntimeDepthEstimator":
        self.model = self.model.to(device).eval()
        return self

    def set_trainable(self, train_parts: str = "depth_head") -> "RuntimeDepthEstimator":
        train_parts = str(train_parts).lower()
        if train_parts != "depth_head":
            raise ValueError(f"unsupported train_parts: {train_parts}")
        for param in self.model.parameters():
            param.requires_grad_(False)
        depth_head = self._model_ref().depth_head
        for param in depth_head.parameters():
            param.requires_grad_(True)
        return self

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def state_dict(self) -> dict:
        return self._model_ref().state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self._model_ref().load_state_dict(state_dict)

    def _model_ref(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def forward_batch(self, images_raw: torch.Tensor) -> torch.Tensor:
        """
        Differentiable DA2 batch forward for training.
        images_raw: uint8/float tensor [B,3,H,W], RGB.
        returns: depth tensor [B,1,H,W].
        """
        if images_raw.ndim != 4:
            raise ValueError(f"images_raw should be [B,3,H,W], got {tuple(images_raw.shape)}")

        _, _, h, w = images_raw.shape
        image = images_raw.to(next(self.model.parameters()).device)
        image = image.float()
        if float(image.detach().max()) > 2.0:
            image = image / 255.0
        resized_h, resized_w = _da2_resize_hw(h, w, self.input_size, multiple_of=14)
        image = F.interpolate(image, size=(resized_h, resized_w), mode="bicubic", align_corners=False)
        mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        image = (image - mean) / std

        depth = self.model(image)
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        if tuple(depth.shape[-2:]) != (h, w):
            depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        return depth

    @torch.no_grad()
    def infer_batch(self, images_raw: torch.Tensor) -> torch.Tensor:
        """
        images_raw: uint8 tensor [B,3,H,W], RGB.
        returns: depth tensor [B,1,H,W], float32.
        """
        if images_raw.ndim != 4:
            raise ValueError(f"images_raw should be [B,3,H,W], got {tuple(images_raw.shape)}")

        b, _, h, w = images_raw.shape
        outs = []
        for i in range(b):
            img = images_raw[i].permute(1, 2, 0).detach().cpu().numpy()
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            depth = self._model_ref().infer_image(img_bgr, self.input_size).astype(np.float32)
            if depth.shape != (h, w):
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            outs.append(torch.from_numpy(depth).unsqueeze(0))
        return torch.stack(outs, dim=0).to(images_raw.device)


def depth_to_height(
    depth: torch.Tensor,
    bg_depth: torch.Tensor,
    camera_height_m: torch.Tensor,
    eps: float = 1e-6,
    assume_inverse: bool = False,
    ground_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    depth/bg_depth: [B,1,H,W], camera_height_m: [B,1,1,1]
    ground_mask: optional [B,1,H,W] bool tensor indicating ground pixels for scale alignment.
      When provided, the foreground depth is scaled so that its ground-region median
      matches the background depth ground-region median, correcting for DA2 scale drift.
    returns (height, valid_mask)
    """
    depth_work = depth
    bg_work = bg_depth

    # Ground-plane scale alignment: align depth_work to bg_work scale per-sample.
    if ground_mask is not None:
        b = depth_work.shape[0]
        depth_aligned = depth_work.clone()
        for i in range(b):
            mask_i = ground_mask[i, 0].bool()  # [H,W]
            d_vals = depth_work[i, 0][mask_i]
            bg_vals = bg_work[i, 0][mask_i]
            if d_vals.numel() >= 4 and bg_vals.numel() >= 4:
                d_median = d_vals.median()
                bg_median = bg_vals.median()
                if d_median.abs() > eps:
                    scale = (bg_median / d_median).detach()
                    depth_aligned[i] = depth_work[i] * scale
        depth_work = depth_aligned

    if assume_inverse:
        depth_work = torch.where(
            torch.isfinite(depth_work),
            1.0 / torch.clamp(depth_work, min=eps),
            depth_work,
        )
        bg_work = torch.where(
            torch.isfinite(bg_work),
            1.0 / torch.clamp(bg_work, min=eps),
            bg_work,
        )

    valid = torch.isfinite(depth_work) & torch.isfinite(bg_work) & (bg_work.abs() > eps)
    height = torch.zeros_like(depth)
    height[valid] = camera_height_m.expand_as(depth)[valid] * (bg_work[valid] - depth_work[valid]) / bg_work[valid]
    upper = camera_height_m.expand_as(depth) * 3.0
    height = torch.clamp(height, min=0.0)
    height = torch.minimum(height, upper)
    return height, valid.float()


def _torch_load_compat(path: str, map_location: str | torch.device, weights_only: bool) -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _constrain_to_multiple_of(x: float, multiple_of: int, min_val: int) -> int:
    y = int(round(x / multiple_of) * multiple_of)
    if y < min_val:
        y = int(np.ceil(x / multiple_of) * multiple_of)
    return y


def _da2_resize_hw(h: int, w: int, input_size: int, multiple_of: int = 14) -> tuple[int, int]:
    scale_h = input_size / float(h)
    scale_w = input_size / float(w)
    if scale_w > scale_h:
        scale_h = scale_w
    else:
        scale_w = scale_h
    new_h = _constrain_to_multiple_of(scale_h * h, multiple_of, input_size)
    new_w = _constrain_to_multiple_of(scale_w * w, multiple_of, input_size)
    return new_h, new_w
