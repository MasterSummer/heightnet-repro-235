from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample = None
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=False)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = F.relu(out + identity, inplace=False)
        return out




class GeometryBranch(nn.Module):
    """Explicit geometric feature MLP for frame-scale person cues."""

    def __init__(self, geo_feat_dim: int = 12, feat_ch: int = 64, hidden_dim: int = 32) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(geo_feat_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, feat_ch),
        )

    def forward(self, geo_feat: torch.Tensor) -> torch.Tensor:
        return self.mlp(geo_feat)


class MultiFeatureFusion(nn.Module):
    """Fuse visual and geometry features while preserving feature width."""

    def __init__(self, feat_ch: int, fusion_type: str = "add") -> None:
        super().__init__()
        self.feat_ch = int(feat_ch)
        self.fusion_type = str(fusion_type).lower()

        if self.fusion_type == "add":
            return
        if self.fusion_type == "concat":
            self.proj = nn.Linear(self.feat_ch * 2, self.feat_ch)
            return
        if self.fusion_type == "gated":
            self.gate = nn.Linear(self.feat_ch * 2, self.feat_ch)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, 0.0)
            return
        if self.fusion_type == "bilinear":
            self.bilinear = nn.Bilinear(self.feat_ch, self.feat_ch, self.feat_ch)
            return
        raise ValueError(f"unsupported fusion_type: {fusion_type}")

    def forward(self, visual_feat: torch.Tensor, geo_feat: torch.Tensor) -> torch.Tensor:
        if visual_feat.shape != geo_feat.shape:
            raise ValueError(
                f"fusion inputs must have identical shape, got {tuple(visual_feat.shape)} and {tuple(geo_feat.shape)}"
            )
        if visual_feat.ndim != 2 or visual_feat.shape[1] != self.feat_ch:
            raise ValueError(
                f"fusion inputs must be [B, {self.feat_ch}], got {tuple(visual_feat.shape)}"
            )

        if self.fusion_type == "add":
            return visual_feat + geo_feat

        joint_feat = torch.cat([visual_feat, geo_feat], dim=1)
        if self.fusion_type == "concat":
            return self.proj(joint_feat)
        if self.fusion_type == "gated":
            gate = torch.sigmoid(self.gate(joint_feat))
            return gate * visual_feat + (1.0 - gate) * geo_feat
        return self.bilinear(visual_feat, geo_feat)


def extract_geo_features(
    height: torch.Tensor,
    mask: torch.Tensor,
    bg_depth: torch.Tensor | None = None,
    person_bbox: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extract frame-scale geometric features from height map, mask, and bbox."""
    b = height.shape[0]
    h, w = height.shape[-2], height.shape[-1]
    m = (mask > 0.5).float()

    rows = []

    for i in range(b):
        mi = m[i, 0]
        hi = height[i, 0]
        nonzero_rows = mi.sum(dim=1) > 0
        if nonzero_rows.sum() == 0:
            rows.append(torch.zeros(12, dtype=torch.float32, device=height.device))
            continue

        row_indices = torch.where(nonzero_rows)[0]
        top_row = row_indices[0].item()
        bot_row = row_indices[-1].item()
        span = bot_row - top_row + 1
        nonzero_cols = mi.sum(dim=0) > 0
        col_indices = torch.where(nonzero_cols)[0]
        left_col = col_indices[0].item()
        right_col = col_indices[-1].item()
        col_span = right_col - left_col + 1

        head_end = top_row + max(1, int(span * 0.2))
        head_mask = mi[top_row:head_end, :]
        head_vals = hi[top_row:head_end, :][head_mask > 0.5]
        head_med = head_vals.median() if head_vals.numel() > 0 else hi.new_tensor(0.0)

        foot_start = bot_row - max(1, int(span * 0.2)) + 1
        foot_mask = mi[foot_start:bot_row + 1, :]
        foot_vals = hi[foot_start:bot_row + 1, :][foot_mask > 0.5]
        foot_med = foot_vals.median() if foot_vals.numel() > 0 else hi.new_tensor(0.0)

        all_vals = hi[mi > 0.5]
        if all_vals.numel() > 1:
            h_mean = all_vals.mean()
            h_std = all_vals.std()
            h_p90 = all_vals.quantile(0.9)
            h_p10 = all_vals.quantile(0.1)
        else:
            h_mean = h_std = h_p90 = h_p10 = hi.new_tensor(0.0)

        if bg_depth is not None:
            bgi = bg_depth[i, 0]
            bg_vals = bgi[mi <= 0.5]
            if bg_vals.numel() > 0 and all_vals.numel() > 0:
                diff = h_mean - bg_vals.mean()
            else:
                diff = hi.new_tensor(0.0)
        else:
            diff = hi.new_tensor(0.0)

        mask_area_ratio = mi.sum() / float(h * w)
        mask_h_ratio = hi.new_tensor(float(span) / float(h))
        mask_w_ratio = hi.new_tensor(float(col_span) / float(w))
        y_bottom_ratio = hi.new_tensor(float(bot_row + 1) / float(h))

        if person_bbox is not None:
            x1, y1, x2, y2 = person_bbox[i].to(device=height.device, dtype=torch.float32)
            bbox_h_ratio = torch.clamp((y2 - y1) / float(h), min=0.0)
            bbox_w_ratio = torch.clamp((x2 - x1) / float(w), min=0.0)
            bbox_area_ratio = bbox_h_ratio * bbox_w_ratio
        else:
            bbox_h_ratio = mask_h_ratio
            bbox_w_ratio = mask_w_ratio
            bbox_area_ratio = mask_area_ratio

        rows.append(torch.stack([
            foot_med,
            head_med,
            h_mean,
            h_std,
            h_p90,
            h_p10,
            diff,
            mask_h_ratio,
            mask_w_ratio,
            mask_area_ratio,
            y_bottom_ratio,
            bbox_area_ratio,
        ]))

    return torch.stack(rows, dim=0)

class PairComparatorHead(nn.Module):
    def __init__(
        self,
        feat_ch: int = 16,
        comparator_type: str = "conv",
        comparator_layers: int = 2,
        comparator_num_heads: int = 4,
        comparator_patch_size: int = 16,
        histogram_min: float = 0.0,
        histogram_max: float = 3.0,
        compare_type: str = "concat",
    ) -> None:
        super().__init__()
        self.feat_ch = int(feat_ch)
        self.comparator_type = str(comparator_type).lower()
        self.comparator_layers = int(comparator_layers)
        self.comparator_num_heads = int(comparator_num_heads)
        self.comparator_patch_size = int(comparator_patch_size)
        self.compare_type = str(compare_type).lower()

        if self.comparator_type == "conv":
            self.embed = self._build_conv_encoder()
        elif self.comparator_type == "resnet":
            self.embed = self._build_resnet_encoder()
        elif self.comparator_type == "resnet34":
            self.embed = self._build_resnet34_encoder()
        elif self.comparator_type == "vit":
            self.patch_embed = nn.Conv2d(
                1,
                self.feat_ch,
                kernel_size=self.comparator_patch_size,
                stride=self.comparator_patch_size,
                bias=False,
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.feat_ch,
                nhead=max(1, self.comparator_num_heads),
                dim_feedforward=self.feat_ch * 4,
                dropout=0.0,
                batch_first=True,
                activation="gelu",
            )
            self.vit_encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, self.comparator_layers))
            self.vit_norm = nn.LayerNorm(self.feat_ch)
        elif self.comparator_type == "histogram":
            centers = torch.linspace(float(histogram_min), float(histogram_max), self.feat_ch)
            sigma = (float(histogram_max) - float(histogram_min)) / max(self.feat_ch - 1, 1)
            self.register_buffer("hist_centers", centers)
            self.register_buffer("hist_sigma", torch.tensor(sigma, dtype=torch.float32))
            self.hist_mlp = nn.Sequential(
                nn.Linear(self.feat_ch, self.feat_ch * 2),
                nn.ReLU(inplace=False),
                nn.Linear(self.feat_ch * 2, self.feat_ch),
            )
        elif self.comparator_type == "resnet_stats":
            self.embed = self._build_resnet_encoder()
            n_stats = 5
            self.stats_proj = nn.Linear(n_stats, self.feat_ch)
        elif self.comparator_type == "histogram_stats":
            centers = torch.linspace(float(histogram_min), float(histogram_max), self.feat_ch)
            sigma = (float(histogram_max) - float(histogram_min)) / max(self.feat_ch - 1, 1)
            self.register_buffer("hist_centers", centers)
            self.register_buffer("hist_sigma", torch.tensor(sigma, dtype=torch.float32))
            n_stats = 5
            self.hist_mlp = nn.Sequential(
                nn.Linear(self.feat_ch + n_stats, self.feat_ch * 2),
                nn.ReLU(inplace=False),
                nn.Linear(self.feat_ch * 2, self.feat_ch),
            )
        else:
            raise ValueError(f"unsupported comparator_type: {self.comparator_type}")

        if self.compare_type in {"score", "antisym"}:
            hidden = max(self.feat_ch * 2, 32)
            self.score_mlp = nn.Sequential(
                nn.Linear(self.feat_ch, hidden),
                nn.ReLU(inplace=False),
                nn.Linear(hidden, hidden),
                nn.ReLU(inplace=False),
                nn.Linear(hidden, 1),
            )
        elif self.compare_type == "xattn":
            n_heads = max(1, min(self.comparator_num_heads, self.feat_ch // 4))
            self.xattn = nn.MultiheadAttention(self.feat_ch, num_heads=n_heads, batch_first=True)
            self.xattn_norm = nn.LayerNorm(self.feat_ch)
            self.cls = nn.Sequential(
                nn.Linear(self.feat_ch * 4, self.feat_ch),
                nn.ReLU(inplace=False),
                nn.Linear(self.feat_ch, 1),
            )
        else:
            self.cls = nn.Sequential(
                nn.Linear(self.feat_ch * 3, self.feat_ch),
                nn.ReLU(inplace=False),
                nn.Linear(self.feat_ch, 1),
            )

    def _build_conv_encoder(self) -> nn.Module:
        layers: list[nn.Module] = []
        in_ch = 1
        for _ in range(max(1, self.comparator_layers)):
            layers.extend(
                [
                    nn.Conv2d(in_ch, self.feat_ch, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(self.feat_ch),
                    nn.ReLU(inplace=False),
                ]
            )
            in_ch = self.feat_ch
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        return nn.Sequential(*layers)

    def _build_resnet_encoder(self) -> nn.Module:
        layers: list[nn.Module] = [
            nn.Conv2d(1, self.feat_ch, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.feat_ch),
            nn.ReLU(inplace=False),
        ]
        for idx in range(max(1, self.comparator_layers)):
            layers.append(ResidualBlock(self.feat_ch, stride=2 if idx == 0 else 1))
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        return nn.Sequential(*layers)

    def _build_resnet34_encoder(self) -> nn.Module:
        backbone = models.resnet34(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.fc = nn.Identity()
        return nn.Sequential(
            backbone,
            nn.Linear(512, self.feat_ch),
        )

    def _extract_stats(self, fg_height: torch.Tensor) -> torch.Tensor:
        b = fg_height.shape[0]
        vals = fg_height.view(b, -1)
        mask = vals > 1e-4
        stats = []
        for i in range(b):
            v = vals[i][mask[i]]
            if v.numel() < 2:
                stats.append(torch.zeros(5, device=fg_height.device))
            else:
                stats.append(torch.stack([
                    v.mean(),
                    v.std(),
                    v.median(),
                    v.quantile(0.9),
                    v.quantile(0.1),
                ]))
        return torch.stack(stats, dim=0)

    def encode(self, fg_height: torch.Tensor) -> torch.Tensor:
        if self.comparator_type == "vit":
            x = self.patch_embed(fg_height)
            b, c, h, w = x.shape
            if h == 0 or w == 0:
                raise ValueError(
                    f"vit comparator patch size {self.comparator_patch_size} is too large for feature map {tuple(fg_height.shape)}"
                )
            tokens = x.flatten(2).transpose(1, 2)
            tokens = self.vit_encoder(tokens)
            tokens = self.vit_norm(tokens)
            return tokens.mean(dim=1)

        if self.comparator_type == "histogram":
            b = fg_height.shape[0]
            vals = fg_height.view(b, -1)
            mask = vals > 1e-4
            centers = self.hist_centers.view(1, 1, -1)
            sigma = self.hist_sigma
            vals_exp = vals.unsqueeze(2)
            weights = torch.exp(-0.5 * ((vals_exp - centers) / sigma) ** 2)
            mask_exp = mask.unsqueeze(2).float()
            hist = (weights * mask_exp).sum(dim=1)
            denom = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
            hist = hist / denom
            return self.hist_mlp(hist)

        if self.comparator_type == "histogram_stats":
            b = fg_height.shape[0]
            vals = fg_height.view(b, -1)
            mask = vals > 1e-4
            centers = self.hist_centers.view(1, 1, -1)
            sigma = self.hist_sigma
            vals_exp = vals.unsqueeze(2)
            weights = torch.exp(-0.5 * ((vals_exp - centers) / sigma) ** 2)
            mask_exp = mask.unsqueeze(2).float()
            hist = (weights * mask_exp).sum(dim=1)
            denom = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
            hist = hist / denom
            stats = self._extract_stats(fg_height)
            return self.hist_mlp(torch.cat([hist, stats], dim=1))

        if self.comparator_type == "resnet_stats":
            feat = self.embed(fg_height).flatten(1)
            stats = self._extract_stats(fg_height)
            return feat + self.stats_proj(stats)

        feat = self.embed(fg_height)
        return feat.flatten(1)

    def compare_encoded(self, fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
        if self.compare_type in {"score", "antisym"}:
            return (self.score_mlp(fa) - self.score_mlp(fb)).squeeze(1)
        if self.compare_type == "xattn":
            a = fa.unsqueeze(1)
            b = fb.unsqueeze(1)
            ab = torch.cat([a, b], dim=1)
            attn_out, _ = self.xattn(ab, ab, ab)
            attn_out = self.xattn_norm(ab + attn_out)
            fa_attn = attn_out[:, 0]
            fb_attn = attn_out[:, 1]
            x = torch.cat([fa_attn, fb_attn, fa_attn - fb_attn, fa * fb], dim=1)
        else:
            x = torch.cat([fa, fb, fa - fb], dim=1)
        return self.cls(x).squeeze(1)

    def forward(self, fg_a: torch.Tensor, fg_b: torch.Tensor) -> torch.Tensor:
        fa = self.encode(fg_a)
        fb = self.encode(fg_b)
        return self.compare_encoded(fa, fb)


class DerivedHeightRanker(nn.Module):
    """Pairwise ranker that consumes derived height maps directly."""

    def __init__(
        self,
        comparator_channels: int = 16,
        comparator_type: str = "conv",
        comparator_layers: int = 2,
        comparator_num_heads: int = 4,
        comparator_patch_size: int = 16,
        person_region_mode: str = "mask",
        bbox_expand_ratio: float = 0.0,
        histogram_min: float = 0.0,
        histogram_max: float = 3.0,
        compare_type: str = "concat",
        use_geometry_branch: bool = False,
        geo_feat_dim: int = 12,
        geo_hidden_dim: int = 32,
        fusion_type: str = "add",
    ) -> None:
        super().__init__()
        self.person_region_mode = str(person_region_mode).lower()
        self.bbox_expand_ratio = float(bbox_expand_ratio)
        self.use_geometry_branch = use_geometry_branch
        self.feature_fusion = MultiFeatureFusion(comparator_channels, fusion_type=fusion_type)
        if self.person_region_mode not in {"mask", "bbox"}:
            raise ValueError(f"unsupported person_region_mode: {self.person_region_mode}")

        self.pair_head = PairComparatorHead(
            feat_ch=comparator_channels,
            comparator_type=comparator_type,
            comparator_layers=comparator_layers,
            comparator_num_heads=comparator_num_heads,
            comparator_patch_size=comparator_patch_size,
            histogram_min=histogram_min,
            histogram_max=histogram_max,
            compare_type=compare_type,
        )

        if self.use_geometry_branch:
            self.geo_branch = GeometryBranch(
                geo_feat_dim=geo_feat_dim,
                feat_ch=comparator_channels,
                hidden_dim=geo_hidden_dim,
            )

    def build_person_region(
        self,
        height: torch.Tensor,
        mask: torch.Tensor,
        bbox: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.person_region_mode == "mask":
            return height * (mask > 0.5).float()

        if bbox is None:
            raise ValueError("bbox is required when person_region_mode='bbox'")
        if bbox.ndim != 2 or bbox.shape[1] != 4:
            raise ValueError(f"bbox should be [B,4], got {tuple(bbox.shape)}")

        b, _, h, w = height.shape
        fg = torch.zeros_like(height)
        for i in range(b):
            x1, y1, x2, y2 = [float(v) for v in bbox[i].tolist()]
            if x2 <= x1 or y2 <= y1:
                continue
            bw = x2 - x1
            bh = y2 - y1
            dx = bw * self.bbox_expand_ratio
            dy = bh * self.bbox_expand_ratio
            x1i = max(0, int(math.floor(x1 - dx)))
            y1i = max(0, int(math.floor(y1 - dy)))
            x2i = min(w, int(math.ceil(x2 + dx)))
            y2i = min(h, int(math.ceil(y2 + dy)))
            if x2i <= x1i or y2i <= y1i:
                continue
            fg[i, :, y1i:y2i, x1i:x2i] = height[i, :, y1i:y2i, x1i:x2i]
        return fg

    @staticmethod
    def _normalize_geometry_inputs(
        bbox: torch.Tensor | None,
        bg_depth: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if bbox is None or bg_depth is None:
            return bbox, bg_depth
        if bbox.ndim == 4 and bg_depth.ndim == 2 and bg_depth.shape[1] == 4:
            return bg_depth, bbox
        return bbox, bg_depth

    def encode_person(self, height: torch.Tensor, mask: torch.Tensor, bbox: torch.Tensor | None = None, bg_depth: torch.Tensor | None = None) -> torch.Tensor:
        bbox, bg_depth = self._normalize_geometry_inputs(bbox, bg_depth)
        fg = self.build_person_region(height, mask, bbox)
        visual_feat = self.pair_head.encode(fg)
        if self.use_geometry_branch:
            geo_feat = extract_geo_features(height, mask, bg_depth, bbox)
            geo_emb = self.geo_branch(geo_feat)
            return self.feature_fusion(visual_feat, geo_emb)
        return visual_feat

    def compare_pair(
        self,
        height_a: torch.Tensor,
        mask_a: torch.Tensor,
        height_b: torch.Tensor,
        mask_b: torch.Tensor,
        bbox_a: torch.Tensor | None = None,
        bbox_b: torch.Tensor | None = None,
        bg_depth_a: torch.Tensor | None = None,
        bg_depth_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fa = self.encode_person(height_a, mask_a, bbox_a, bg_depth_a)
        fb = self.encode_person(height_b, mask_b, bbox_b, bg_depth_b)
        return self.compare_encoded(fa, fb)

    def compare_encoded(self, fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
        return self.pair_head.compare_encoded(fa, fb)

    def forward(
        self,
        pair_inputs: dict | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> dict:
        out = {"pred_height_map": None, "pair_logit": None}
        if pair_inputs is None:
            return out
        if isinstance(pair_inputs, dict):
            source = pair_inputs["height_map"]
            idx_i = pair_inputs["idx_i"]
            idx_j = pair_inputs["idx_j"]
            person_mask = pair_inputs["person_mask"]
            person_bbox = pair_inputs.get("person_bbox")
            h_a = source.index_select(0, idx_i)
            m_a = person_mask.index_select(0, idx_i)
            h_b = source.index_select(0, idx_j)
            m_b = person_mask.index_select(0, idx_j)
            bg_depth = pair_inputs.get("bg_depth")
            b_a = person_bbox.index_select(0, idx_i) if person_bbox is not None else None
            b_b = person_bbox.index_select(0, idx_j) if person_bbox is not None else None
            bg_a = bg_depth.index_select(0, idx_i) if bg_depth is not None else None
            bg_b = bg_depth.index_select(0, idx_j) if bg_depth is not None else None
        elif len(pair_inputs) == 4:
            h_a, m_a, h_b, m_b = pair_inputs
            b_a = None
            b_b = None
            bg_a = None
            bg_b = None
        else:
            raise ValueError("unsupported pair_inputs format")

        out["pair_logit"] = self.compare_pair(h_a, m_a, h_b, m_b, b_a, b_b, bg_a, bg_b)
        return out


class HeightNetTiny(nn.Module):
    """Encoder-decoder for height map + pairwise comparator head."""

    def __init__(
        self,
        base_channels: int = 32,
        comparator_channels: int = 16,
        comparator_type: str = "conv",
        comparator_layers: int = 2,
        comparator_num_heads: int = 4,
        comparator_patch_size: int = 16,
        person_region_mode: str = "mask",
        bbox_expand_ratio: float = 0.0,
        use_geometry_branch: bool = False,
        geo_feat_dim: int = 12,
        geo_hidden_dim: int = 32,
        fusion_type: str = "add",
    ) -> None:
        super().__init__()
        b = base_channels
        self.person_region_mode = str(person_region_mode).lower()
        self.bbox_expand_ratio = float(bbox_expand_ratio)
        self.use_geometry_branch = bool(use_geometry_branch)
        self.feature_fusion = MultiFeatureFusion(comparator_channels, fusion_type=fusion_type)
        if self.person_region_mode not in {"mask", "bbox"}:
            raise ValueError(f"unsupported person_region_mode: {self.person_region_mode}")

        self.enc1 = ConvBlock(3, b)
        self.enc2 = ConvBlock(b, b * 2)
        self.enc3 = ConvBlock(b * 2, b * 4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(b * 4, b * 8)

        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(b * 8, b * 4)

        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(b * 4, b * 2)

        self.up1 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(b * 2, b)

        self.height_head = nn.Conv2d(b, 1, kernel_size=1)
        self.pair_head = PairComparatorHead(
            feat_ch=comparator_channels,
            comparator_type=comparator_type,
            comparator_layers=comparator_layers,
            comparator_num_heads=comparator_num_heads,
            comparator_patch_size=comparator_patch_size,
        )
        if self.use_geometry_branch:
            self.geo_branch = GeometryBranch(
                geo_feat_dim=geo_feat_dim,
                feat_ch=comparator_channels,
                hidden_dim=geo_hidden_dim,
            )

    def build_person_region(
        self,
        height: torch.Tensor,
        mask: torch.Tensor,
        bbox: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.person_region_mode == "mask":
            return height * (mask > 0.5).float()

        if bbox is None:
            raise ValueError("bbox is required when person_region_mode='bbox'")
        if bbox.ndim != 2 or bbox.shape[1] != 4:
            raise ValueError(f"bbox should be [B,4], got {tuple(bbox.shape)}")

        b, _, h, w = height.shape
        fg = torch.zeros_like(height)
        for i in range(b):
            x1, y1, x2, y2 = [float(v) for v in bbox[i].tolist()]
            if x2 <= x1 or y2 <= y1:
                continue
            bw = x2 - x1
            bh = y2 - y1
            dx = bw * self.bbox_expand_ratio
            dy = bh * self.bbox_expand_ratio
            x1i = max(0, int(math.floor(x1 - dx)))
            y1i = max(0, int(math.floor(y1 - dy)))
            x2i = min(w, int(math.ceil(x2 + dx)))
            y2i = min(h, int(math.ceil(y2 + dy)))
            if x2i <= x1i or y2i <= y1i:
                continue
            fg[i, :, y1i:y2i, x1i:x2i] = height[i, :, y1i:y2i, x1i:x2i]
        return fg

    def predict_height(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bt = self.bottleneck(self.pool(e3))

        d3 = self.up3(bt)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return F.softplus(self.height_head(d1))

    def compare_pair(
        self,
        height_a: torch.Tensor,
        mask_a: torch.Tensor,
        height_b: torch.Tensor,
        mask_b: torch.Tensor,
        bbox_a: torch.Tensor | None = None,
        bbox_b: torch.Tensor | None = None,
        bg_depth_a: torch.Tensor | None = None,
        bg_depth_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fa = self.encode_person(height_a, mask_a, bbox_a, bg_depth_a)
        fb = self.encode_person(height_b, mask_b, bbox_b, bg_depth_b)
        return self.compare_encoded(fa, fb)

    def encode_person(self, height: torch.Tensor, mask: torch.Tensor, bbox: torch.Tensor | None = None, bg_depth: torch.Tensor | None = None) -> torch.Tensor:
        bbox, bg_depth = DerivedHeightRanker._normalize_geometry_inputs(bbox, bg_depth)
        fg = self.build_person_region(height, mask, bbox)
        visual_feat = self.pair_head.encode(fg)
        if self.use_geometry_branch:
            geo_feat = extract_geo_features(height, mask, bg_depth, bbox)
            geo_emb = self.geo_branch(geo_feat)
            return self.feature_fusion(visual_feat, geo_emb)
        return visual_feat

    def compare_encoded(self, fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
        return self.pair_head.compare_encoded(fa, fb)

    def forward(
        self,
        image: torch.Tensor | None = None,
        pair_inputs: dict | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> dict:
        pred_height_map = self.predict_height(image) if image is not None else None
        out = {"pred_height_map": pred_height_map, "pair_logit": None}
        if pair_inputs is not None:
            if isinstance(pair_inputs, dict):
                if pred_height_map is None:
                    raise ValueError("pred_height_map is required when pair_inputs is index-based")
                idx_i = pair_inputs["idx_i"]
                idx_j = pair_inputs["idx_j"]
                person_mask = pair_inputs["person_mask"]
                person_bbox = pair_inputs.get("person_bbox")
                source = pred_height_map[: person_mask.shape[0]]
                h_a = source.index_select(0, idx_i)
                m_a = person_mask.index_select(0, idx_i)
                h_b = source.index_select(0, idx_j)
                m_b = person_mask.index_select(0, idx_j)
                b_a = person_bbox.index_select(0, idx_i) if person_bbox is not None else None
                b_b = person_bbox.index_select(0, idx_j) if person_bbox is not None else None
            elif len(pair_inputs) == 4:
                h_a, m_a, h_b, m_b = pair_inputs
                b_a = None
                b_b = None
            else:
                raise ValueError("unsupported pair_inputs format")
            out["pair_logit"] = self.compare_pair(h_a, m_a, h_b, m_b, b_a, b_b)
        return out
