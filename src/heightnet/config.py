from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import yaml


@dataclass
class PathsConfig:
    train_manifest: str = ""
    val_manifest: str = ""
    test_manifest: str = ""
    train_video_root: str = ""
    val_video_root: str = ""
    test_video_root: str = ""
    bg_depth_root: str = ""
    output_dir: str = ""


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    num_workers: int
    lr: float
    weight_decay: float
    grad_clip_norm: float
    use_amp: bool
    use_tensorboard: bool = True
    log_interval: int = 20
    tb_num_images: int = 4


@dataclass
class LossConfig:
    lambda_rmse: float
    lambda_rank: float
    lambda_cons: float
    eps: float
    min_valid_pixels: int
    min_valid_ratio: float
    pairwise_json: str
    consistency_mode: str = "map"
    listwise_lambda: float = 0.0
    height_labels_json: str = ""


@dataclass
class DataConfig:
    image_size: List[int]
    normalize_rgb: bool
    use_pair_consistency: bool


@dataclass
class ModelConfig:
    name: str
    base_channels: int
    comparator_channels: int = 16
    comparator_type: str = "conv"
    comparator_layers: int = 2
    comparator_num_heads: int = 4
    comparator_patch_size: int = 16
    person_region_mode: str = "mask"
    bbox_expand_ratio: float = 0.25
    histogram_min: float = 0.0
    histogram_max: float = 3.0
    compare_type: str = "concat"
    use_geometry_branch: bool = False
    fusion_type: str = "add"
    geo_feat_dim: int = 12
    geo_hidden_dim: int = 32
    crop_and_resize_person: bool = True


@dataclass
class EvalConfig:
    save_visualizations: bool
    vis_limit: int
    video_feature_frames: int = 8
    save_train_gallery: bool = True
    frames_per_person_eval: int = 10
    frame_eval_seed: int = 42
    eval_batch_size: int = 16
    validate_every_epochs: int = 1


@dataclass
class RuntimeSegConfig:
    enabled: bool
    model_path: str
    conf: float
    iou: float
    imgsz: int
    strict_native: bool
    allow_inference_fallback: bool = True


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


@dataclass
class Config:
    seed: int
    device: str
    paths: PathsConfig
    train: TrainConfig
    loss: LossConfig
    data: DataConfig
    model: ModelConfig
    eval: EvalConfig
    runtime_seg: RuntimeSegConfig
    runtime_depth: RuntimeDepthConfig


def _to_abs(base_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    base_dir = os.path.dirname(os.path.abspath(path))
    workspace = os.path.dirname(base_dir)

    paths = PathsConfig(
        train_manifest=_to_abs(workspace, raw["paths"]["train_manifest"]) if raw["paths"].get("train_manifest") else "",
        val_manifest=_to_abs(workspace, raw["paths"]["val_manifest"]) if raw["paths"].get("val_manifest") else "",
        test_manifest=_to_abs(workspace, raw["paths"]["test_manifest"]) if raw["paths"].get("test_manifest") else "",
        train_video_root=_to_abs(workspace, raw["paths"]["train_video_root"]) if raw["paths"].get("train_video_root") else "",
        val_video_root=_to_abs(workspace, raw["paths"]["val_video_root"]) if raw["paths"].get("val_video_root") else "",
        test_video_root=_to_abs(workspace, raw["paths"]["test_video_root"]) if raw["paths"].get("test_video_root") else "",
        bg_depth_root=_to_abs(workspace, raw["paths"]["bg_depth_root"]) if raw["paths"].get("bg_depth_root") else "",
        output_dir=_to_abs(workspace, raw["paths"]["output_dir"]),
    )

    loss_raw = dict(raw["loss"])
    if loss_raw.get("pairwise_json"):
        loss_raw["pairwise_json"] = _to_abs(workspace, loss_raw["pairwise_json"])
    loss_raw.setdefault("consistency_mode", "map")
    if loss_raw.get("height_labels_json"):
        loss_raw["height_labels_json"] = _to_abs(workspace, loss_raw["height_labels_json"])
    loss_raw.setdefault("listwise_lambda", 0.0)
    loss_raw.setdefault("height_labels_json", "")

    runtime_seg_raw = raw.get(
        "runtime_seg",
        {
            "enabled": True,
            "model_path": "",
            "conf": 0.25,
            "iou": 0.7,
            "imgsz": 640,
            "strict_native": True,
        },
    )
    runtime_depth_raw = raw.get(
        "runtime_depth",
        {
            "enabled": False,
            "depthanything_root": "",
            "encoder": "vitl",
            "checkpoint": "",
            "input_size": 518,
            "assume_inverse": False,
        },
    )
    if runtime_depth_raw.get("depthanything_root"):
        runtime_depth_raw["depthanything_root"] = _to_abs(workspace, runtime_depth_raw["depthanything_root"])
    if runtime_depth_raw.get("checkpoint"):
        runtime_depth_raw["checkpoint"] = _to_abs(workspace, runtime_depth_raw["checkpoint"])
    runtime_depth_raw.setdefault("assume_inverse", False)
    runtime_depth_raw.setdefault("use_ground_anchor", True)
    runtime_depth_raw.setdefault("trainable", False)
    runtime_depth_raw.setdefault("train_parts", "depth_head")
    runtime_depth_raw.setdefault("depth_lr", 1e-5)
    runtime_depth_raw.setdefault("use_depth_cache_during_train", True)
    runtime_seg_raw.setdefault("allow_inference_fallback", True)

    eval_raw = dict(raw["eval"])
    eval_raw.setdefault("video_feature_frames", 8)
    eval_raw.setdefault("save_train_gallery", True)
    eval_raw.setdefault("frames_per_person_eval", 10)
    eval_raw.setdefault("frame_eval_seed", 42)
    eval_raw.setdefault("eval_batch_size", 16)
    eval_raw.setdefault("validate_every_epochs", 1)

    model_raw = dict(raw["model"])
    model_raw.setdefault("comparator_channels", 16)
    model_raw.setdefault("comparator_type", "conv")
    model_raw.setdefault("comparator_layers", 2)
    model_raw.setdefault("comparator_num_heads", 4)
    model_raw.setdefault("comparator_patch_size", 16)
    model_raw.setdefault("person_region_mode", "mask")
    model_raw.setdefault("bbox_expand_ratio", 0.25)
    model_raw.setdefault("histogram_min", 0.0)
    model_raw.setdefault("histogram_max", 3.0)
    model_raw.setdefault("compare_type", "concat")
    model_raw.setdefault("crop_and_resize_person", True)
    model_raw.setdefault("use_geometry_branch", False)
    model_raw.setdefault("fusion_type", "add")
    model_raw.setdefault("geo_feat_dim", 12)
    model_raw.setdefault("geo_hidden_dim", 32)

    return Config(
        seed=raw["seed"],
        device=raw["device"],
        paths=paths,
        train=TrainConfig(**raw["train"]),
        loss=LossConfig(**loss_raw),
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**model_raw),
        eval=EvalConfig(**eval_raw),
        runtime_seg=RuntimeSegConfig(**runtime_seg_raw),
        runtime_depth=RuntimeDepthConfig(**runtime_depth_raw),
    )
