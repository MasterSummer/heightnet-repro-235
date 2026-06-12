from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset

_HERE = os.path.abspath(os.path.dirname(__file__))
for _path in [os.path.join(_HERE, "src"), _HERE]:
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.append(_path)

from heightnet.config import load_config
from heightnet.gallery import sample_frame_row_indices_per_person
from heightnet.losses import person_mask_is_valid
from heightnet.model import DerivedHeightRanker
from heightnet.person_cache import infer_or_load_person_regions
from heightnet.runtime_depth import RuntimeDepthEstimator
from heightnet.runtime_seg import PersonSegmenter
from heightnet.utils import ensure_dir
from train_derived_rank import _build_dataset, collate_fn, derive_height_batch


def _torch_load_compat(path: str, map_location: str | torch.device, *, weights_only: bool):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_model(cfg, checkpoint: str, device: torch.device) -> DerivedHeightRanker:
    model = DerivedHeightRanker(
        comparator_channels=cfg.model.comparator_channels,
        comparator_type=cfg.model.comparator_type,
        comparator_layers=cfg.model.comparator_layers,
        comparator_num_heads=cfg.model.comparator_num_heads,
        comparator_patch_size=cfg.model.comparator_patch_size,
        person_region_mode=cfg.model.person_region_mode,
        bbox_expand_ratio=cfg.model.bbox_expand_ratio,
        histogram_min=getattr(cfg.model, "histogram_min", 0.0),
        histogram_max=getattr(cfg.model, "histogram_max", 3.0),
        compare_type=getattr(cfg.model, "compare_type", "concat"),
        use_geometry_branch=getattr(cfg.model, "use_geometry_branch", False),
        geo_feat_dim=getattr(cfg.model, "geo_feat_dim", 12),
        geo_hidden_dim=getattr(cfg.model, "geo_hidden_dim", 32),
    ).to(device)
    ckpt = _torch_load_compat(checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def _sample_frame_row_indices_per_sequence(dataset, max_per_sequence: int, seed: int) -> list[int]:
    import random

    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(dataset.rows):
        grouped[str(row.sequence_id)].append(idx)
    rng = random.Random(seed)
    out: list[int] = []
    for _, indices in sorted(grouped.items()):
        indices = list(indices)
        rng.shuffle(indices)
        out.extend(sorted(indices[: max(1, int(max_per_sequence))]))
    return sorted(out)


@torch.no_grad()
def _build_sequence_frame_gallery(
    model: DerivedHeightRanker,
    dataset,
    device: torch.device,
    runtime_depth: RuntimeDepthEstimator,
    segmenter: PersonSegmenter,
    row_indices: list[int],
    batch_size: int,
    num_workers: int,
    min_valid_pixels: int,
    min_valid_ratio: float,
    eps: float,
    assume_inverse: bool,
    use_ground_anchor: bool,
) -> list[dict]:
    records: list[dict] = []
    loader = DataLoader(
        Subset(dataset, row_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )
    was_training = model.training
    model.eval()
    for batch in loader:
        frame_paths = batch.get("frame_path", [])
        if frame_paths and all(isinstance(path, str) and path for path in frame_paths):
            person_mask, person_bbox = infer_or_load_person_regions(
                images_raw=batch["image_raw"],
                frame_paths=frame_paths,
                segmenter=segmenter,
                device=device,
                allow_inference_fallback=bool(getattr(segmenter, "allow_inference_fallback", True)),
            )
        else:
            person_mask, person_bbox = segmenter.infer_batch_regions(batch["image_raw"], device)
        derived = derive_height_batch(
            batch,
            runtime_depth,
            device,
            eps,
            assume_inverse,
            person_bbox=person_bbox,
            use_ground_anchor=use_ground_anchor,
        )
        if tuple(person_mask.shape[-2:]) != tuple(derived.shape[-2:]):
            src_h, src_w = person_mask.shape[-2:]
            dst_h, dst_w = derived.shape[-2:]
            scale_x = float(dst_w) / float(src_w)
            scale_y = float(dst_h) / float(src_h)
            person_mask = torch.nn.functional.interpolate(person_mask.to(device), size=(dst_h, dst_w), mode="nearest")
            person_bbox = person_bbox.to(device=device, dtype=torch.float32).clone()
            person_bbox[:, [0, 2]] *= scale_x
            person_bbox[:, [1, 3]] *= scale_y
        else:
            person_mask = person_mask.to(device)
            person_bbox = person_bbox.to(device=device, dtype=torch.float32)

        keep = person_mask_is_valid(person_mask, min_valid_pixels=min_valid_pixels, min_valid_ratio=min_valid_ratio)
        if not bool(keep.any().item()):
            continue
        derived_bg = batch["bg_depth"].to(device)
        feats = model.encode_person(derived, person_mask, person_bbox, derived_bg).detach().cpu()
        for idx, is_valid in enumerate(keep.detach().cpu().tolist()):
            if not is_valid:
                continue
            records.append(
                {
                    "sequence_id": str(batch["sequence_id"][idx]),
                    "person_id": str(batch["person_id"][idx]),
                    "camera_id": str(batch["camera_id"][idx]),
                    "frame_idx": int(batch["frame_idx"][idx]),
                    "frame_path": str(batch.get("frame_path", [""])[idx]) if batch.get("frame_path") else "",
                    "feature": feats[idx],
                }
            )
    model.train(was_training)
    return records


def _group_sequences(records: list[dict]) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        grouped[str(rec["camera_id"])][str(rec["sequence_id"])].append(rec)

    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for camera_id, seqs in grouped.items():
        for sequence_id, recs in seqs.items():
            out[camera_id][sequence_id] = {
                "features": [r["feature"] for r in recs],
                "person_id": recs[0]["person_id"],
                "n_frames": len(recs),
                "frame_indices": [int(r["frame_idx"]) for r in recs],
                "frame_paths": [r["frame_path"] for r in recs if r.get("frame_path")],
            }
    return out


@torch.no_grad()
def _rank_sequences_for_camera(model: DerivedHeightRanker, seq_features: dict[str, dict], device: torch.device) -> list[dict]:
    ids = sorted(seq_features)
    wins = {sid: 0 for sid in ids}
    comparisons = {sid: 0 for sid in ids}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            sid_i = ids[i]
            sid_j = ids[j]
            vote_i = 0
            vote_j = 0
            for fi in seq_features[sid_i]["features"]:
                for fj in seq_features[sid_j]["features"]:
                    prob = torch.sigmoid(model.compare_encoded(fi.unsqueeze(0).to(device), fj.unsqueeze(0).to(device)))[0].item()
                    if prob >= 0.5:
                        vote_i += 1
                    else:
                        vote_j += 1
            if vote_i >= vote_j:
                wins[sid_i] += 1
            else:
                wins[sid_j] += 1
            comparisons[sid_i] += 1
            comparisons[sid_j] += 1

    ranking = []
    for sid in ids:
        info = seq_features[sid]
        score = float(wins[sid] / max(comparisons[sid], 1))
        ranking.append(
            {
                "sequence_id": sid,
                "person_id": info["person_id"],
                "score": score,
                "wins": int(wins[sid]),
                "comparisons": int(comparisons[sid]),
                "n_frames": int(info["n_frames"]),
                "frame_indices": info["frame_indices"],
                "frame_paths": info["frame_paths"][:5],
            }
        )
    ranking.sort(key=lambda item: (-item["score"], -item["wins"], item["sequence_id"]))
    for idx, item in enumerate(ranking, start=1):
        item["rank"] = idx
    return ranking


def main() -> None:
    parser = argparse.ArgumentParser(description="Export video/sequence-level rankings from a DerivedHeightRanker checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--frames-per-sequence", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = _load_model(cfg, args.checkpoint, device)
    runtime_depth = RuntimeDepthEstimator(
        depthanything_root=cfg.runtime_depth.depthanything_root,
        encoder=cfg.runtime_depth.encoder,
        checkpoint=cfg.runtime_depth.checkpoint,
        input_size=cfg.runtime_depth.input_size,
    ).to(device)
    segmenter = None
    if bool(getattr(cfg.runtime_seg, "allow_inference_fallback", True)):
        segmenter = PersonSegmenter(
            model_path=cfg.runtime_seg.model_path,
            conf=cfg.runtime_seg.conf,
            iou=cfg.runtime_seg.iou,
            imgsz=cfg.runtime_seg.imgsz,
            strict_native=cfg.runtime_seg.strict_native,
        )
        segmenter.allow_inference_fallback = True

    split_to_manifest = {
        "train": (cfg.paths.train_manifest, cfg.paths.train_video_root),
        "val": (cfg.paths.val_manifest, cfg.paths.val_video_root),
        "test": (cfg.paths.test_manifest, cfg.paths.test_video_root),
    }
    selected = ["train", "val", "test"] if args.split == "all" else [args.split]
    all_records: list[dict] = []
    split_counts = {}
    for split in selected:
        manifest, video_root = split_to_manifest[split]
        dataset = _build_dataset(cfg, manifest, video_root, train_mode=False)
        indices = _sample_frame_row_indices_per_sequence(dataset, max_per_sequence=args.frames_per_sequence, seed=cfg.eval.frame_eval_seed)
        records = _build_sequence_frame_gallery(
            model=model,
            dataset=dataset,
            device=device,
            runtime_depth=runtime_depth,
            segmenter=segmenter,
            row_indices=indices,
            batch_size=getattr(cfg.eval, "eval_batch_size", 16),
            num_workers=cfg.train.num_workers,
            min_valid_pixels=cfg.loss.min_valid_pixels,
            min_valid_ratio=cfg.loss.min_valid_ratio,
            eps=cfg.loss.eps,
            assume_inverse=cfg.runtime_depth.assume_inverse,
            use_ground_anchor=getattr(cfg.runtime_depth, "use_ground_anchor", True),
        )
        split_counts[split] = {"rows": len(dataset), "sampled_rows": len(indices), "valid_feature_frames": len(records)}
        all_records.extend(records)

    grouped = _group_sequences(all_records)
    cameras = {}
    for camera_id, seq_features in sorted(grouped.items()):
        cameras[camera_id] = {
            "num_sequences": len(seq_features),
            "ranking": _rank_sequences_for_camera(model, seq_features, device),
        }

    output = {
        "config": os.path.abspath(args.config),
        "checkpoint": os.path.abspath(args.checkpoint),
        "split": args.split,
        "frames_per_sequence": int(args.frames_per_sequence),
        "split_counts": split_counts,
        "summary": {
            "num_cameras": len(cameras),
            "num_sequences": sum(cam["num_sequences"] for cam in cameras.values()),
            "num_feature_frames": len(all_records),
        },
        "cameras": cameras,
    }
    ensure_dir(os.path.dirname(args.out))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[SEQUENCE_RANK] wrote {args.out}")
    print(json.dumps(output["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
