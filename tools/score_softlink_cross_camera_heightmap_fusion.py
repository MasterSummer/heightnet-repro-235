#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.cross_camera_heightmap_fusion_core import (
    FEATURE_SCHEMA,
    CrossCameraFusionRanker,
    camera_geometry_features,
    camera_height_m,
    load_height_labels,
    load_npz_frames,
    soft_copeland_scores,
)


def _torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = _torch_load(checkpoint_path)
    required = {"model_state_dict", "camera_vocab", "feature_schema", "mu", "sd", "embedding_dim"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"{checkpoint_path}: missing checkpoint fields {missing}")
    if list(checkpoint["feature_schema"]) != FEATURE_SCHEMA:
        raise ValueError(f"{checkpoint_path}: feature schema mismatch")
    camera_vocab = {str(key): int(value) for key, value in checkpoint["camera_vocab"].items()}
    camera_geometry_dim = int(checkpoint.get("camera_geometry_dim", 0) or 0)
    model = CrossCameraFusionRanker(
        camera_count=len(camera_vocab),
        embedding_dim=int(checkpoint["embedding_dim"]),
        camera_geometry_dim=camera_geometry_dim,
        crop_encoder=str(checkpoint.get("crop_encoder", "cnn")),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, camera_vocab, np.asarray(checkpoint["mu"], dtype=np.float32), np.asarray(checkpoint["sd"], dtype=np.float32), checkpoint


def _encode_candidate(
    model,
    candidate: dict,
    camera_vocab: dict[str, int],
    mu: np.ndarray,
    sd: np.ndarray,
    device: torch.device,
    checkpoint: dict,
    geometry_config: dict | None = None,
) -> torch.Tensor:
    camera_id = str(candidate["camera_id"])
    if camera_id not in camera_vocab:
        raise ValueError(f"unknown camera_id={camera_id!r}")
    frames = load_npz_frames(candidate["expected_npz_path"])
    tabular, crop = frames.aggregate()
    crop_encoder = str(checkpoint.get("crop_encoder", "cnn"))
    encoder_track_frames = int(checkpoint.get("encoder_track_frames", 1) or 1)
    if crop_encoder == "geovt":
        if encoder_track_frames <= 1 or frames.count == 1:
            indices = [0]
        else:
            indices = np.linspace(0, frames.count - 1, num=encoder_track_frames, dtype=np.int64).astype(int).tolist()
        crop_tensor = torch.tensor(frames.crops[indices][None], dtype=torch.float32, device=device)
    else:
        crop_tensor = torch.tensor(crop[None], dtype=torch.float32, device=device)
    extra = []
    if geometry_config and geometry_config.get("enabled"):
        kwargs = {
            "image_width": int(geometry_config.get("image_width", 800)),
            "image_height": int(geometry_config.get("image_height", 600)),
            "pitch_deg": float(geometry_config.get("pitch_deg", 18.0)),
            "focal_px": geometry_config.get("focal_px"),
        }
        if kwargs["focal_px"] is not None:
            kwargs["focal_px"] = float(kwargs["focal_px"])
        extra.append(torch.tensor(camera_geometry_features(tabular, camera_id, **kwargs)[None], dtype=torch.float32, device=device))
    with torch.no_grad():
        return model.encode(
            torch.tensor(((tabular - mu) / sd)[None], dtype=torch.float32, device=device),
            crop_tensor,
            torch.tensor([camera_vocab[camera_id]], dtype=torch.long, device=device),
            torch.tensor([camera_height_m(camera_id)], dtype=torch.float32, device=device),
            *extra,
        )[0].cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--coverage-manifest", type=Path, default=Path("runs/home_data_heightmap_fusion/coverage_manifest.json"))
    parser.add_argument("--label-path", type=Path, default=Path("/home/zyding/data/label/rank.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/cross_camera_heightmap_fusion"))
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model, camera_vocab, mu, sd, checkpoint = _load_model(args.checkpoint, device)
    geometry_config = checkpoint.get("camera_geometry")
    labels = load_height_labels(args.label_path)
    coverage = json.loads(args.coverage_manifest.read_text(encoding="utf-8"))
    candidates = [dict(item) for item in coverage["candidates"] if item.get("source") == "softlink"]
    if args.limit_videos > 0:
        candidates = candidates[: args.limit_videos]

    details: list[dict] = []
    person_embeddings: dict[str, list[torch.Tensor]] = defaultdict(list)
    status_counts: Counter[str] = Counter()
    for candidate in candidates:
        detail = {
            key: candidate.get(key)
            for key in ("candidate_id", "person_id", "group", "video_path", "video_filename", "video_stem", "camera_id", "expected_npz_path")
        }
        detail["has_height_label"] = candidate["person_id"] in labels
        detail["height_cm"] = labels.get(candidate["person_id"])
        detail["status"] = candidate.get("status", "scored")
        detail["reasons"] = list(candidate.get("reasons") or [])
        if detail["status"] == "scored":
            try:
                embedding = _encode_candidate(model, candidate, camera_vocab, mu, sd, device, checkpoint, geometry_config)
                person_embeddings[str(candidate["person_id"])].append(embedding)
                detail["embedding_contribution"] = 1
            except Exception as exc:
                detail["status"] = "partial"
                detail["reasons"].append("encode_failed")
                detail["error"] = str(exc)
        status_counts[detail["status"]] += 1
        details.append(detail)

    people = sorted(person_embeddings)
    embeddings = torch.stack([torch.stack(person_embeddings[pid]).mean(0) for pid in people]).to(device)
    with torch.no_grad():
        scores = soft_copeland_scores(embeddings, model.compare_encoded, chunk_size=args.chunk_size).cpu().tolist()
    ranking = []
    for rank, index in enumerate(sorted(range(len(people)), key=lambda idx: (-scores[idx], people[idx])), start=1):
        pid = people[index]
        ranking.append({
            "rank": rank,
            "person_id": pid,
            "score": float(scores[index]),
            "video_count": len(person_embeddings[pid]),
            "height_cm": labels.get(pid),
            "has_height_label": pid in labels,
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "softlink_person_rankings.json").write_text(json.dumps(ranking, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "softlink_video_details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "candidate_count": len(candidates),
        "ranked_video_count": sum(status == "scored" for status in [detail["status"] for detail in details]),
        "skipped_or_partial_video_count": sum(status != "scored" for status in [detail["status"] for detail in details]),
        "status_counts": dict(status_counts),
        "ranked_person_count": len(ranking),
        "labeled_person_count": sum(item["has_height_label"] for item in ranking),
        "unlabeled_person_count": sum(not item["has_height_label"] for item in ranking),
        "coverage_reconciles": len(details) == len(candidates),
        "camera_vocab": camera_vocab,
        "training": {
            key: checkpoint.get(key)
            for key in ("lambda_cross", "lambda_track", "seed", "best_epoch", "best_val_metrics", "split_overlap_summary")
        },
        "camera_geometry": geometry_config,
    }
    (args.out_dir / "softlink_coverage_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["coverage_reconciles"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
