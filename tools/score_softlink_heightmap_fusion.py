#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.home_data_heightmap_fusion import FEATURE_SCHEMA, FusionRegressor, aggregate_sequence_npz, base_camera_from_name, load_rank_csv


DEFAULT_CHECKPOINT_DIR = Path("/home/zyding/height/heightnet_repro/runs/home_data_heightmap_fusion/task8_full/checkpoints")
DEFAULT_COVERAGE_MANIFEST = Path("/home/zyding/height/heightnet_repro/runs/home_data_heightmap_fusion/coverage_manifest.json")
DEFAULT_FEATURE_ROOT = Path("/home/zyding/height/jianzhi_2511_sequence/features")
DEFAULT_OUTPUT = Path("/home/zyding/height/heightnet_repro/runs/home_data_heightmap_fusion/task9_softlink_smoke.json")
DEFAULT_SOFTLINK_ROOT = Path("/data2/dataset/jianzhi_2511/jianzhi_2511_spilt_coat_softlink")
DEFAULT_LABEL_PATH = Path("/home/zyding/data/label/rank.json")


class LoadedCheckpoint:
    def __init__(self, camera_id: str, variant: str, path: Path, model: FusionRegressor, mu: np.ndarray, sd: np.ndarray):
        self.camera_id = camera_id
        self.variant = variant
        self.path = path
        self.model = model
        self.mu = mu
        self.sd = sd


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_fusion_checkpoint(path: Path, camera_id: str, variant: str, device: str = "cpu") -> LoadedCheckpoint:
    ckpt = _torch_load(path)
    required = ["model_state_dict", "mu", "sd", "feature_schema", "camera_id", "variant"]
    missing = [k for k in required if k not in ckpt]
    if missing:
        raise ValueError(f"{path}: missing checkpoint fields {missing}")
    ckpt_camera_id = ckpt["camera_id"]
    ckpt_variant = ckpt["variant"]
    ckpt_feature_schema = ckpt["feature_schema"]
    if ckpt_camera_id != camera_id:
        raise ValueError(f"{path}: camera_id {ckpt_camera_id!r} does not match expected {camera_id!r}")
    if ckpt_variant != variant:
        raise ValueError(f"{path}: variant {ckpt_variant!r} does not match expected {variant!r}")
    if list(ckpt_feature_schema) != list(FEATURE_SCHEMA):
        raise ValueError(f"{path}: feature_schema mismatch; expected {list(FEATURE_SCHEMA)!r}, got {ckpt_feature_schema!r}")
    mu = np.asarray(ckpt["mu"], dtype=np.float32)
    sd = np.asarray(ckpt["sd"], dtype=np.float32)
    if mu.shape != (len(FEATURE_SCHEMA),):
        raise ValueError(f"{path}: mu must have shape ({len(FEATURE_SCHEMA)},), got {mu.shape}")
    if sd.shape != (len(FEATURE_SCHEMA),):
        raise ValueError(f"{path}: sd must have shape ({len(FEATURE_SCHEMA)},), got {sd.shape}")
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sd)):
        raise ValueError(f"{path}: mu/sd contain non-finite values")
    sd = sd.copy()
    sd[sd < 1e-6] = 1.0
    model = FusionRegressor(variant, tabular_dim=len(FEATURE_SCHEMA), hidden=128)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return LoadedCheckpoint(camera_id, variant, path, model, mu, sd)


def _manifest_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates") if isinstance(payload, dict) else payload
    if not isinstance(candidates, list):
        raise ValueError(f"{path}: expected a list or dict with candidates")
    return [dict(c) for c in candidates if c.get("source") == "softlink"]


def _softlink_candidates(softlink_root: Path, feature_root: Path, label_path: Path | None) -> list[dict[str, Any]]:
    heights = load_rank_csv(label_path) if label_path and label_path.exists() else {}
    candidates: list[dict[str, Any]] = []
    for video_path in sorted(softlink_root.rglob("*.mp4")):
        person_id = video_path.parent.name
        video_stem = video_path.stem
        camera_id = base_camera_from_name(video_path.name)
        reasons: list[str] = []
        status = "scored"
        if not camera_id:
            status = "skipped"
            reasons.append("missing_camera_id")
            camera_id = "unknown"
        npz_path = feature_root / person_id / f"{video_stem}.npz"
        if not npz_path.exists():
            status = "skipped"
            reasons.append("missing_expected_npz")
        has_label = person_id in heights
        candidates.append({
            "candidate_id": f"softlink:{person_id}:{video_stem}",
            "source": "softlink",
            "split": "softlink",
            "person_id": person_id,
            "video_path": str(video_path),
            "video_filename": video_path.name,
            "video_stem": video_stem,
            "camera_id": camera_id,
            "expected_npz_path": str(npz_path),
            "height_cm": float(heights[person_id]) if has_label else None,
            "has_height_label": has_label,
            "status": status,
            "reasons": reasons,
        })
    return candidates


def select_limited_candidates(candidates: list[dict[str, Any]], limit_videos: int | None) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=lambda c: (str(c.get("camera_id", "")), str(c.get("person_id", "")), str(c.get("video_stem", "")), str(c.get("candidate_id", ""))))
    if not limit_videos or limit_videos <= 0 or len(ordered) <= limit_videos:
        return ordered
    selected = ordered[:limit_videos]
    has_labeled = any(bool(c.get("has_height_label")) for c in selected)
    has_unlabeled = any(not bool(c.get("has_height_label")) for c in selected)
    all_has_labeled = any(bool(c.get("has_height_label")) for c in ordered)
    all_has_unlabeled = any(not bool(c.get("has_height_label")) for c in ordered)
    if limit_videos >= 2 and all_has_labeled and all_has_unlabeled and not (has_labeled and has_unlabeled):
        labeled = next(c for c in ordered if bool(c.get("has_height_label")))
        unlabeled = next(c for c in ordered if not bool(c.get("has_height_label")))
        selected = [unlabeled, labeled]
        seen = {id(unlabeled), id(labeled)}
        for c in ordered:
            if id(c) in seen:
                continue
            selected.append(c)
            if len(selected) >= limit_videos:
                break
    return selected


def _candidate_npz_path(candidate: dict[str, Any], feature_root: Path) -> Path:
    expected = candidate.get("expected_npz_path")
    if expected:
        return Path(expected)
    return feature_root / str(candidate["person_id"]) / f"{candidate[video_stem]}.npz"


def _entry_base(candidate: dict[str, Any], checkpoint_path: str | None = None) -> dict[str, Any]:
    has_label = bool(candidate.get("has_height_label"))
    height = candidate.get("height_cm") if has_label else None
    reasons = candidate.get("reasons") or []
    return {
        "candidate_id": candidate.get("candidate_id"),
        "source": candidate.get("source", "softlink"),
        "split": candidate.get("split", "softlink"),
        "person_id": candidate.get("person_id"),
        "video_filename": candidate.get("video_filename"),
        "video_path": candidate.get("video_path"),
        "video_stem": candidate.get("video_stem"),
        "group": candidate.get("group", candidate.get("camera_id")),
        "camera_id": candidate.get("camera_id"),
        "height_cm": float(height) if height is not None else None,
        "has_height_label": has_label,
        "status": candidate.get("status", "scored"),
        "reasons": list(reasons),
        "score": None,
        "rank": None,
        "npz_path": None,
        "checkpoint_path": checkpoint_path,
    }


def score_manifest_candidates(
    checkpoint_dir: Path,
    coverage_manifest: Path | None = None,
    feature_root: Path = DEFAULT_FEATURE_ROOT,
    variant: str = "fused_concat",
    limit_videos: int | None = None,
    device: str = "cpu",
    softlink_root: Path | None = None,
    label_path: Path | None = DEFAULT_LABEL_PATH,
) -> dict[str, Any]:
    if coverage_manifest is not None:
        candidates = _manifest_candidates(Path(coverage_manifest))
        candidate_source = str(coverage_manifest)
    elif softlink_root is not None:
        candidates = _softlink_candidates(Path(softlink_root), Path(feature_root), label_path)
        candidate_source = str(softlink_root)
    else:
        raise ValueError("coverage_manifest or softlink_root is required")

    selected = select_limited_candidates(candidates, limit_videos)
    checkpoints: dict[str, LoadedCheckpoint] = {}
    by_camera: dict[str, list[dict[str, Any]]] = defaultdict(list)
    checkpoint_errors: dict[str, str] = {}

    for candidate in selected:
        camera_id = candidate.get("camera_id")
        ckpt_path = Path(checkpoint_dir) / f"{camera_id}_{variant}_best.pt"
        entry = _entry_base(candidate, str(ckpt_path))
        npz_path = _candidate_npz_path(candidate, Path(feature_root))
        entry["npz_path"] = str(npz_path)
        if entry["status"] != "scored":
            by_camera[str(camera_id)].append(entry)
            continue
        if not ckpt_path.exists():
            entry["status"] = "skipped"
            entry["reasons"].append("missing_checkpoint")
            by_camera[str(camera_id)].append(entry)
            continue
        if str(camera_id) not in checkpoints:
            try:
                checkpoints[str(camera_id)] = load_fusion_checkpoint(ckpt_path, str(camera_id), variant, device=device)
            except Exception as exc:
                checkpoint_errors[str(camera_id)] = str(exc)
                entry["status"] = "skipped"
                entry["reasons"].append("checkpoint_load_failed")
                entry["checkpoint_error"] = str(exc)
                by_camera[str(camera_id)].append(entry)
                continue
        if str(camera_id) in checkpoint_errors:
            entry["status"] = "skipped"
            entry["reasons"].append("checkpoint_load_failed")
            entry["checkpoint_error"] = checkpoint_errors[str(camera_id)]
            by_camera[str(camera_id)].append(entry)
            continue
        if not npz_path.exists():
            entry["status"] = "skipped"
            entry["reasons"].append("missing_expected_npz")
            by_camera[str(camera_id)].append(entry)
            continue
        try:
            tabular, crop = aggregate_sequence_npz(npz_path)
            loaded = checkpoints[str(camera_id)]
            x = ((tabular - loaded.mu) / loaded.sd).astype(np.float32, copy=False)
            with torch.no_grad():
                score = loaded.model(
                    torch.tensor(x[None, :], dtype=torch.float32, device=device),
                    torch.tensor(crop[None, :, :, :], dtype=torch.float32, device=device),
                ).detach().cpu().item()
            entry["score"] = float(score)
            entry["status"] = "scored"
            entry["reasons"] = []
        except Exception as exc:
            entry["status"] = "partial"
            entry["reasons"].append("score_failed")
            entry["score_error"] = str(exc)
        by_camera[str(camera_id)].append(entry)

    rankings: dict[str, list[dict[str, Any]]] = {}
    for camera_id, entries in sorted(by_camera.items()):
        entries.sort(key=lambda e: (e["status"] != "scored", -(e["score"] if e["score"] is not None else -1e30), str(e.get("candidate_id"))))
        rank = 1
        for entry in entries:
            if entry["status"] == "scored":
                entry["rank"] = rank
                rank += 1
        rankings[camera_id] = entries

    flat = [entry for entries in rankings.values() for entry in entries]
    status_counts = Counter(entry["status"] for entry in flat)
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "coverage_manifest": str(coverage_manifest) if coverage_manifest else None,
        "softlink_root": str(softlink_root) if softlink_root else None,
        "candidate_source": candidate_source,
        "feature_root": str(feature_root),
        "variant": variant,
        "feature_schema": list(FEATURE_SCHEMA),
        "limit_videos": limit_videos,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "scored_count": int(status_counts.get("scored", 0)),
        "labeled_count": sum(1 for entry in flat if entry["has_height_label"]),
        "unlabeled_count": sum(1 for entry in flat if not entry["has_height_label"]),
        "status_counts": dict(status_counts),
        "camera_rankings": rankings,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    ap.add_argument("--coverage-manifest", type=Path, default=DEFAULT_COVERAGE_MANIFEST)
    ap.add_argument("--softlink-root", type=Path, default=None)
    ap.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    ap.add_argument("--label-path", type=Path, default=DEFAULT_LABEL_PATH)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--variant", default="fused_concat")
    ap.add_argument("--limit-videos", type=int, default=0)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    manifest = args.coverage_manifest if args.coverage_manifest else None
    softlink_root = args.softlink_root if args.softlink_root else None
    payload = score_manifest_candidates(
        checkpoint_dir=args.checkpoint_dir,
        coverage_manifest=manifest,
        softlink_root=softlink_root,
        feature_root=args.feature_root,
        label_path=args.label_path,
        variant=args.variant,
        limit_videos=args.limit_videos,
        device=args.device,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OUT]", args.out)
    print("selected", payload["selected_count"], "scored", payload["scored_count"], "labeled", payload["labeled_count"], "unlabeled", payload["unlabeled_count"], "status", payload["status_counts"])


if __name__ == "__main__":
    main()
