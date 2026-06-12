from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .parsing_mask import build_full_frame_mask_from_parsing
from .person_cache import person_bbox_cache_path, person_mask_cache_path


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass(frozen=True)
class JianzhiParsingRecord:
    sequence_id: str
    person_id: str
    frame_idx: int
    bbox_xyxy: np.ndarray
    parsing_path: Path | None
    source_json: Path | None = None


class JianzhiParsingIndex:
    def __init__(self, records: Iterable[JianzhiParsingRecord]) -> None:
        self.records = list(records)
        self._by_key: dict[tuple[str, int], JianzhiParsingRecord] = {}
        for rec in self.records:
            for seq in _sequence_variants(rec.sequence_id, rec.person_id):
                self._by_key.setdefault((seq, int(rec.frame_idx)), rec)

    def lookup(self, sequence_id: str, frame_idx: int) -> JianzhiParsingRecord | None:
        return self._by_key.get((str(sequence_id), int(frame_idx)))


def normalize_bbox_xyxy(
    raw_bbox: Any,
    image_width: int | None = None,
    image_height: int | None = None,
    bbox_format: str = "auto",
) -> np.ndarray:
    """Normalize common bbox encodings to xyxy float32, optionally clipped."""
    fmt = bbox_format.lower()
    if isinstance(raw_bbox, dict):
        lower = {str(k).lower(): v for k, v in raw_bbox.items()}
        if {"x1", "y1", "x2", "y2"}.issubset(lower):
            vals = [lower["x1"], lower["y1"], lower["x2"], lower["y2"]]
            fmt = "xyxy"
        elif {"left", "top", "right", "bottom"}.issubset(lower):
            vals = [lower["left"], lower["top"], lower["right"], lower["bottom"]]
            fmt = "xyxy"
        elif {"x", "y", "w", "h"}.issubset(lower):
            vals = [lower["x"], lower["y"], lower["w"], lower["h"]]
            fmt = "xywh"
        elif "bbox" in lower:
            return normalize_bbox_xyxy(lower["bbox"], image_width, image_height, bbox_format=fmt)
        else:
            raise ValueError(f"unsupported bbox dict keys: {sorted(raw_bbox.keys())}")
    else:
        vals = list(raw_bbox)
        if len(vals) < 4:
            raise ValueError(f"bbox needs at least 4 values: {raw_bbox!r}")
        vals = vals[:4]
        if fmt == "auto":
            fmt = "xyxy"

    x1, y1, a, b = [float(v) for v in vals]
    if fmt == "xywh":
        x2 = x1 + a
        y2 = y1 + b
    elif fmt == "xyxy":
        x2 = a
        y2 = b
    else:
        raise ValueError(f"unsupported bbox_format={bbox_format!r}")

    box = np.array([x1, y1, x2, y2], dtype=np.float32)
    if image_width is not None:
        box[[0, 2]] = np.clip(box[[0, 2]], 0.0, float(image_width))
    if image_height is not None:
        box[[1, 3]] = np.clip(box[[1, 3]], 0.0, float(image_height))
    return box


def build_jianzhi_parsing_index(
    parsing_json_root: str | os.PathLike[str],
    parsing_bitmap_root: str | os.PathLike[str] | None = None,
    include_video_stems: set[str] | None = None,
) -> JianzhiParsingIndex:
    root = Path(parsing_json_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"parsing json root not found: {root}")
    bitmap_root = Path(parsing_bitmap_root).expanduser().resolve() if parsing_bitmap_root else None
    records: list[JianzhiParsingRecord] = []
    for json_path in sorted(root.rglob("*.json")):
        if include_video_stems and not (_json_video_stem_candidates(json_path.stem) & include_video_stems):
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        person_hint = _infer_person_id(json_path, root, data)
        records.extend(
            _records_from_json(
                data,
                json_path=json_path,
                root=root,
                person_hint=person_hint,
                parsing_bitmap_root=bitmap_root,
            )
        )
    return JianzhiParsingIndex(records)


def _from_records(items: Iterable[dict[str, Any]]) -> JianzhiParsingIndex:
    records = []
    for item in items:
        records.append(
            JianzhiParsingRecord(
                sequence_id=str(item["sequence_id"]),
                person_id=str(item.get("person_id") or _person_from_sequence(str(item["sequence_id"]))),
                frame_idx=int(item["frame_idx"]),
                bbox_xyxy=normalize_bbox_xyxy(item["bbox"], bbox_format=str(item.get("bbox_format", "auto"))),
                parsing_path=Path(str(item["parsing_path"])).expanduser().resolve()
                if item.get("parsing_path") else None,
                source_json=None,
            )
        )
    return JianzhiParsingIndex(records)


build_jianzhi_parsing_index.from_records = _from_records  # type: ignore[attr-defined]


def generate_person_region_cache_from_index(
    rows: Iterable[dict[str, Any]],
    index: JianzhiParsingIndex,
    overwrite_existing: bool = False,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "total": 0,
        "generated": 0,
        "skipped_existing": 0,
        "missing_record": 0,
        "missing_frame": 0,
        "missing_parsing": 0,
        "invalid_bbox": 0,
        "empty_mask": 0,
        "errors": [],
    }
    for row in rows:
        report["total"] += 1
        frame_path = str(row.get("frame_path", "")).strip()
        sequence_id = str(row.get("sequence_id", "")).strip()
        frame_idx = int(row.get("decoded_frame_idx", row.get("frame_idx", 0)))
        if not frame_path or not os.path.exists(frame_path):
            report["missing_frame"] += 1
            continue

        rec = index.lookup(sequence_id, frame_idx)
        if rec is None and "person_id" in row:
            rec = index.lookup(f"{row['person_id']}__{Path(str(row.get('video_path', sequence_id))).stem}", frame_idx)
        if rec is None:
            report["missing_record"] += 1
            continue
        if rec.parsing_path is None:
            report["missing_parsing"] += 1
            continue

        out_mask = person_mask_cache_path(frame_path)
        out_bbox = person_bbox_cache_path(frame_path)
        if not overwrite_existing and os.path.exists(out_mask) and os.path.exists(out_bbox):
            report["skipped_existing"] += 1
            continue
        if rec.parsing_path is not None and not rec.parsing_path.exists():
            report["missing_parsing"] += 1
            continue

        frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        if frame is None:
            report["missing_frame"] += 1
            continue
        image_h, image_w = frame.shape[:2]
        bbox = normalize_bbox_xyxy(rec.bbox_xyxy, image_width=image_w, image_height=image_h, bbox_format="xyxy")
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            report["invalid_bbox"] += 1
            continue

        parsing_mask = cv2.imread(str(rec.parsing_path), cv2.IMREAD_GRAYSCALE)
        if parsing_mask is None:
            report["missing_parsing"] += 1
            continue
        full_mask = build_full_frame_mask_from_parsing(parsing_mask, bbox, image_h, image_w)
        if int((full_mask > 0).sum()) == 0:
            report["empty_mask"] += 1
            continue

        Path(out_mask).parent.mkdir(parents=True, exist_ok=True)
        np.save(out_mask, (full_mask > 0).astype(np.uint8))
        np.save(out_bbox, bbox.astype(np.float32))
        report["generated"] += 1
    return report


def _records_from_json(
    data: Any,
    json_path: Path,
    root: Path,
    person_hint: str,
    parsing_bitmap_root: Path | None = None,
) -> list[JianzhiParsingRecord]:
    out: list[JianzhiParsingRecord] = []
    video_stem = _video_stem_from_json(data, json_path)
    sequence_id = f"{person_hint}__{video_stem}"
    for item, track_id in _iter_frame_items_with_context(data):
        frame_idx = _extract_frame_idx(item)
        bbox, bbox_format = _extract_bbox_and_format(item)
        parsing = _find_first(item, _is_parsing_path_value)
        if frame_idx is None or bbox is None:
            continue
        parsing_path = _resolve_path(root, json_path.parent, str(parsing)) if parsing is not None else None
        if parsing_path is None and parsing_bitmap_root is not None:
            parsing_path = _resolve_parsing_bitmap(parsing_bitmap_root, json_path.stem, int(frame_idx), track_id)
        try:
            bbox_xyxy = normalize_bbox_xyxy(bbox, bbox_format=bbox_format)
        except (TypeError, ValueError):
            continue
        out.append(
            JianzhiParsingRecord(
                sequence_id=sequence_id,
                person_id=person_hint,
                frame_idx=int(frame_idx),
                bbox_xyxy=bbox_xyxy,
                parsing_path=parsing_path,
                source_json=json_path,
            )
        )
    return out


def _iter_frame_items(data: Any) -> Iterable[dict[str, Any]]:
    for item, _ in _iter_frame_items_with_context(data):
        yield item


def _iter_frame_items_with_context(data: Any, track_id: str | None = None) -> Iterable[tuple[dict[str, Any], str | None]]:
    if isinstance(data, dict):
        for key in ("frames", "frame_infos", "items", "results", "data"):
            value = data.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item, track_id
                return
        if any(str(k).lower() in {"frame", "frame_idx", "frame_id", "bbox"} for k in data):
            yield data, track_id
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                next_track = str(key) if str(key).isdigit() else track_id
                yield from _iter_frame_items_with_context(value, next_track)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                if any(str(k).lower() in {"frame", "frame_idx", "frame_id", "bbox", "rect"} for k in item):
                    yield item, track_id
                yield from _iter_frame_items_with_context(item, track_id)


def _find_first(value: Any, pred) -> Any | None:
    if pred(value):
        return value
    if isinstance(value, dict):
        for child in value.values():
            found = _find_first(child, pred)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_first(child, pred)
            if found is not None:
                return found
    return None


def _is_bbox_value(value: Any) -> bool:
    if isinstance(value, dict):
        keys = {str(k).lower() for k in value.keys()}
        return bool({"x1", "y1", "x2", "y2"}.issubset(keys) or {"x", "y", "w", "h"}.issubset(keys))
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return all(isinstance(x, (int, float)) for x in value[:4])
    return False


def _extract_bbox_and_format(item: dict[str, Any]) -> tuple[Any | None, str]:
    for key in ("bbox", "box", "person_bbox", "body_bbox"):
        if key in item and _is_bbox_value(item[key]):
            return item[key], "xyxy"
    if "rect" in item and _is_bbox_value(item["rect"]):
        return item["rect"], "xywh"
    bbox = _find_first(item, _is_bbox_value)
    return bbox, "auto"


def _is_parsing_path_value(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    suffix = Path(value).suffix.lower()
    return suffix in {".png", ".jpg", ".jpeg", ".bmp", ".npy"}


def _extract_frame_idx(item: dict[str, Any]) -> int | None:
    for key in ("frame_idx", "frame_id", "frame", "image_id", "idx"):
        if key in item:
            return _parse_frame_idx(item[key])
    found = _find_first(item, lambda v: isinstance(v, str) and re.search(r"(\d{1,8})\.(?:png|jpg|jpeg|bmp)$", v))
    return _parse_frame_idx(found) if found is not None else None


def _parse_frame_idx(value: Any) -> int | None:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and np.isfinite(value):
        return int(value)
    if isinstance(value, str):
        m = re.search(r"(\d{1,8})(?:\.[A-Za-z0-9]+)?$", value)
        if m:
            return int(m.group(1))
    return None


def _video_stem_from_json(data: Any, json_path: Path) -> str:
    if isinstance(data, dict):
        for key in ("sequence_id", "video_name", "video", "video_path", "name"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return Path(value).stem
    person = _person_from_filename(json_path.stem)
    if person and json_path.stem.startswith(person + "_"):
        return json_path.stem[len(person) + 1 :]
    if "_" in json_path.stem:
        return json_path.stem.split("_", 1)[1]
    return json_path.stem


def _json_video_stem_candidates(stem: str) -> set[str]:
    out = {stem}
    person = _person_from_filename(stem)
    if person and stem.startswith(person + "_"):
        out.add(stem[len(person) + 1 :])
    if "_" in stem:
        out.add(stem.split("_", 1)[1])
    return out


def _infer_person_id(json_path: Path, root: Path, data: Any) -> str:
    if isinstance(data, dict):
        for key in ("person_id", "pid", "identity", "id"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    rel = json_path.relative_to(root)
    if len(rel.parts) > 1:
        return rel.parts[0]
    person = _person_from_filename(json_path.stem)
    if person:
        return person
    if "_" in json_path.stem:
        return json_path.stem.split("_", 1)[0]
    return _person_from_sequence(json_path.stem)


def _person_from_sequence(sequence_id: str) -> str:
    return str(sequence_id).split("__", 1)[0]


def _sequence_variants(sequence_id: str, person_id: str) -> set[str]:
    seq = str(sequence_id)
    variants = {seq}
    if "__" in seq:
        variants.add(seq.split("__", 1)[1])
    variants.add(f"{person_id}__{Path(seq).stem}")
    return variants


def _resolve_path(root: Path, json_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    for base in (json_dir, root):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (json_dir / path).resolve()


def _resolve_parsing_bitmap(root: Path, video_stem: str, frame_idx: int, track_id: str | None) -> Path | None:
    # The real jianzhi_2511 layout is parsing/<video_stem>/<track_id>/<frame_idx>.bmp.
    # Do not stat/glob here: the full index has ~1.7M records, so existence checks
    # belong in the cache generation stage where rows are already sampled/filtered.
    name = f"{int(frame_idx)}.bmp"
    if track_id:
        return (root / video_stem / str(track_id) / name).resolve()
    return (root / video_stem / name).resolve()


def _person_from_filename(stem: str) -> str | None:
    match = re.search(r"(\d{4}_(?:man|woman)\d+)", str(stem), flags=re.IGNORECASE)
    return match.group(1) if match else None
