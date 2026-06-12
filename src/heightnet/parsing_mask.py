from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

import cv2
import numpy as np

_PARSING_MEMBER_RE = re.compile(r"^parsing/([^/]+)/([^/]+)/([0-9]+)\.bmp$")


ParsingIndex = Dict[str, Dict[int, str]]


def parsing_dir_from_sequence_id(sequence_id: str) -> str:
    """Convert manifest sequence id pid__video_stem to parsing archive directory."""
    return str(sequence_id).replace("__", "_", 1)


def parse_parsing_filelist(lines: Iterable[str]) -> ParsingIndex:
    index: Dict[str, Dict[int, str]] = defaultdict(dict)
    for raw in lines:
        member = raw.strip()
        match = _PARSING_MEMBER_RE.match(member)
        if not match:
            continue
        parsing_dir, _track_id, frame_idx_text = match.groups()
        frame_idx = int(frame_idx_text)
        index[parsing_dir].setdefault(frame_idx, member)
    return {key: dict(value) for key, value in index.items()}


def parse_parsing_filelist_path(path: str | Path) -> ParsingIndex:
    with Path(path).open("r", encoding="utf-8") as f:
        return parse_parsing_filelist(f)


def build_full_frame_mask_from_parsing(
    parsing_mask: np.ndarray,
    bbox_xyxy: np.ndarray,
    image_height: int,
    image_width: int,
) -> np.ndarray:
    mask = np.zeros((int(image_height), int(image_width)), dtype=np.float32)
    if parsing_mask.size == 0:
        return mask

    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_xyxy[:4]]
    x1 = max(0, min(int(image_width), x1))
    x2 = max(0, min(int(image_width), x2))
    y1 = max(0, min(int(image_height), y1))
    y2 = max(0, min(int(image_height), y2))
    if x2 <= x1 or y2 <= y1:
        return mask

    crop = (parsing_mask > 0).astype(np.uint8)
    resized = cv2.resize(crop, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    mask[y1:y2, x1:x2] = resized.astype(np.float32)
    return mask


def resolve_extracted_parsing_path(extract_root: str | Path, member: str) -> Path:
    return Path(extract_root) / member
