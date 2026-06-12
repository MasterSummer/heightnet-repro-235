import cv2
import numpy as np


def expand_bbox_xyxy(bbox: np.ndarray, expand_ratio: float) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    dx = w * float(expand_ratio)
    dy = h * float(expand_ratio)
    return np.array([x1 - dx, y1 - dy, x2 + dx, y2 + dy], dtype=np.float32)


def clamp_bbox_xyxy(bbox: np.ndarray, image_w: int, image_h: int) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return np.array([
        np.clip(x1, 0.0, float(image_w)),
        np.clip(y1, 0.0, float(image_h)),
        np.clip(x2, 0.0, float(image_w)),
        np.clip(y2, 0.0, float(image_h)),
    ], dtype=np.float32)


def resize_and_pad_map(arr: np.ndarray, target_h: int, target_w: int, pad_value: float):
    src_h, src_w = arr.shape[:2]
    scale = min(float(target_h) / float(src_h), float(target_w) / float(src_w))
    scaled_h = max(1, int(round(src_h * scale)))
    scaled_w = max(1, int(round(src_w * scale)))
    resized = cv2.resize(arr, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    pad_top = (target_h - scaled_h) // 2
    pad_bottom = target_h - scaled_h - pad_top
    pad_left = (target_w - scaled_w) // 2
    pad_right = target_w - scaled_w - pad_left
    out = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=pad_value)
    return out, {
        "scale": scale,
        "scaled_h": scaled_h,
        "scaled_w": scaled_w,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_left": pad_left,
        "pad_right": pad_right,
    }
