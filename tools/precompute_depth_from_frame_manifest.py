from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from heightnet.config import load_config
from heightnet.runtime_depth import RuntimeDepthEstimator


def _load_rgb_chw(frame_path: str) -> torch.Tensor | None:
    img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.transpose(img, (2, 0, 1))).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute frame_path + .depth.npy for extracted frame manifests.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--manifest", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    runtime_depth = RuntimeDepthEstimator(
        depthanything_root=cfg.runtime_depth.depthanything_root,
        encoder=cfg.runtime_depth.encoder,
        checkpoint=cfg.runtime_depth.checkpoint,
        input_size=cfg.runtime_depth.input_size,
    ).to(device)

    rows = []
    for manifest in args.manifest:
        rows.extend(pd.read_csv(manifest).to_dict(orient="records"))

    pending = []
    skipped_existing = 0
    missing = 0
    for row in rows:
        frame_path = str(row.get("frame_path", "")).strip()
        out_path = frame_path + ".depth.npy"
        if not frame_path or not os.path.exists(frame_path):
            missing += 1
            continue
        if os.path.exists(out_path) and not args.overwrite_existing:
            skipped_existing += 1
            continue
        pending.append(frame_path)

    saved = 0
    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(pending), batch_size):
        paths = pending[start : start + batch_size]
        images = []
        valid_paths = []
        for path in paths:
            tensor = _load_rgb_chw(path)
            if tensor is None:
                missing += 1
                continue
            images.append(tensor)
            valid_paths.append(path)
        if not images:
            continue
        batch = torch.stack(images, dim=0).to(device=device, dtype=torch.uint8)
        depth = runtime_depth.infer_batch(batch).detach().cpu().numpy()
        for idx, path in enumerate(valid_paths):
            np.save(path + ".depth.npy", depth[idx, 0].astype(np.float32))
            saved += 1
        if saved and saved % 500 == 0:
            print(f"[DEPTH_CACHE] saved={saved}/{len(pending)}")

    print(
        f"[DEPTH_CACHE] total={len(rows)} pending={len(pending)} saved={saved} "
        f"skipped_existing={skipped_existing} missing_frame={missing}"
    )


if __name__ == "__main__":
    main()
