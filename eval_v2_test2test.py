"""Evaluate v2 experiments with query=test, gallery=test."""
from __future__ import annotations
import argparse, json, os, sys, torch
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from heightnet.config import load_config
from heightnet.losses import HeightNetLoss
from heightnet.model import DerivedHeightRanker
from heightnet.ranking import build_ranked_candidates
from heightnet.runtime_depth import RuntimeDepthEstimator
from heightnet.runtime_seg import PersonSegmenter
from train_derived_rank import _build_dataset, evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = _build_dataset(cfg, cfg.paths.test_manifest, cfg.paths.test_video_root, train_mode=False)

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
        geo_feat_dim=getattr(cfg.model, "geo_feat_dim", 5),
        geo_hidden_dim=getattr(cfg.model, "geo_hidden_dim", 32),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    runtime_depth = RuntimeDepthEstimator(
        depthanything_root=cfg.runtime_depth.depthanything_root,
        encoder=cfg.runtime_depth.encoder,
        checkpoint=cfg.runtime_depth.checkpoint,
        input_size=cfg.runtime_depth.input_size,
    ).to(device)
    segmenter = PersonSegmenter(
        model_path=cfg.runtime_seg.model_path,
        conf=cfg.runtime_seg.conf,
        iou=cfg.runtime_seg.iou,
        imgsz=cfg.runtime_seg.imgsz,
        strict_native=cfg.runtime_seg.strict_native,
    )
    pairwise_labels = HeightNetLoss.load_pairwise_labels(cfg.loss.pairwise_json)

    # KEY FIX: query=test, gallery=test
    result = evaluate(
        model=model,
        query_ds=test_ds,
        gallery_ds=test_ds,
        device=device,
        runtime_depth=runtime_depth,
        segmenter=segmenter,
        pairwise_labels=pairwise_labels,
        cfg=cfg,
        return_records=True,
    )
    summary = result["summary"]
    rankings = build_ranked_candidates(result["query_records"], result["gallery_records"])
    payload = {"summary": summary, "rankings": rankings}
    print(json.dumps(payload, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
