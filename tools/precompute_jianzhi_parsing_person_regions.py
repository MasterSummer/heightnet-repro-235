from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from heightnet.jianzhi_parsing import build_jianzhi_parsing_index, generate_person_region_cache_from_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute .person_bbox.npy and .person_mask.npy from jianzhi parsing JSON bbox records."
    )
    parser.add_argument("--manifest", nargs="+", required=True, help="Extracted frame manifests with frame_path populated.")
    parser.add_argument(
        "--parsing-json-root",
        type=str,
        default="/data2/dataset/jianzhi_2511/jianzhi_spilt_coat_result1/scanner/main_card_out",
    )
    parser.add_argument(
        "--parsing-bitmap-root",
        type=str,
        default="/data2/dataset/jianzhi_2511/jianzhi_spilt_coat_result1/parsing",
        help="Directory containing parsing/<video_stem>/<track_id>/<frame_idx>.bmp crops. Missing crops are skipped.",
    )
    parser.add_argument("--out-report", type=str, required=True)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()

    rows = []
    for manifest in args.manifest:
        frame = pd.read_csv(manifest)
        rows.extend(frame.to_dict(orient="records"))
    include_video_stems = {
        str(row.get("sequence_id", "")).split("__", 1)[1]
        for row in rows
        if "__" in str(row.get("sequence_id", ""))
    }
    index = build_jianzhi_parsing_index(
        args.parsing_json_root,
        parsing_bitmap_root=args.parsing_bitmap_root,
        include_video_stems=include_video_stems or None,
    )

    report = generate_person_region_cache_from_index(
        rows,
        index,
        overwrite_existing=bool(args.overwrite_existing),
    )
    report["parsing_json_root"] = str(Path(args.parsing_json_root).expanduser().resolve())
    report["parsing_bitmap_root"] = str(Path(args.parsing_bitmap_root).expanduser().resolve()) if args.parsing_bitmap_root else ""
    report["num_index_records"] = len(index.records)
    report["num_index_video_stems"] = len(include_video_stems)
    report["manifests"] = [str(Path(x).expanduser().resolve()) for x in args.manifest]

    out_report = Path(args.out_report).expanduser().resolve()
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with out_report.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(
        "[JIANZHI_PERSON_CACHE] "
        f"total={report['total']} generated={report['generated']} "
        f"skipped_existing={report['skipped_existing']} missing_record={report['missing_record']} "
        f"missing_parsing={report['missing_parsing']} invalid_bbox={report['invalid_bbox']} empty_mask={report['empty_mask']}"
    )
    print(f"[REPORT] {out_report}")


if __name__ == "__main__":
    main()
