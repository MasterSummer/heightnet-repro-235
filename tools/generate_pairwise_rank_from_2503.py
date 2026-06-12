from __future__ import annotations

import argparse
import json
import os


def ranking_to_pairs(ranking: list[str]) -> list[dict]:
    pairs = []
    for i in range(len(ranking)):
        for j in range(i + 1, len(ranking)):
            pairs.append({"id_i": ranking[i], "id_j": ranking[j], "y": 1})
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank-dir", type=str, default="/Users/yiding/code/gait/2503_test_rank")
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted([x for x in os.listdir(args.rank_dir) if x.endswith(".json")])
    if not files:
        raise RuntimeError(f"No json files in {args.rank_dir}")

    all_pairs = []
    for name in files:
        p = os.path.join(args.rank_dir, name)
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)

        camera = obj.get("camera")
        ranking = obj.get("ranking", [])
        if not camera or not isinstance(ranking, list):
            continue

        pairs = ranking_to_pairs(ranking)
        for item in pairs:
            item["camera"] = camera
        all_pairs.extend(pairs)

        out_file = os.path.join(args.out_dir, f"camera_{camera}_pairs.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.out_dir, "all_pairs.json"), "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_pairs)} pairwise labels to {args.out_dir}")


if __name__ == "__main__":
    main()
