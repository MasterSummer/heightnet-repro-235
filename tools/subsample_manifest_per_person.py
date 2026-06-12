from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd


def subsample_rows(
    frame: pd.DataFrame,
    target_rows: int,
    seed: int,
    per_person_cap: int = 0,
) -> pd.DataFrame:
    if target_rows <= 0:
        raise ValueError("target_rows must be > 0")
    if "person_id" not in frame.columns:
        raise ValueError("manifest must contain person_id column")

    by_person: dict[str, list[dict]] = defaultdict(list)
    for row in frame.to_dict(orient="records"):
        by_person[str(row["person_id"])].append(row)

    rnd = random.Random(seed)
    people = sorted(by_person)
    for person_id in people:
        rnd.shuffle(by_person[person_id])
        if per_person_cap > 0:
            by_person[person_id] = by_person[person_id][:per_person_cap]

    queues = {person_id: deque(rows) for person_id, rows in by_person.items() if rows}
    ordered_people = people[:]
    rnd.shuffle(ordered_people)

    picked: list[dict] = []
    while len(picked) < target_rows and queues:
        progressed = False
        for person_id in list(ordered_people):
            queue = queues.get(person_id)
            if not queue:
                continue
            picked.append(queue.popleft())
            progressed = True
            if not queue:
                queues.pop(person_id, None)
            if len(picked) >= target_rows:
                break
        if not progressed:
            break
        ordered_people = [person_id for person_id in ordered_people if person_id in queues]

    if not picked:
        raise RuntimeError("subsample result is empty")
    return pd.DataFrame(picked)


def main() -> None:
    parser = argparse.ArgumentParser(description="Balanced manifest subsampling with per-person coverage.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--target-rows", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-person-cap", type=int, default=0)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    out_path = Path(args.out).resolve()
    frame = pd.read_csv(manifest_path)
    sampled = subsample_rows(
        frame=frame,
        target_rows=int(args.target_rows),
        seed=int(args.seed),
        per_person_cap=int(args.per_person_cap),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(out_path, index=False)

    summary = {
        "input_manifest": str(manifest_path),
        "output_manifest": str(out_path),
        "rows_in": int(len(frame)),
        "rows_out": int(len(sampled)),
        "num_people": int(sampled["person_id"].nunique()),
        "per_person_counts": sampled["person_id"].value_counts().sort_index().to_dict(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
