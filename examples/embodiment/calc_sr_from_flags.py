#!/usr/bin/env python3

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_target_ids(args: argparse.Namespace) -> list[int]:
    if args.target_ids_json:
        payload = json.loads(args.target_ids_json)
        if isinstance(payload, int):
            return [int(payload)]
        return [int(x) for x in payload]
    if args.target_ids:
        return [int(x.strip()) for x in args.target_ids.split(",") if x.strip()]
    if args.target_start is not None and args.target_end is not None:
        if args.target_end < args.target_start:
            raise ValueError("target_end must be >= target_start")
        return list(range(args.target_start, args.target_end + 1))
    raise ValueError(
        "Please provide target ids via --target-ids-json, --target-ids, or --target-start/--target-end."
    )


def load_flags(csv_path: Path) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"reset_state_id", "success_flag"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"CSV must contain columns {required}, got {reader.fieldnames}."
            )
        for row in reader:
            rid = int(row["reset_state_id"])
            success = int(row["success_flag"])
            rows.append((rid, success))
    return rows


def compute_sr(rows: list[tuple[int, int]], target_ids: list[int]) -> tuple[float, int, int]:
    grouped = defaultdict(list)
    for rid, success in rows:
        grouped[rid].append(float(success))

    per_id_success = []
    missing = 0
    for rid in target_ids:
        if rid in grouped:
            # Aggressive mode: if any repeated trial succeeds, mark this id as success.
            per_id_success.append(max(grouped[rid]))
        else:
            missing += 1

    if not per_id_success:
        raise ValueError("No target ids were found in CSV.")
    return sum(per_id_success) / len(per_id_success), len(per_id_success), missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute SR from per-episode flags CSV."
    )
    parser.add_argument("--csv", required=True, help="Path to per_episode_flags.csv")
    parser.add_argument(
        "--target-ids-json",
        default=None,
        help="JSON list/int of target reset_state_ids, e.g. '[200,201,...]'.",
    )
    parser.add_argument(
        "--target-ids",
        default=None,
        help="Comma-separated target reset_state_ids, e.g. '200,201,...'.",
    )
    parser.add_argument(
        "--target-start",
        type=int,
        default=None,
        help="Start id (inclusive) if using range mode.",
    )
    parser.add_argument(
        "--target-end",
        type=int,
        default=None,
        help="End id (inclusive) if using range mode.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    target_ids = parse_target_ids(args)
    rows = load_flags(csv_path)
    sr, matched, missing = compute_sr(rows, target_ids)

    print(f"sr={sr:.8f}")
    print(f"matched_target_ids={matched}")
    print(f"missing_target_ids={missing}")
    print(f"total_target_ids={len(target_ids)}")


if __name__ == "__main__":
    main()
