#!/usr/bin/env python3
"""Export evaluation success rates and matching offline rewards to CSV.

Usage:
    python toolkits/extract_eval_metrics.py /path/to/logdir

The script expects evaluation runs under ``logdir/eval_logs_l1`` and training
TensorBoard events under ``logdir/tensorboard``. It writes
``logdir/eval_metrics_sorted.csv``.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

EVAL_TAG = "eval/success_once"
OFFLINE_REWARD_TAG = "dataset_offline_eval/reward_mean"
STEP_PATTERN = re.compile(r"_global_step_(\d+)$")
EVENT_PATTERN = "events.out.tfevents.*"
OUTPUT_NAME = "eval_metrics_sorted.csv"


@dataclass(frozen=True)
class EvalResult:
    """A success-rate observation associated with a checkpoint step."""

    step_num: int
    success_rate: float
    wall_time: float
    source_dir: Path


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def _load_scalar_events(event_dir: Path, tag: str):
    """Load all scalar events for a tag from a TensorBoard directory."""
    accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    if tag not in accumulator.Tags().get("scalars", []):
        return []
    return accumulator.Scalars(tag)


def _event_dirs(root: Path) -> list[Path]:
    """Return unique directories containing TensorBoard event files."""
    return sorted({path.parent for path in root.rglob(EVENT_PATTERN)})


def _read_eval_result(eval_dir: Path, step_num: int) -> EvalResult | None:
    """Read the newest eval/success_once value below one evaluation run."""
    observations: list[EvalResult] = []
    for event_dir in _event_dirs(eval_dir):
        try:
            events = _load_scalar_events(event_dir, EVAL_TAG)
        except Exception as exc:  # TensorBoard may encounter a partial event file.
            _warn(f"could not read {event_dir}: {exc}")
            continue
        observations.extend(
            EvalResult(step_num, event.value, event.wall_time, eval_dir)
            for event in events
            if math.isfinite(event.value)
        )

    if not observations:
        _warn(f"{EVAL_TAG} not found below {eval_dir}")
        return None
    return max(observations, key=lambda item: item.wall_time)


def collect_eval_results(eval_root: Path) -> dict[int, EvalResult]:
    """Collect one result per checkpoint step, preferring the newest rerun."""
    results: dict[int, EvalResult] = {}
    for eval_dir in sorted(path for path in eval_root.iterdir() if path.is_dir()):
        match = STEP_PATTERN.search(eval_dir.name)
        if match is None:
            continue
        step_num = int(match.group(1))
        result = _read_eval_result(eval_dir, step_num)
        if result is None:
            continue

        previous = results.get(step_num)
        if previous is None or result.wall_time > previous.wall_time:
            if previous is not None:
                _warn(
                    f"multiple evaluations for step {step_num}; using newer run "
                    f"{result.source_dir.name} instead of {previous.source_dir.name}"
                )
            results[step_num] = result
        else:
            _warn(
                f"multiple evaluations for step {step_num}; keeping newer run "
                f"{previous.source_dir.name} and ignoring {result.source_dir.name}"
            )
    return results


def collect_offline_rewards(training_tensorboard: Path) -> dict[int, float]:
    """Read offline reward means, keeping the newest event at each train step."""
    newest_by_step: dict[int, tuple[float, float]] = {}
    if not training_tensorboard.is_dir():
        _warn(f"training TensorBoard directory does not exist: {training_tensorboard}")
        return {}

    try:
        events = _load_scalar_events(training_tensorboard, OFFLINE_REWARD_TAG)
    except Exception as exc:
        _warn(f"could not read {training_tensorboard}: {exc}")
        return {}

    for event in events:
        if not math.isfinite(event.value):
            continue
        previous = newest_by_step.get(event.step)
        if previous is None or event.wall_time > previous[0]:
            newest_by_step[event.step] = (event.wall_time, event.value)

    if not newest_by_step:
        _warn(f"{OFFLINE_REWARD_TAG} not found in {training_tensorboard}")
    return {step: item[1] for step, item in newest_by_step.items()}


def _reward_for_checkpoint(
    checkpoint_step: int, offline_rewards: dict[int, float]
) -> float | None:
    """Map checkpoint N to the metric logged for training iteration N - 1."""
    train_step = checkpoint_step - 1
    if train_step in offline_rewards:
        return offline_rewards[train_step]

    # A global_step_0 evaluation or logs produced by a runner with a different
    # save/log order may use the checkpoint number directly.
    if checkpoint_step in offline_rewards:
        _warn(
            f"offline reward for checkpoint {checkpoint_step} was found at exact "
            "TensorBoard step instead of checkpoint_step - 1"
        )
        return offline_rewards[checkpoint_step]
    return None


def write_csv(
    output_path: Path,
    eval_results: dict[int, EvalResult],
    offline_rewards: dict[int, float],
) -> None:
    """Write rows sorted by descending success rate, then ascending step."""
    rows = sorted(
        eval_results.values(),
        key=lambda result: (-result.success_rate, result.step_num),
    )
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["step_num", "success_rate", "offline_reward_mean"])
        for result in rows:
            reward = _reward_for_checkpoint(result.step_num, offline_rewards)
            writer.writerow(
                [
                    result.step_num,
                    format(result.success_rate, ".10g"),
                    "" if reward is None else format(reward, ".10g"),
                ]
            )


def parse_args() -> argparse.Namespace:
    """Parse the single required log-directory argument."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "logdir",
        type=Path,
        help="experiment log directory containing eval_logs_l1/ and tensorboard/",
    )
    return parser.parse_args()


def main() -> int:
    """Run the export and return a process exit status."""
    args = parse_args()
    logdir = args.logdir.expanduser().resolve()
    eval_root = logdir / "eval_logs_l1"
    training_tensorboard = logdir / "tensorboard"
    output_path = logdir / OUTPUT_NAME

    if not eval_root.is_dir():
        print(
            f"error: evaluation directory does not exist: {eval_root}", file=sys.stderr
        )
        return 2

    eval_results = collect_eval_results(eval_root)
    if not eval_results:
        print(
            f"error: no {EVAL_TAG} values found below {eval_root}",
            file=sys.stderr,
        )
        return 1

    offline_rewards = collect_offline_rewards(training_tensorboard)
    write_csv(output_path, eval_results, offline_rewards)
    missing_rewards = sum(
        _reward_for_checkpoint(step, offline_rewards) is None for step in eval_results
    )
    print(f"wrote {len(eval_results)} rows to {output_path}")
    if missing_rewards:
        _warn(
            f"offline reward was unavailable for {missing_rewards} evaluation step(s)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
