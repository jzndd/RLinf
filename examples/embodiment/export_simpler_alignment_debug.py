#!/usr/bin/env python3
"""Export Bridge training samples for SimplerEnv train/eval alignment checks."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_EVAL_PROMPTS = {
    "widowx_spoon_on_towel": "put the spoon on the towel",
    "widowx_carrot_on_plate": "put carrot on plate",
    "widowx_stack_cube": "stack the green block on the yellow block",
    "widowx_put_eggplant_in_basket": "put eggplant into yellow basket",
}

DEFAULT_QUERY_GROUPS = {
    "widowx_spoon_on_towel": ["spoon towel"],
    "widowx_carrot_on_plate": ["carrot plate"],
    "widowx_stack_cube": ["green cube yellow cube", "green block yellow block"],
    "widowx_put_eggplant_in_basket": ["eggplant basket"],
}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _slug(text: str) -> str:
    keep = []
    for char in text.lower():
        if char.isalnum():
            keep.append(char)
        elif keep and keep[-1] != "_":
            keep.append("_")
    return "".join(keep).strip("_")[:120] or "empty"


def _matches(task: str, query: str) -> bool:
    task_lower = task.lower()
    return all(token in task_lower for token in query.lower().split())


def _save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def _extract_frames_with_imageio(video_path: Path, output_dir: Path, frame_indices: set[int]) -> list[Path]:
    import imageio.v3 as iio

    saved = []
    for idx, frame in enumerate(iio.imiter(video_path)):
        if idx in frame_indices:
            out = output_dir / f"frame_{idx:04d}.png"
            _save_png(out, frame)
            saved.append(out)
        if idx > max(frame_indices):
            break
    return saved


def _extract_frames_with_cv2(video_path: Path, output_dir: Path, frame_indices: set[int]) -> list[Path]:
    import cv2

    saved = []
    cap = cv2.VideoCapture(str(video_path))
    try:
        idx = 0
        while cap.isOpened():
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if idx in frame_indices:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                out = output_dir / f"frame_{idx:04d}.png"
                _save_png(out, frame_rgb)
                saved.append(out)
            if idx > max(frame_indices):
                break
            idx += 1
    finally:
        cap.release()
    return saved


def _extract_frames_with_ffmpeg(video_path: Path, output_dir: Path, frame_indices: set[int]) -> list[Path]:
    saved = []
    for idx in sorted(frame_indices):
        out = output_dir / f"frame_{idx:04d}.png"
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"select=eq(n\\,{idx})",
            "-vframes",
            "1",
            str(out),
        ]
        subprocess.run(cmd, check=True)
        if out.exists():
            saved.append(out)
    return saved


def extract_frames(video_path: Path, output_dir: Path, episode_length: int, frames_per_episode: int) -> tuple[list[Path], str | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_count = max(1, min(frames_per_episode, episode_length))
    frame_indices = set(np.linspace(0, max(0, episode_length - 1), frame_count, dtype=int).tolist())
    errors = []
    for name, fn in [
        ("imageio", _extract_frames_with_imageio),
        ("cv2", _extract_frames_with_cv2),
        ("ffmpeg", _extract_frames_with_ffmpeg),
    ]:
        try:
            saved = fn(video_path, output_dir, frame_indices)
            if saved:
                return saved, None
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    return [], "; ".join(errors)


def read_parquet_preview(parquet_path: Path, rows: int) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(parquet_path)
        columns = table.column_names
        preview_columns = [col for col in ["observation.state", "action", "timestamp", "frame_index", "episode_index", "task_index"] if col in columns]
        preview = pq.read_table(parquet_path, columns=preview_columns).slice(0, rows).to_pydict()
        return {"columns": columns, "preview": preview}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def select_episodes(
    episodes: list[dict[str, Any]],
    query_groups: dict[str, list[str]],
    episodes_per_query: int,
) -> dict[str, list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = {}
    for task_name, queries in query_groups.items():
        matches = []
        for episode in episodes:
            task_text = " ".join(episode.get("tasks", []))
            if any(_matches(task_text, query) for query in queries):
                matches.append(episode)
        selected[task_name] = matches[:episodes_per_query]
    return selected


def prompt_match_report(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    task_texts = [" ".join(episode.get("tasks", [])) for episode in episodes]
    report = {}
    for task_name, eval_prompt in DEFAULT_EVAL_PROMPTS.items():
        queries = DEFAULT_QUERY_GROUPS.get(task_name, [])
        exact_matches = [text for text in task_texts if text == eval_prompt]
        query_matches = [
            text
            for text in task_texts
            if any(_matches(text, query) for query in queries)
        ]
        report[task_name] = {
            "eval_prompt": eval_prompt,
            "exact_match_count": len(exact_matches),
            "query_match_count": len(query_matches),
            "queries": queries,
            "exact_examples": exact_matches[:8],
            "query_examples": query_matches[:8],
        }
    return report


def video_path_for_episode(dataset_root: Path, image_key: str, episode_index: int) -> Path:
    chunk = episode_index // 1000
    return dataset_root / "videos" / f"chunk-{chunk:03d}" / image_key / f"episode_{episode_index:06d}.mp4"


def parquet_path_for_episode(dataset_root: Path, episode_index: int) -> Path:
    chunk = episode_index // 1000
    return dataset_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def export_selected_samples(
    dataset_root: Path,
    output_dir: Path,
    image_key: str,
    selected: dict[str, list[dict[str, Any]]],
    frames_per_episode: int,
    copy_video: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "image_key": image_key,
        "eval_prompts": DEFAULT_EVAL_PROMPTS,
        "tasks": {},
    }
    for task_name, episodes in selected.items():
        task_dir = output_dir / "train" / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        task_records = []
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            episode_dir = task_dir / f"episode_{episode_index:06d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_path_for_episode(dataset_root, image_key, episode_index)
            parquet_path = parquet_path_for_episode(dataset_root, episode_index)

            copied_video = None
            if copy_video and video_path.exists():
                copied_video = episode_dir / video_path.name
                shutil.copy2(video_path, copied_video)

            frame_paths = []
            frame_error = None
            if video_path.exists():
                frame_paths, frame_error = extract_frames(video_path, episode_dir / "frames", int(episode["length"]), frames_per_episode)

            metadata = {
                "episode": episode,
                "training_prompt": " ".join(episode.get("tasks", [])),
                "eval_default_prompt": DEFAULT_EVAL_PROMPTS.get(task_name),
                "video_path": video_path,
                "copied_video": copied_video,
                "parquet_path": parquet_path,
                "frame_paths": frame_paths,
                "frame_error": frame_error,
                "parquet": read_parquet_preview(parquet_path, rows=5) if parquet_path.exists() else {"error": "parquet missing"},
            }
            (episode_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True, default=_json_default) + "\n",
                encoding="utf-8",
            )
            task_records.append(metadata)
        report["tasks"][task_name] = {
            "num_selected": len(episodes),
            "selected_prompts": [" ".join(ep.get("tasks", [])) for ep in episodes],
            "episodes": task_records,
        }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/mnt/project_rlinf/jzn/workspace/openpi/data/bridge_orig_lerobot"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--image-key", default="observation.images.image_0")
    parser.add_argument("--episodes-per-query", type=int, default=4)
    parser.add_argument("--frames-per-episode", type=int, default=8)
    parser.add_argument("--no-copy-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("/mnt/project_rlinf/jzn/workspace/continual_learning/RLinf/logs/simpler_alignment_debug") / timestamp
    args.output_dir.mkdir(parents=True, exist_ok=True)

    episodes = _read_jsonl(args.dataset_root / "meta" / "episodes.jsonl")
    selected = select_episodes(episodes, DEFAULT_QUERY_GROUPS, args.episodes_per_query)
    report = export_selected_samples(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        image_key=args.image_key,
        selected=selected,
        frames_per_episode=args.frames_per_episode,
        copy_video=not args.no_copy_video,
    )
    report["prompt_match_report"] = prompt_match_report(episodes)
    (args.output_dir / "prompt_image_alignment_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "tasks": {k: v["num_selected"] for k, v in report["tasks"].items()}}, indent=2))


if __name__ == "__main__":
    main()
