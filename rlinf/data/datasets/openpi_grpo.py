from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class OpenPIGRPOItem:
    obs: dict[str, Any]
    actions: torch.Tensor
    idx: int
    episode_index: int
    frame_index: int


class OpenPIGRPODataset(Dataset[OpenPIGRPOItem]):
    """Build offline GRPO samples from a LeRobot/OpenPI episode dataset.

    Each item uses one observation frame as the prompt-equivalent input and a
    fixed-length future action chunk as the supervision target.
    """

    def __init__(
        self,
        data_paths: str | list[str],
        config: DictConfig,
    ) -> None:
        super().__init__()

        raw_paths = [data_paths] if isinstance(data_paths, str) else list(data_paths)
        assert len(raw_paths) == 1, (
            "OpenPIGRPODataset expects exactly one dataset root path. "
            f"Got {raw_paths}."
        )

        self.cfg = config
        self.dataset_root = Path(raw_paths[0]).expanduser().resolve()
        assert self.dataset_root.exists(), (
            f"Dataset root does not exist: {self.dataset_root}"
        )

        self.action_chunk = int(self.cfg.data.action_chunk)
        assert self.action_chunk > 0, "data.action_chunk must be greater than 0"

        self.image_key = str(self.cfg.data.get("image_key", "image"))
        self.wrist_image_key = str(
            self.cfg.data.get("wrist_image_key", "wrist_image")
        )
        self.state_key = str(self.cfg.data.get("state_key", "state"))
        self.action_key = str(self.cfg.data.get("action_key", "actions"))
        self.task_index_key = str(self.cfg.data.get("task_index_key", "task_index"))
        self.frame_index_key = str(
            self.cfg.data.get("frame_index_key", "frame_index")
        )
        self.max_cache_episodes = int(self.cfg.data.get("max_cache_episodes", 4))
        assert self.max_cache_episodes >= 1, (
            "data.max_cache_episodes must be greater than or equal to 1"
        )

        self.meta_dir = self.dataset_root / "meta"
        self.data_dir = self.dataset_root / "data"
        assert self.meta_dir.exists(), f"Meta directory not found: {self.meta_dir}"
        assert self.data_dir.exists(), f"Data directory not found: {self.data_dir}"

        self.info = self._load_json(self.meta_dir / "info.json")
        self.data_path_template = str(self.info["data_path"])
        self.chunk_size = int(self.info["chunks_size"])

        self.task_text_by_index = self._load_tasks(self.meta_dir / "tasks.jsonl")
        self.task_index_by_text = {
            task_text: task_index for task_index, task_text in self.task_text_by_index.items()
        }
        self.episode_records = self._load_jsonl(self.meta_dir / "episodes.jsonl")
        self.sample_index = self._build_sample_index()
        self.episode_cache: OrderedDict[int, pd.DataFrame] = OrderedDict()

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> OpenPIGRPOItem:
        episode_index, start_frame = self.sample_index[idx]
        episode_df = self._load_episode_dataframe(episode_index)

        current_row = episode_df.iloc[start_frame]
        action_window = episode_df.iloc[start_frame : start_frame + self.action_chunk]

        task_index = int(current_row[self.task_index_key])
        task_description = self.task_text_by_index[task_index]

        obs = {
            "main_images": self._decode_image(current_row[self.image_key]),
            "wrist_images": self._decode_image(current_row[self.wrist_image_key]),
            "extra_view_images": None,
            "states": torch.from_numpy(
                np.asarray(current_row[self.state_key], dtype=np.float32)
            ),
            "task_descriptions": task_description,
        }
        actions = torch.from_numpy(
            np.stack(action_window[self.action_key].tolist(), axis=0).astype(np.float32)
        )

        return OpenPIGRPOItem(
            obs=obs,
            actions=actions,
            idx=idx,
            episode_index=episode_index,
            frame_index=int(current_row[self.frame_index_key]),
        )

    def _load_json(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_jsonl(self, path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
        return records

    def _load_tasks(self, path: Path) -> dict[int, str]:
        tasks: dict[int, str] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                tasks[int(record["task_index"])] = str(record["task"])
        return tasks

    def _task_index_from_episode_record(self, record: dict[str, Any]) -> int:
        if self.task_index_key in record:
            return int(record[self.task_index_key])

        task_descs = record.get("tasks", [])
        if not task_descs:
            raise KeyError(
                f"Episode record {record.get('episode_index')} does not contain "
                f"'{self.task_index_key}' or non-empty 'tasks'."
            )

        task_desc = str(task_descs[0])
        if task_desc not in self.task_index_by_text:
            raise KeyError(
                f"Task description from episode record is not in tasks.jsonl: {task_desc}"
            )
        return int(self.task_index_by_text[task_desc])

    def _build_sample_index(self) -> list[tuple[int, int]]:
        sample_index: list[tuple[int, int]] = []
        for record in self.episode_records:
            episode_index = int(record["episode_index"])
            episode_length = int(record["length"])
            required_frames = self.action_chunk
            valid_starts = episode_length - required_frames + 1
            if valid_starts <= 0:
                continue
            for start_frame in range(valid_starts):
                sample_index.append((episode_index, start_frame))
        assert sample_index, "OpenPIGRPODataset built zero valid samples."
        return sample_index

    def _episode_path(self, episode_index: int) -> Path:
        relative_path = self.data_path_template.format(
            episode_chunk=episode_index // self.chunk_size,
            episode_index=episode_index,
        )
        episode_path = self.dataset_root / relative_path
        assert episode_path.exists(), f"Episode parquet not found: {episode_path}"
        return episode_path

    def _load_episode_dataframe(self, episode_index: int) -> pd.DataFrame:
        if episode_index in self.episode_cache:
            self.episode_cache.move_to_end(episode_index)
            return self.episode_cache[episode_index]

        episode_df = pd.read_parquet(self._episode_path(episode_index))
        self.episode_cache[episode_index] = episode_df
        while len(self.episode_cache) > self.max_cache_episodes:
            self.episode_cache.popitem(last=False)
        return episode_df

    def _decode_image(self, image_record: dict[str, Any]) -> torch.Tensor:
        image = Image.open(BytesIO(image_record["bytes"]))
        image = image.convert("RGB")
        array = np.array(image, dtype=np.uint8, copy=True)
        return torch.from_numpy(array)


class ContinualLearningOpenPIGRPODataset(OpenPIGRPODataset):
    """Deterministic task-subset view over one full OpenPI/LeRobot dataset."""

    def __init__(
        self,
        data_paths: str | list[str],
        config: DictConfig,
    ) -> None:
        super().__init__(data_paths, config)

        subset_id = int(self.cfg.data.get("continual_subset_id", 1))
        initial_task_count = int(self.cfg.data.get("continual_initial_task_count", 6))
        new_task_traj = int(self.cfg.data.get("new_task_traj", 10))
        old_task_traj = int(self.cfg.data.get("old_task_traj", 5))
        replay_seed = int(self.cfg.data.get("continual_replay_seed", 0))
        eval_all_task_episodes = bool(
            self.cfg.data.get("continual_eval_full_dataset", False)
        )

        (
            self.episode_records,
            self.continual_selected_episodes,
            self.continual_selection_summary,
        ) = self.select_episodes(
            records=self.episode_records,
            task_text_by_index=self.task_text_by_index,
            task_index_key=self.task_index_key,
            subset_id=subset_id,
            initial_task_count=initial_task_count,
            new_task_traj=new_task_traj,
            old_task_traj=old_task_traj,
            replay_seed=replay_seed,
            eval_all_task_episodes=eval_all_task_episodes,
        )
        self.sample_index = self._build_sample_index()

    @classmethod
    def select_episodes(
        cls,
        *,
        records: list[dict[str, Any]],
        task_text_by_index: dict[int, str],
        task_index_key: str,
        subset_id: int,
        initial_task_count: int,
        new_task_traj: int,
        old_task_traj: int,
        replay_seed: int,
        eval_all_task_episodes: bool = False,
    ) -> tuple[list[dict[str, Any]], list[int], dict[int, int]]:
        assert subset_id >= 1, "data.continual_subset_id must be >= 1"
        assert initial_task_count >= 1, (
            "data.continual_initial_task_count must be >= 1"
        )
        assert new_task_traj >= 0, "data.new_task_traj must be >= 0"
        assert old_task_traj >= 0, "data.old_task_traj must be >= 0"

        task_index_by_text = {
            task_text: task_index for task_index, task_text in task_text_by_index.items()
        }

        def task_index_from_record(record: dict[str, Any]) -> int:
            if task_index_key in record:
                return int(record[task_index_key])

            task_descs = record.get("tasks", [])
            if not task_descs:
                raise KeyError(
                    f"Episode record {record.get('episode_index')} does not contain "
                    f"'{task_index_key}' or non-empty 'tasks'."
                )

            task_desc = str(task_descs[0])
            if task_desc not in task_index_by_text:
                raise KeyError(
                    "Task description from episode record is not in tasks.jsonl: "
                    f"{task_desc}"
                )
            return int(task_index_by_text[task_desc])

        if subset_id == 1:
            new_task_ids = set(range(initial_task_count))
            required_task_ids = list(range(initial_task_count))
        else:
            current_task_id = initial_task_count + subset_id - 2
            new_task_ids = {current_task_id}
            required_task_ids = list(range(current_task_id + 1))

        records_by_task: dict[int, list[dict[str, Any]]] = {
            task_id: [] for task_id in required_task_ids
        }
        for record in records:
            task_id = task_index_from_record(record)
            if task_id in records_by_task:
                records_by_task[task_id].append(record)

        selected_records: list[dict[str, Any]] = []
        selection_summary: dict[int, int] = {}
        for task_id in required_task_ids:
            task_records = sorted(
                records_by_task.get(task_id, []),
                key=lambda record: int(record["episode_index"]),
            )
            if not task_records:
                continue

            if eval_all_task_episodes:
                traj_limit = len(task_records)
            else:
                traj_limit = new_task_traj if task_id in new_task_ids else old_task_traj
            if traj_limit <= 0:
                continue

            if len(task_records) > traj_limit:
                task_records = sorted(
                    task_records,
                    key=lambda record: cls._stable_replay_key(
                        replay_seed=replay_seed,
                        task_index=task_id,
                        episode_index=int(record["episode_index"]),
                    ),
                )[:traj_limit]
                task_records = sorted(
                    task_records,
                    key=lambda record: int(record["episode_index"]),
                )

            selected_records.extend(task_records)
            selection_summary[task_id] = len(task_records)

        assert selected_records, (
            "Continual selection produced zero episodes. "
            f"subset_id={subset_id}, initial_task_count={initial_task_count}, "
            f"new_task_traj={new_task_traj}, old_task_traj={old_task_traj}"
        )
        selected_episode_ids = [
            int(record["episode_index"]) for record in selected_records
        ]
        return selected_records, selected_episode_ids, selection_summary

    @staticmethod
    def _stable_replay_key(
        *,
        replay_seed: int,
        task_index: int,
        episode_index: int,
    ) -> tuple[str, int]:
        key = f"{replay_seed}:{task_index}:{episode_index}".encode("utf-8")
        return hashlib.sha256(key).hexdigest(), episode_index


def openpi_grpo_collate_fn(
    data_list: list[OpenPIGRPOItem],
) -> dict[str, Any]:
    assert data_list, "openpi_grpo_collate_fn received an empty batch"

    main_images = torch.stack([item.obs["main_images"] for item in data_list], dim=0)
    wrist_images = torch.stack(
        [item.obs["wrist_images"] for item in data_list], dim=0
    )
    states = torch.stack([item.obs["states"] for item in data_list], dim=0)
    actions = torch.stack([item.actions for item in data_list], dim=0)

    return {
        "obs": {
            "main_images": main_images,
            "wrist_images": wrist_images,
            "extra_view_images": None,
            "states": states,
            "task_descriptions": [
                item.obs["task_descriptions"] for item in data_list
            ],
        },
        "actions": actions,
        "idx": torch.tensor([item.idx for item in data_list], dtype=torch.long),
        "episode_index": torch.tensor(
            [item.episode_index for item in data_list], dtype=torch.long
        ),
        "frame_index": torch.tensor(
            [item.frame_index for item in data_list], dtype=torch.long
        ),
    }
