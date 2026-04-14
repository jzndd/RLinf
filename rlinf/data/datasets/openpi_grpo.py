from __future__ import annotations

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

    def _build_sample_index(self) -> list[tuple[int, int]]:
        sample_index: list[tuple[int, int]] = []
        for record in self.episode_records:
            episode_index = int(record["episode_index"])
            episode_length = int(record["length"])
            valid_starts = episode_length - self.action_chunk + 1
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
