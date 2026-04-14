# Offline OpenPI GRPO Dataset

## Scope

`OpenPIGRPODataset` turns a LeRobot/OpenPI episode dataset into offline GRPO samples.

## Input layout

- Dataset root: `<root>/meta` and `<root>/data`
- Metadata files used:
  - `meta/info.json`
  - `meta/tasks.jsonl`
  - `meta/episodes.jsonl`
- Episode payloads used:
  - `data/chunk-*/episode_*.parquet`

## Sample contract

Each sample contains:

- `obs.main_images`: current frame RGB image in `HWC uint8`
- `obs.wrist_images`: current wrist RGB image in `HWC uint8`
- `obs.states`: current proprio state in `float32[8]`
- `obs.task_descriptions`: task string looked up from `task_index`
- `actions`: future action chunk in `float32[action_chunk, action_dim]`

## Chunk rule

- Sample index is `(episode_index, start_frame)`.
