#!/usr/bin/env bash

set -euo pipefail

EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$(dirname "${EMBODIED_PATH}")")"

CHAIN_ROOT="${CHAIN_ROOT:-${REPO_PATH}/logs_goal/libero_goal_offline_grpo_openpi_pi05_flow_noise_subset_chain-step2-cosinelr-subsetaware-base90_oneshot_new10_old5}"
SUBSET_ID="${SUBSET_ID:-5}"
SUBSET_NAME="subset${SUBSET_ID}"
EXP_NAME="libero_goal_offline_grpo_openpi_pi05_flow_noise_subset${SUBSET_ID}"
SUBSET_LOG_DIR="${CHAIN_ROOT}/${EXP_NAME}"
CKPT_ROOT="${SUBSET_LOG_DIR}/${EXP_NAME}/checkpoints"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${SUBSET_LOG_DIR}/tensorboard}"

TOP_K="${TOP_K:-6}"
WORLD_SIZE="${WORLD_SIZE:-8}"
DENOISE_STEP="${DENOISE_STEP:-2}"
CONFIG_NAME="${CONFIG_NAME:-libero_goal_grpo_openpi_pi05_noise_base90_oneshot}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${EMBODIED_PATH}/eval_libero_goal_subset_tasks_multi_ckpt.sh}"
RESULT_ROOT="${RESULT_ROOT:-${SUBSET_LOG_DIR}/eval_top${TOP_K}_from_tensorboard}"
RUN_ID="${RUN_ID:-$(date +'%Y%m%d-%H%M%S')}"
ONLINE_EVAL_DIR="${ONLINE_EVAL_DIR:-${RESULT_ROOT}/online_eval_${RUN_ID}}"
TOP_REWARD_CSV="${TOP_REWARD_CSV:-${RESULT_ROOT}/top${TOP_K}_dataset_offline_eval_reward_mean.csv}"
SUCCESS_CSV="${SUCCESS_CSV:-${RESULT_ROOT}/online_eval_success_once.csv}"
SELECT_ONLY="${SELECT_ONLY:-false}"

if [[ -f /opt/venv/openpi/bin/activate ]]; then
    source /opt/venv/openpi/bin/activate
fi

export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"

mkdir -p "${RESULT_ROOT}"

select_top_reward_checkpoints() {
    TOP_K="${TOP_K}" TENSORBOARD_DIR="${TENSORBOARD_DIR}" CKPT_ROOT="${CKPT_ROOT}" TOP_REWARD_CSV="${TOP_REWARD_CSV}" python - <<'PY'
import csv
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from tensorboard.backend.event_processing import event_accumulator

top_k = int(os.environ["TOP_K"])
tb_dir = Path(os.environ["TENSORBOARD_DIR"])
ckpt_root = Path(os.environ["CKPT_ROOT"])
output_csv = Path(os.environ["TOP_REWARD_CSV"])
tag = "dataset_offline_eval/reward_mean"

if not tb_dir.exists():
    raise SystemExit(f"tensorboard dir not found: {tb_dir}")
if not ckpt_root.exists():
    raise SystemExit(f"checkpoint root not found: {ckpt_root}")

ea = event_accumulator.EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
ea.Reload()
if tag not in ea.Tags().get("scalars", []):
    raise SystemExit(f"missing scalar tag: {tag}")

rows = []
for event in ea.Scalars(tag):
    # Existing LIBERO offline eval events are logged at step N-1 for checkpoint global_step_N.
    for global_step in (int(event.step) + 1, int(event.step)):
        ckpt_file = ckpt_root / f"global_step_{global_step}" / "actor" / "model_state_dict" / "full_weights.pt"
        if ckpt_file.exists():
            rows.append(
                {
                    "global_step": global_step,
                    "event_step": int(event.step),
                    "dataset_offline_eval_reward_mean": float(event.value),
                    "checkpoint_path": str(ckpt_file),
                }
            )
            break

if len(rows) < top_k:
    raise SystemExit(f"only {len(rows)} reward/checkpoint pairs found, need {top_k}")

deduped = {}
for row in rows:
    step = row["global_step"]
    prev = deduped.get(step)
    if prev is None or row["dataset_offline_eval_reward_mean"] > prev["dataset_offline_eval_reward_mean"]:
        deduped[step] = row

ranked = sorted(
    deduped.values(),
    key=lambda row: (row["dataset_offline_eval_reward_mean"], row["global_step"]),
    reverse=True,
)[:top_k]

output_csv.parent.mkdir(parents=True, exist_ok=True)
with output_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "global_step",
            "event_step",
            "dataset_offline_eval_reward_mean",
            "checkpoint_path",
        ],
    )
    writer.writeheader()
    for row in ranked:
        writer.writerow(
            {
                "global_step": row["global_step"],
                "event_step": row["event_step"],
                "dataset_offline_eval_reward_mean": f"{row['dataset_offline_eval_reward_mean']:.10f}",
                "checkpoint_path": row["checkpoint_path"],
            }
        )
PY
}

csv_column() {
    local csv_path="$1"
    local column_name="$2"
    python - <<'PY' "${csv_path}" "${column_name}"
import csv
import sys

csv_path, column_name = sys.argv[1], sys.argv[2]
with open(csv_path, "r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        print(row[column_name])
PY
}

parse_online_eval_success() {
    ONLINE_EVAL_DIR="${ONLINE_EVAL_DIR}" TOP_REWARD_CSV="${TOP_REWARD_CSV}" SUCCESS_CSV="${SUCCESS_CSV}" python - <<'PY'
import csv
import os
import re
from pathlib import Path

online_eval_dir = Path(os.environ["ONLINE_EVAL_DIR"])
top_reward_csv = Path(os.environ["TOP_REWARD_CSV"])
success_csv = Path(os.environ["SUCCESS_CSV"])

reward_by_step = {}
ckpt_by_step = {}
with top_reward_csv.open("r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        step = int(row["global_step"])
        reward_by_step[step] = float(row["dataset_offline_eval_reward_mean"])
        ckpt_by_step[step] = row["checkpoint_path"]

number = r"([0-9eE+.\-]+)"
optional_array = r"(?:array\(\s*)?"
patterns = [
    re.compile(r"(?:'|\")eval/success_once(?:'|\")\s*:\s*" + optional_array + number),
    re.compile(r"eval/success_once\s*[=:]\s*" + optional_array + number),
    re.compile(r"success_once\s*[=:]\s*" + optional_array + number),
]

rows = []
for step, reward in reward_by_step.items():
    logs = []
    logs.extend(online_eval_dir.glob(f"*global_step_{step}*/eval_embodiment.log"))
    logs.extend(online_eval_dir.glob(f"*global_step_{step}*/**/all_tasks/eval_embodiment.log"))
    logs.extend(online_eval_dir.glob(f"**/*global_step_{step}*/eval_embodiment.log"))
    logs.extend(online_eval_dir.glob(f"**/*global_step_{step}*/**/eval_embodiment.log"))
    logs = sorted(set(logs))
    if not logs:
        raise SystemExit(f"missing online eval log for global_step_{step} under {online_eval_dir}")

    text = logs[-1].read_text(encoding="utf-8", errors="ignore")
    matches = []
    for pattern in patterns:
        matches = pattern.findall(text)
        if matches:
            break
    if not matches:
        raise SystemExit(f"missing success_once in {logs[-1]}")
    rows.append(
        {
            "global_step": step,
            "dataset_offline_eval_reward_mean": reward,
            "success_once": float(matches[-1]),
            "checkpoint_path": ckpt_by_step[step],
            "eval_log": str(logs[-1]),
        }
    )

ranked = sorted(
    rows,
    key=lambda row: (
        row["success_once"],
        row["dataset_offline_eval_reward_mean"],
        row["global_step"],
    ),
    reverse=True,
)

success_csv.parent.mkdir(parents=True, exist_ok=True)
with success_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "global_step",
            "dataset_offline_eval_reward_mean",
            "success_once",
            "checkpoint_path",
            "eval_log",
        ],
    )
    writer.writeheader()
    for row in ranked:
        writer.writerow(
            {
                "global_step": row["global_step"],
                "dataset_offline_eval_reward_mean": f"{row['dataset_offline_eval_reward_mean']:.10f}",
                "success_once": f"{row['success_once']:.10f}",
                "checkpoint_path": row["checkpoint_path"],
                "eval_log": row["eval_log"],
            }
        )
PY
}

echo "Selecting top ${TOP_K} checkpoints from ${TENSORBOARD_DIR}"
select_top_reward_checkpoints
echo "Top checkpoint CSV saved to: ${TOP_REWARD_CSV}"
cat "${TOP_REWARD_CSV}"

if [[ "${SELECT_ONLY}" == "true" || "${SELECT_ONLY}" == "1" ]]; then
    echo "SELECT_ONLY=${SELECT_ONLY}; skip online eval."
    exit 0
fi

mapfile -t top_ckpts < <(csv_column "${TOP_REWARD_CSV}" checkpoint_path)
if [[ "${#top_ckpts[@]}" -ne "${TOP_K}" ]]; then
    echo "Expected ${TOP_K} checkpoints, got ${#top_ckpts[@]}." >&2
    exit 1
fi

mkdir -p "${ONLINE_EVAL_DIR}"
echo "Running online eval for ${SUBSET_NAME}; logs under: ${ONLINE_EVAL_DIR}"
bash "${EVAL_SCRIPT}" \
    "${SUBSET_NAME}" \
    "${CONFIG_NAME}" \
    "${WORLD_SIZE}" \
    "${ONLINE_EVAL_DIR}" \
    "${DENOISE_STEP}" \
    "${top_ckpts[@]}"

echo "Parsing success_once into CSV: ${SUCCESS_CSV}"
parse_online_eval_success
cat "${SUCCESS_CSV}"
