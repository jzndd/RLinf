#!/usr/bin/env bash
set -euo pipefail

RLINF_ROOT="${RLINF_ROOT:-/mnt/project_rlinf/jzn/workspace/continual_learning/RLinf}"
OPENPI_ROOT="${OPENPI_ROOT:-/mnt/project_rlinf/jzn/workspace/openpi}"
OPENPI_PYTHON="${OPENPI_PYTHON:-/opt/venv/openpi/bin/python}"
SIMPLERENV_PATH="${SIMPLERENV_PATH:-/mnt/project_rlinf/jzn/workspace/third_party/SimplerEnv}"
SIMPLER_PYTHON="${SIMPLER_PYTHON:-/mnt/project_rlinf/jzn/conda_envs/simpler_env/bin/python}"
MODEL_PATH="${MODEL_PATH:-${OPENPI_ROOT}/checkpoints/sft/pi05_simpler_google_robot_rt1/simpler_google_robot_rt1_full_sft/55000}"
DETECT_SCRIPT="${DETECT_SCRIPT:-/mnt/project_rlinf/jzn/workspace/detect.py}"

PORT="${PORT:-21000}"
EVAL_GPUS="${EVAL_GPUS:-${GPUS:-0,1,2,3,4,5,6,7}}"
NUM_EVAL_WORKERS="${NUM_EVAL_WORKERS:-}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-100}"
ACTION_CHUNK="${ACTION_CHUNK:-4}"
SEED="${SEED:-2022}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-}"
SAVE_VIDEO="${SAVE_VIDEO:-0}"
RUN_DETECT="${RUN_DETECT:-0}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-1800}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${RLINF_ROOT}/logs/simpler_google_robot_visual_matching/${TIMESTAMP}}"
mkdir -p "${OUTPUT_DIR}"

if [[ -z "${VK_ICD_FILENAMES:-}" && -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

IFS=',' read -r -a ALL_GPUS <<< "${EVAL_GPUS}"
if [[ "${#ALL_GPUS[@]}" -eq 0 || -z "${ALL_GPUS[0]}" ]]; then
  echo "EVAL_GPUS must contain comma-separated GPU ids" >&2
  exit 1
fi
if [[ -z "${NUM_EVAL_WORKERS}" ]]; then
  NUM_EVAL_WORKERS="${#ALL_GPUS[@]}"
fi
if (( NUM_EVAL_WORKERS < 1 || NUM_EVAL_WORKERS > ${#ALL_GPUS[@]} )); then
  echo "NUM_EVAL_WORKERS=${NUM_EVAL_WORKERS} is invalid for EVAL_GPUS=${EVAL_GPUS}" >&2
  exit 1
fi
WORKER_GPUS=("${ALL_GPUS[@]:0:${NUM_EVAL_WORKERS}}")

TASKS=(
  google_robot_pick_coke_can
  google_robot_move_near
  google_robot_open_drawer
  google_robot_close_drawer
  google_robot_place_apple_in_closed_top_drawer
)
SERVER_PIDS=()
SERVER_PORTS=()
COMMON_PYTHONPATH="${OPENPI_ROOT}/packages/openpi-client/src:${SIMPLERENV_PATH}:${RLINF_ROOT}:${PYTHONPATH:-}"

run_detect_once() {
  if [[ "${RUN_DETECT}" == "1" && -f "${DETECT_SCRIPT}" ]]; then
    if ! pgrep -f "${DETECT_SCRIPT}" >/dev/null 2>&1; then
      nohup "${OPENPI_PYTHON}" "${DETECT_SCRIPT}" >> "${OUTPUT_DIR}/detect.log" 2>&1 &
      echo "Started detect.py with pid $!" >> "${OUTPUT_DIR}/detect.log"
    fi
  fi
}

cleanup() {
  local exit_code=$?
  for pid in "${SERVER_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done
  for pid in "${SERVER_PIDS[@]:-}"; do
    [[ -z "${pid}" ]] || wait "${pid}" >/dev/null 2>&1 || true
  done
  run_detect_once
  exit "${exit_code}"
}
trap cleanup EXIT

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "$2: $1" >&2
    exit 1
  fi
}

require_file "${MODEL_PATH}/model.safetensors" "Missing OpenPI checkpoint"
NORM_STATS_ASSET="${MODEL_PATH}/assets/IPEC-COMMUNITY/fractal20220817_data_lerobot/norm_stats.json"
NORM_STATS_SERVER="${MODEL_PATH}/IPEC-COMMUNITY/fractal20220817_data_lerobot/norm_stats.json"
require_file "${NORM_STATS_ASSET}" "Missing Fractal norm stats"
if [[ ! -f "${NORM_STATS_SERVER}" ]]; then
  mkdir -p "$(dirname "${NORM_STATS_SERVER}")"
  ln -sfn "../../assets/IPEC-COMMUNITY/fractal20220817_data_lerobot/norm_stats.json" "${NORM_STATS_SERVER}"
fi
require_file "${OPENPI_PYTHON}" "Missing OpenPI python"
require_file "${SIMPLER_PYTHON}" "Missing SimplerEnv python"
require_file "${RLINF_ROOT}/examples/embodiment/simpler_eval_openpi.py" "Missing evaluator"

PYTHONPATH="${OPENPI_ROOT}/src:${PYTHONPATH:-}" "${OPENPI_PYTHON}" - <<'PY'
from openpi.training import config

config.get_config("pi05_simpler_google_robot_rt1")
print("openpi config pi05_simpler_google_robot_rt1 ok")
PY

PYTHONPATH="${SIMPLERENV_PATH}:${PYTHONPATH:-}" "${SIMPLER_PYTHON}" - <<'PY'
from simpler_env import ENVIRONMENTS

required = {
    "google_robot_pick_coke_can",
    "google_robot_move_near",
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_place_apple_in_closed_top_drawer",
}
missing = sorted(required.difference(ENVIRONMENTS))
if missing:
    raise SystemExit(f"missing SimplerEnv tasks: {missing}")
print("simpler_env Google Robot tasks ok")
PY

{
  echo "timestamp=${TIMESTAMP}"
  echo "model_path=${MODEL_PATH}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "eval_gpus=${EVAL_GPUS}"
  echo "num_eval_workers=${NUM_EVAL_WORKERS}"
  echo "episodes_per_task=${EPISODES_PER_TASK}"
  echo "action_chunk=${ACTION_CHUNK}"
  echo "seed=${SEED}"
  echo "tasks=${TASKS[*]}"
  echo "vk_icd_filenames=${VK_ICD_FILENAMES:-}"
  echo "RLinf HEAD=$(git -C "${RLINF_ROOT}" rev-parse HEAD 2>/dev/null || true)"
  echo "OpenPI HEAD=$(git -C "${OPENPI_ROOT}" rev-parse HEAD 2>/dev/null || true)"
  echo "SimplerEnv HEAD=$(git -C "${SIMPLERENV_PATH}" rev-parse HEAD 2>/dev/null || true)"
} > "${OUTPUT_DIR}/env_manifest.txt"

start_policy_server() {
  local worker_idx="$1"
  local gpu="${WORKER_GPUS[worker_idx]}"
  local port="$((PORT + worker_idx))"
  local worker_dir="${OUTPUT_DIR}/workers/worker_${worker_idx}"
  mkdir -p "${worker_dir}"
  (
    cd "${OPENPI_ROOT}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export PYTHONPATH="${OPENPI_ROOT}/src:${PYTHONPATH:-}"
    exec "${OPENPI_PYTHON}" scripts/serve_policy.py \
      --port "${port}" \
      policy:checkpoint \
      --policy.config pi05_simpler_google_robot_rt1 \
      --policy.dir "${MODEL_PATH}"
  ) > "${worker_dir}/openpi_server.log" 2>&1 &
  SERVER_PIDS[worker_idx]=$!
  SERVER_PORTS[worker_idx]="${port}"
  echo "Started policy worker=${worker_idx} gpu=${gpu} port=${port}"
}

wait_for_policy_server() {
  local worker_idx="$1"
  local port="${SERVER_PORTS[worker_idx]}"
  timeout "${SERVER_START_TIMEOUT}" env \
    PYTHONPATH="${COMMON_PYTHONPATH}" \
    POLICY_PORT="${port}" \
    "${SIMPLER_PYTHON}" - <<'PY'
import os
import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

policy = WebsocketClientPolicy(host="127.0.0.1", port=int(os.environ["POLICY_PORT"]))
result = policy.infer({
    "observation/image": np.zeros((256, 320, 3), dtype=np.uint8),
    "observation/state": np.zeros(8, dtype=np.float32),
    "prompt": "pick coke can",
})
actions = np.asarray(result["actions"])
if actions.shape != (16, 7) or not np.isfinite(actions).all():
    raise RuntimeError(f"invalid policy actions: shape={actions.shape}")
PY
}

episode_count_for_worker() {
  local base=$((EPISODES_PER_TASK / NUM_EVAL_WORKERS))
  local rem=$((EPISODES_PER_TASK % NUM_EVAL_WORKERS))
  local count="${base}"
  if (( $1 < rem )); then count=$((count + 1)); fi
  echo "${count}"
}

episode_offset_for_worker() {
  local idx="$1"
  local base=$((EPISODES_PER_TASK / NUM_EVAL_WORKERS))
  local rem=$((EPISODES_PER_TASK % NUM_EVAL_WORKERS))
  if (( idx < rem )); then
    echo $((idx * base + idx))
  else
    echo $((idx * base + rem))
  fi
}

aggregate_results() {
  STAGE_DIR="${OUTPUT_DIR}" TASKS="${TASKS[*]}" "${OPENPI_PYTHON}" - <<'PY'
import json
import math
import os
import time
from pathlib import Path

stage_dir = Path(os.environ["STAGE_DIR"])
tasks = os.environ["TASKS"].split()
task_order = {task: index for index, task in enumerate(tasks)}
records = []
for path in sorted(stage_dir.glob("shard_*/episodes.jsonl")):
    shard = path.parent.name
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            record.setdefault("shard", shard)
            records.append(record)
records.sort(key=lambda item: (task_order.get(item.get("task"), 999), item.get("episode_index", -1)))
with (stage_dir / "episodes.jsonl").open("w", encoding="utf-8") as output:
    for record in records:
        output.write(json.dumps(record, sort_keys=True) + "\n")

task_metrics = {}
for task in tasks:
    selected = [record for record in records if record.get("task") == task]
    successes = sum(int(bool(record.get("success"))) for record in selected)
    episodes = len(selected)
    task_metrics[task] = {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes if episodes else math.nan,
        "avg_steps": sum(record.get("steps", 0) for record in selected) / episodes if episodes else math.nan,
        "total_clipped_actions": sum(int(record.get("clipped_actions", 0)) for record in selected),
    }
rates = [metric["success_rate"] for metric in task_metrics.values() if not math.isnan(metric["success_rate"])]
metrics = {
    "aggregated_at": time.time(),
    "tasks": task_metrics,
    "average_success_rate": sum(rates) / len(rates) if rates else math.nan,
    "total_episodes": len(records),
    "total_successes": sum(int(bool(record.get("success"))) for record in records),
}
(stage_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(metrics, indent=2, sort_keys=True))
PY
}

for ((worker_idx = 0; worker_idx < NUM_EVAL_WORKERS; worker_idx++)); do
  start_policy_server "${worker_idx}"
done

READY_PIDS=()
for ((worker_idx = 0; worker_idx < NUM_EVAL_WORKERS; worker_idx++)); do
  wait_for_policy_server "${worker_idx}" &
  READY_PIDS[worker_idx]=$!
done
for pid in "${READY_PIDS[@]}"; do
  wait "${pid}"
done

EVAL_PIDS=()
used_workers=0
for ((worker_idx = 0; worker_idx < NUM_EVAL_WORKERS; worker_idx++)); do
  count="$(episode_count_for_worker "${worker_idx}")"
  if (( count == 0 )); then continue; fi
  offset="$(episode_offset_for_worker "${worker_idx}")"
  gpu="${WORKER_GPUS[worker_idx]}"
  port="${SERVER_PORTS[worker_idx]}"
  shard_dir="${OUTPUT_DIR}/shard_${worker_idx}"
  mkdir -p "${shard_dir}"
  eval_args=(
    "${RLINF_ROOT}/examples/embodiment/simpler_eval_openpi.py"
    --host 127.0.0.1
    --port "${port}"
    --output-dir "${shard_dir}"
    --robot-setup google_robot
    --episodes-per-task "${count}"
    --episode-offset "${offset}"
    --action-chunk "${ACTION_CHUNK}"
    --seed "${SEED}"
    --video-fps 3
    --rotation-mode axis_angle
    --gripper-mode env
    --tasks "${TASKS[@]}"
  )
  if [[ -n "${MAX_EPISODE_STEPS}" ]]; then
    eval_args+=(--max-episode-steps "${MAX_EPISODE_STEPS}")
  fi
  if [[ "${SAVE_VIDEO}" == "1" ]]; then
    eval_args+=(--save-video)
  fi
  echo "Starting eval shard=${worker_idx} gpu=${gpu} episodes_per_task=${count} offset=${offset}"
  env \
    PYTHONPATH="${COMMON_PYTHONPATH}" \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    MUJOCO_GL="${MUJOCO_GL:-egl}" \
    DISPLAY="${DISPLAY:-}" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "${SIMPLER_PYTHON}" "${eval_args[@]}" \
    > "${shard_dir}/eval.log" 2>&1 &
  EVAL_PIDS+=("$!")
  used_workers=$((used_workers + 1))
done

failed=0
for pid in "${EVAL_PIDS[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if (( failed != 0 )); then
  for log in "${OUTPUT_DIR}"/shard_*/eval.log; do
    [[ -f "${log}" ]] && { echo "===== ${log} =====" >&2; tail -80 "${log}" >&2; }
  done
  exit 1
fi

echo "Completed ${used_workers} shard(s); aggregating results"
aggregate_results
echo "Evaluation complete: ${OUTPUT_DIR}/metrics.json"
