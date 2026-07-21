#!/usr/bin/env bash
set -euo pipefail

RLINF_ROOT="${RLINF_ROOT:-/mnt/project_rlinf/jzn/workspace/continual_learning/RLinf}"
OPENPI_ROOT="${OPENPI_ROOT:-/mnt/project_rlinf/jzn/workspace/openpi}"
OPENPI_PYTHON="${OPENPI_PYTHON:-/opt/venv/openpi/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/project_rlinf/jzn/workspace/openpi/checkpoints/sft/pi05_simpler_widowx_bridge/simpler_widowx_bridge_sft/55000}"
SIMPLERENV_PATH="${SIMPLERENV_PATH:-/mnt/project_rlinf/jzn/workspace/third_party/SimplerEnv}"
SIMPLER_ENV_PREFIX="${SIMPLER_ENV_PREFIX:-/mnt/project_rlinf/jzn/conda_envs/simpler_env}"
SIMPLER_PYTHON="${SIMPLER_PYTHON:-${SIMPLER_ENV_PREFIX}/bin/python}"
DETECT_SCRIPT="${DETECT_SCRIPT:-/mnt/project_rlinf/jzn/workspace/detect.py}"

PORT="${PORT:-8000}"
EVAL_GPUS="${EVAL_GPUS:-${GPUS:-0,1,2,3,4,5,6,7}}"
NUM_EVAL_WORKERS="${NUM_EVAL_WORKERS:-}"
SMOKE="${SMOKE:-0}"
SMOKE_THEN_FULL="${SMOKE_THEN_FULL:-1}"
POLICY_SMOKE_ONLY="${POLICY_SMOKE_ONLY:-0}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-100}"
SMOKE_EPISODES="${SMOKE_EPISODES:-1}"
ACTION_CHUNK="${ACTION_CHUNK:-4}"
SEED="${SEED:-2022}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-}"
CAMERA_NAME="${CAMERA_NAME:-}"
SAVE_VIDEO="${SAVE_VIDEO:-0}"
DEBUG_EXPORT="${DEBUG_EXPORT:-0}"
DEBUG_EXPORT_STEPS="${DEBUG_EXPORT_STEPS:-16}"
PROMPT_OVERRIDES_JSON="${PROMPT_OVERRIDES_JSON:-}"
RUN_DETECT="${RUN_DETECT:-1}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-1800}"
ROTATION_MODE="${ROTATION_MODE:-rpy}"
GRIPPER_MODE="${GRIPPER_MODE:-open01}"

if [[ -z "${VK_ICD_FILENAMES:-}" && -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${RLINF_ROOT}/logs/simpler_widowx_visual_matching/${TIMESTAMP}}"
mkdir -p "${OUTPUT_DIR}"

IFS=',' read -r -a ALL_GPUS <<< "${EVAL_GPUS}"
if [[ "${#ALL_GPUS[@]}" -eq 0 || -z "${ALL_GPUS[0]}" ]]; then
  echo "EVAL_GPUS is empty; expected comma-separated GPU ids, e.g. 0,1,2,3,4,5,6,7" >&2
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

SERVER_PIDS=()
SERVER_PORTS=()
COMMON_PYTHONPATH="${OPENPI_ROOT}/packages/openpi-client/src:${SIMPLERENV_PATH}:${RLINF_ROOT}:${PYTHONPATH:-}"
POLICY_CLIENT_PYTHON="${POLICY_CLIENT_PYTHON:-${SIMPLER_PYTHON}}"

run_detect_once() {
  if [[ "${RUN_DETECT}" == "1" && -f "${DETECT_SCRIPT}" ]]; then
    if pgrep -f "${DETECT_SCRIPT}" >/dev/null 2>&1; then
      echo "detect.py is already running" >> "${OUTPUT_DIR}/detect.log"
    else
      nohup "${OPENPI_PYTHON}" "${DETECT_SCRIPT}" >> "${OUTPUT_DIR}/detect.log" 2>&1 &
      echo "Started detect.py in background with pid $!" >> "${OUTPUT_DIR}/detect.log"
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
    if [[ -n "${pid}" ]]; then
      wait "${pid}" >/dev/null 2>&1 || true
    fi
  done
  # run_detect_once
  exit "${exit_code}"
}
trap cleanup EXIT

require_file() {
  local path="$1"
  local message="$2"
  if [[ ! -f "${path}" ]]; then
    echo "${message}: ${path}" >&2
    exit 1
  fi
}

require_file "${MODEL_PATH}/model.safetensors" "Missing OpenPI checkpoint"
NORM_STATS_ASSET="${MODEL_PATH}/assets/IPEC-COMMUNITY/bridge_orig_lerobot/norm_stats.json"
NORM_STATS_SERVER="${MODEL_PATH}/IPEC-COMMUNITY/bridge_orig_lerobot/norm_stats.json"
require_file "${NORM_STATS_ASSET}" "Missing Bridge norm stats"
if [[ ! -f "${NORM_STATS_SERVER}" ]]; then
  mkdir -p "$(dirname "${NORM_STATS_SERVER}")"
  ln -sfn "../../assets/IPEC-COMMUNITY/bridge_orig_lerobot/norm_stats.json" "${NORM_STATS_SERVER}"
fi
require_file "${NORM_STATS_SERVER}" "Missing OpenPI server norm stats compatibility link"
require_file "${OPENPI_PYTHON}" "Missing OpenPI python"
if [[ "${POLICY_SMOKE_ONLY}" == "1" && ! -f "${POLICY_CLIENT_PYTHON}" ]]; then
  POLICY_CLIENT_PYTHON="${OPENPI_PYTHON}"
fi
require_file "${POLICY_CLIENT_PYTHON}" "Missing policy client python"

if [[ "${POLICY_SMOKE_ONLY}" != "1" ]]; then
  require_file "${SIMPLER_PYTHON}" "Missing SimplerEnv python; run examples/embodiment/setup_simpler_env.sh first"
  if [[ ! -d "${SIMPLERENV_PATH}/.git" ]]; then
    echo "Missing SimplerEnv source at ${SIMPLERENV_PATH}; run examples/embodiment/setup_simpler_env.sh first" >&2
    exit 1
  fi
fi

if ! PYTHONPATH="${OPENPI_ROOT}/src:${PYTHONPATH:-}" "${OPENPI_PYTHON}" - <<'PY'; then
from openpi.training import config

config.get_config("pi05_simpler_widowx_bridge")
print("openpi config pi05_simpler_widowx_bridge ok")
PY
  echo "OpenPI workspace source cannot load pi05_simpler_widowx_bridge" >&2
  exit 1
fi

if [[ "${POLICY_SMOKE_ONLY}" != "1" ]]; then
  if ! PYTHONPATH="${SIMPLERENV_PATH}:${PYTHONPATH:-}" "${SIMPLER_PYTHON}" - <<'PY'; then
import simpler_env
from simpler_env import ENVIRONMENTS

required = [
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "widowx_put_eggplant_in_basket",
]
missing = [task for task in required if task not in ENVIRONMENTS]
if missing:
    raise SystemExit(f"missing SimplerEnv tasks: {missing}")
print("simpler_env import/tasks ok")
PY
    echo "SimplerEnv import failed; run examples/embodiment/setup_simpler_env.sh and check its logs" >&2
    exit 1
  fi
fi

{
  echo "timestamp=${TIMESTAMP}"
  echo "rlinf_root=${RLINF_ROOT}"
  echo "openpi_root=${OPENPI_ROOT}"
  echo "model_path=${MODEL_PATH}"
  echo "simpler_env_path=${SIMPLERENV_PATH}"
  echo "simpler_python=${SIMPLER_PYTHON}"
  echo "policy_client_python=${POLICY_CLIENT_PYTHON}"
  echo "base_port=${PORT}"
  echo "eval_gpus=${EVAL_GPUS}"
  echo "num_eval_workers=${NUM_EVAL_WORKERS}"
  echo "worker_gpus=${WORKER_GPUS[*]}"
  echo "policy_smoke_only=${POLICY_SMOKE_ONLY}"
  echo "action_chunk=${ACTION_CHUNK}"
  echo "debug_export=${DEBUG_EXPORT}"
  echo "debug_export_steps=${DEBUG_EXPORT_STEPS}"
  echo "prompt_overrides_json=${PROMPT_OVERRIDES_JSON}"
  echo "rotation_mode=${ROTATION_MODE}"
  echo "gripper_mode=${GRIPPER_MODE}"
  echo "vk_icd_filenames=${VK_ICD_FILENAMES:-}"
  echo "norm_stats_asset=${NORM_STATS_ASSET}"
  echo "norm_stats_server=${NORM_STATS_SERVER}"
  echo
  echo "RLinf HEAD:"
  git -C "${RLINF_ROOT}" rev-parse HEAD || true
  echo
  echo "RLinf status:"
  git -C "${RLINF_ROOT}" status --short || true
  echo
  echo "OpenPI HEAD:"
  git -C "${OPENPI_ROOT}" rev-parse HEAD || true
  echo
  echo "OpenPI status:"
  git -C "${OPENPI_ROOT}" status --short || true
  echo
  echo "SimplerEnv HEAD:"
  git -C "${SIMPLERENV_PATH}" rev-parse HEAD || true
  echo
  echo "SimplerEnv submodules:"
  git -C "${SIMPLERENV_PATH}" submodule status --recursive || true
} > "${OUTPUT_DIR}/env_manifest.txt"

"${OPENPI_PYTHON}" -m pip freeze > "${OUTPUT_DIR}/openpi_pip_freeze.txt" 2>/dev/null || true
if [[ -f "${SIMPLER_PYTHON}" ]]; then
  "${SIMPLER_PYTHON}" -m pip freeze > "${OUTPUT_DIR}/simpler_env_pip_freeze.txt" 2>/dev/null || true
fi

start_policy_server() {
  local worker_idx="$1"
  local gpu="${WORKER_GPUS[worker_idx]}"
  local port="$((PORT + worker_idx))"
  local worker_dir="${OUTPUT_DIR}/workers/worker_${worker_idx}"
  mkdir -p "${worker_dir}"

  echo "Starting OpenPI policy server worker=${worker_idx} gpu=${gpu} port=${port}; logs: ${worker_dir}/openpi_server.log"
  (
    cd "${OPENPI_ROOT}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export PYTHONPATH="${OPENPI_ROOT}/src:${PYTHONPATH:-}"
    exec "${OPENPI_PYTHON}" scripts/serve_policy.py \
      --port "${port}" \
      policy:checkpoint \
      --policy.config pi05_simpler_widowx_bridge \
      --policy.dir "${MODEL_PATH}"
  ) > "${worker_dir}/openpi_server.log" 2>&1 &

  SERVER_PIDS[worker_idx]=$!
  SERVER_PORTS[worker_idx]="${port}"
}

policy_smoke_worker() {
  local worker_idx="$1"
  local gpu="${WORKER_GPUS[worker_idx]}"
  local port="${SERVER_PORTS[worker_idx]}"
  local worker_dir="${OUTPUT_DIR}/workers/worker_${worker_idx}"

  timeout "${SERVER_START_TIMEOUT}" env \
    PYTHONPATH="${COMMON_PYTHONPATH}" \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    "${POLICY_CLIENT_PYTHON}" "${RLINF_ROOT}/examples/embodiment/simpler_eval_openpi.py" \
      --host 127.0.0.1 \
      --port "${port}" \
      --output-dir "${worker_dir}/policy_smoke" \
      --policy-smoke-only \
    > "${worker_dir}/policy_smoke.log" 2>&1
}

episode_count_for_worker() {
  local total="$1"
  local idx="$2"
  local workers="$3"
  local base=$((total / workers))
  local rem=$((total % workers))
  local count="${base}"
  if (( idx < rem )); then
    count=$((count + 1))
  fi
  echo "${count}"
}

episode_offset_for_worker() {
  local total="$1"
  local idx="$2"
  local workers="$3"
  local base=$((total / workers))
  local rem=$((total % workers))
  local offset=$((idx * base))
  if (( idx < rem )); then
    offset=$((offset + idx))
  else
    offset=$((offset + rem))
  fi
  echo "${offset}"
}

build_eval_args() {
  local out_dir="$1"
  local episodes="$2"
  local episode_offset="$3"
  local port="$4"
  shift 4
  EVAL_ARGS=(
    "${RLINF_ROOT}/examples/embodiment/simpler_eval_openpi.py"
    --host 127.0.0.1
    --port "${port}"
    --output-dir "${out_dir}"
    --episodes-per-task "${episodes}"
    --episode-offset "${episode_offset}"
    --action-chunk "${ACTION_CHUNK}"
    --seed "${SEED}"
    --rotation-mode "${ROTATION_MODE}"
    --gripper-mode "${GRIPPER_MODE}"
    --tasks "$@"
  )
  if [[ -n "${MAX_EPISODE_STEPS}" ]]; then
    EVAL_ARGS+=(--max-episode-steps "${MAX_EPISODE_STEPS}")
  fi
  if [[ -n "${CAMERA_NAME}" ]]; then
    EVAL_ARGS+=(--camera-name "${CAMERA_NAME}")
  fi
  if [[ "${SAVE_VIDEO}" == "1" ]]; then
    EVAL_ARGS+=(--save-video)
  fi
  if [[ "${DEBUG_EXPORT}" == "1" ]]; then
    EVAL_ARGS+=(--debug-export --debug-export-steps "${DEBUG_EXPORT_STEPS}")
  fi
  if [[ -n "${PROMPT_OVERRIDES_JSON}" ]]; then
    EVAL_ARGS+=(--prompt-overrides-json "${PROMPT_OVERRIDES_JSON}")
  fi
}

aggregate_stage() {
  local stage_dir="$1"
  shift
  STAGE_DIR="${stage_dir}" TASKS="$*" "${OPENPI_PYTHON}" - <<'PY'
import json
import math
import os
import time
from pathlib import Path

stage_dir = Path(os.environ["STAGE_DIR"])
tasks = os.environ["TASKS"].split()
task_order = {task: idx for idx, task in enumerate(tasks)}
records = []
for path in sorted(stage_dir.glob("shard_*/episodes.jsonl")):
    shard = path.parent.name
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            record.setdefault("shard", shard)
            records.append(record)

records.sort(key=lambda item: (task_order.get(item.get("task"), 999), item.get("episode_index", -1), item.get("shard", "")))
with (stage_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, sort_keys=True) + "\n")

task_metrics = {}
for task in tasks:
    task_records = [record for record in records if record.get("task") == task]
    successes = sum(int(bool(record.get("success"))) for record in task_records)
    episodes = len(task_records)
    task_metrics[task] = {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes if episodes else math.nan,
        "avg_steps": sum(record.get("steps", 0) for record in task_records) / episodes if episodes else math.nan,
        "total_clipped_actions": sum(int(record.get("clipped_actions", 0)) for record in task_records),
    }

rates = [value["success_rate"] for value in task_metrics.values() if not math.isnan(value["success_rate"])]
metrics = {
    "elapsed_s": None,
    "aggregated_at": time.time(),
    "tasks": task_metrics,
    "average_success_rate": sum(rates) / len(rates) if rates else math.nan,
    "total_episodes": len(records),
    "total_successes": sum(int(bool(record.get("success"))) for record in records),
    "shards": sorted(path.parent.name for path in stage_dir.glob("shard_*/episodes.jsonl")),
}
(stage_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(metrics, indent=2, sort_keys=True))
PY
}

run_eval_shards() {
  local stage_dir="$1"
  local episodes_per_task="$2"
  shift 2
  local tasks=("$@")
  local pids=()
  local used_workers=0

  mkdir -p "${stage_dir}"
  echo "Running ${stage_dir##*/}: episodes_per_task=${episodes_per_task} tasks=${tasks[*]}"
  for ((worker_idx = 0; worker_idx < NUM_EVAL_WORKERS; worker_idx++)); do
    local count
    count="$(episode_count_for_worker "${episodes_per_task}" "${worker_idx}" "${NUM_EVAL_WORKERS}")"
    if (( count == 0 )); then
      continue
    fi
    local offset
    offset="$(episode_offset_for_worker "${episodes_per_task}" "${worker_idx}" "${NUM_EVAL_WORKERS}")"
    local gpu="${WORKER_GPUS[worker_idx]}"
    local port="${SERVER_PORTS[worker_idx]}"
    local shard_dir="${stage_dir}/shard_${worker_idx}"
    mkdir -p "${shard_dir}"
    build_eval_args "${shard_dir}" "${count}" "${offset}" "${port}" "${tasks[@]}"
    echo "  shard=${worker_idx} gpu=${gpu} port=${port} episodes=${count} offset=${offset}"
    env \
      PYTHONPATH="${COMMON_PYTHONPATH}" \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      MUJOCO_GL="${MUJOCO_GL:-egl}" \
      DISPLAY="${DISPLAY:-}" \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      "${SIMPLER_PYTHON}" "${EVAL_ARGS[@]}" \
      > "${shard_dir}/eval.log" 2>&1 &
    pids+=("$!")
    used_workers=$((used_workers + 1))
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if (( failed != 0 )); then
    echo "One or more eval shards failed. Recent shard logs:" >&2
    for log in "${stage_dir}"/shard_*/eval.log; do
      [[ -f "${log}" ]] || continue
      echo "===== ${log} =====" >&2
      tail -80 "${log}" >&2 || true
    done
    exit 1
  fi

  echo "Completed ${used_workers} shard(s); aggregating ${stage_dir}"
  aggregate_stage "${stage_dir}" "${tasks[@]}"
}

for ((worker_idx = 0; worker_idx < NUM_EVAL_WORKERS; worker_idx++)); do
  start_policy_server "${worker_idx}"
done

echo "Waiting for ${NUM_EVAL_WORKERS} policy server(s) with fake-observation smoke."
SMOKE_PIDS=()
for ((worker_idx = 0; worker_idx < NUM_EVAL_WORKERS; worker_idx++)); do
  policy_smoke_worker "${worker_idx}" &
  SMOKE_PIDS[worker_idx]=$!
done

smoke_failed=0
for ((worker_idx = 0; worker_idx < NUM_EVAL_WORKERS; worker_idx++)); do
  if ! wait "${SMOKE_PIDS[worker_idx]}"; then
    smoke_failed=1
  fi
done
if (( smoke_failed != 0 )); then
  echo "One or more policy server smoke checks failed. Recent worker logs:" >&2
  for ((worker_idx = 0; worker_idx < NUM_EVAL_WORKERS; worker_idx++)); do
    worker_dir="${OUTPUT_DIR}/workers/worker_${worker_idx}"
    echo "===== worker ${worker_idx} server =====" >&2
    tail -80 "${worker_dir}/openpi_server.log" >&2 || true
    echo "===== worker ${worker_idx} smoke =====" >&2
    tail -80 "${worker_dir}/policy_smoke.log" >&2 || true
  done
  exit 1
fi

if [[ "${POLICY_SMOKE_ONLY}" == "1" ]]; then
  echo "Policy smoke completed for ${NUM_EVAL_WORKERS} worker(s)."
  echo "Output directory: ${OUTPUT_DIR}"
  exit 0
fi

if [[ "${SMOKE}" == "1" ]]; then
  run_eval_shards "${OUTPUT_DIR}/smoke" "${SMOKE_EPISODES}" widowx_spoon_on_towel
else
  if [[ "${SMOKE_THEN_FULL}" == "1" ]]; then
    run_eval_shards "${OUTPUT_DIR}/smoke" "${SMOKE_EPISODES}" widowx_spoon_on_towel
  fi
  run_eval_shards "${OUTPUT_DIR}/full" "${EPISODES_PER_TASK}" \
    widowx_spoon_on_towel \
    widowx_carrot_on_plate \
    widowx_stack_cube \
    widowx_put_eggplant_in_basket
fi

echo "Evaluation complete."
echo "Output directory: ${OUTPUT_DIR}"
