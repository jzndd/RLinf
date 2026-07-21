#!/usr/bin/env bash
set -euo pipefail

RLINF_ROOT="${RLINF_ROOT:-/mnt/project_rlinf/jzn/workspace/continual_learning/RLinf}"
OPENPI_PYTHON="${OPENPI_PYTHON:-/opt/venv/openpi/bin/python}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/mnt/project_rlinf/jzn/workspace/openpi/checkpoints/sft/pi05_simpler_google_robot_rt1/simpler_google_robot_rt1_full_sft}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-55000 50000 45000 40000}"
PORT_BASE="${PORT_BASE:-21000}"
PORT_STRIDE="${PORT_STRIDE:-100}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
RUN_DETECT="${RUN_DETECT:-0}"
DETECT_SCRIPT="${DETECT_SCRIPT:-/mnt/project_rlinf/jzn/workspace/detect.py}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT:-${RLINF_ROOT}/logs/simpler_google_robot_visual_matching_sweep/${TIMESTAMP}}"
RESULTS_CSV="${SWEEP_OUTPUT_ROOT}/results.csv"
mkdir -p "${SWEEP_OUTPUT_ROOT}"

TASKS=(
  google_robot_pick_coke_can
  google_robot_move_near
  google_robot_open_drawer
  google_robot_close_drawer
  google_robot_place_apple_in_closed_top_drawer
)

run_detect_once() {
  if [[ "${RUN_DETECT}" == "1" && -f "${DETECT_SCRIPT}" ]]; then
    if ! pgrep -f "${DETECT_SCRIPT}" >/dev/null 2>&1; then
      nohup "${OPENPI_PYTHON}" "${DETECT_SCRIPT}" >> "${SWEEP_OUTPUT_ROOT}/detect.log" 2>&1 &
      echo "Started detect.py with pid $!" >> "${SWEEP_OUTPUT_ROOT}/detect.log"
    fi
  fi
}
trap run_detect_once EXIT

{
  echo "timestamp=${TIMESTAMP}"
  echo "checkpoint_root=${CHECKPOINT_ROOT}"
  echo "checkpoint_steps=${CHECKPOINT_STEPS}"
  echo "episodes_per_task=${EPISODES_PER_TASK:-100}"
  echo "action_chunk=${ACTION_CHUNK:-4}"
  echo "eval_gpus=${EVAL_GPUS:-0,1,2,3,4,5,6,7}"
} > "${SWEEP_OUTPUT_ROOT}/sweep_manifest.txt"

CSV_PATH="${RESULTS_CSV}" "${OPENPI_PYTHON}" - <<'PY'
import csv
import os

header = [
    "checkpoint",
    "overall_sr",
    "pick_coke_can_sr",
    "move_near_sr",
    "open_drawer_sr",
    "close_drawer_sr",
    "place_apple_in_closed_top_drawer_sr",
]
with open(os.environ["CSV_PATH"], "w", newline="", encoding="utf-8") as output:
    csv.writer(output).writerow(header)
PY

append_metrics_row() {
  local step="$1"
  local metrics_path="$2"
  STEP="${step}" METRICS_PATH="${metrics_path}" CSV_PATH="${RESULTS_CSV}" "${OPENPI_PYTHON}" - <<'PY'
import csv
import json
import os

tasks = [
    "google_robot_pick_coke_can",
    "google_robot_move_near",
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_place_apple_in_closed_top_drawer",
]
metrics_path = os.environ["METRICS_PATH"]
if metrics_path:
    metrics = json.loads(open(metrics_path, encoding="utf-8").read())
    row = [os.environ["STEP"], metrics["average_success_rate"]]
    row.extend(metrics["tasks"][task]["success_rate"] for task in tasks)
else:
    row = [os.environ["STEP"], *("nan" for _ in range(6))]
with open(os.environ["CSV_PATH"], "a", newline="", encoding="utf-8") as output:
    csv.writer(output).writerow(row)
PY
}

idx=0
failed=0
for step in ${CHECKPOINT_STEPS}; do
  model_path="${CHECKPOINT_ROOT}/${step}"
  output_dir="${SWEEP_OUTPUT_ROOT}/ckpt_${step}"
  port=$((PORT_BASE + idx * PORT_STRIDE))
  idx=$((idx + 1))

  echo "================================================================"
  echo "Evaluating Google Robot checkpoint ${step}"
  if [[ ! -f "${model_path}/model.safetensors" ]]; then
    echo "Missing checkpoint: ${model_path}" >&2
    append_metrics_row "${step}" ""
    failed=1
    [[ "${CONTINUE_ON_ERROR}" == "1" ]] && continue
    exit 1
  fi

  if MODEL_PATH="${model_path}" \
    OUTPUT_DIR="${output_dir}" \
    PORT="${port}" \
    RUN_DETECT=0 \
    bash "${RLINF_ROOT}/examples/embodiment/run_simpler_google_robot_vm_eval.sh"; then
    append_metrics_row "${step}" "${output_dir}/metrics.json"
  else
    echo "Checkpoint ${step} evaluation failed" >&2
    append_metrics_row "${step}" ""
    failed=1
    [[ "${CONTINUE_ON_ERROR}" == "1" ]] && continue
    exit 1
  fi
done

echo "================================================================"
echo "Sweep results: ${RESULTS_CSV}"
cat "${RESULTS_CSV}"

if (( failed != 0 )); then
  exit 1
fi
