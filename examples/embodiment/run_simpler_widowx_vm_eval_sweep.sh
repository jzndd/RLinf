#!/usr/bin/env bash
set -euo pipefail

RLINF_ROOT="${RLINF_ROOT:-/mnt/project_rlinf/jzn/workspace/continual_learning/RLinf}"
OPENPI_PYTHON="${OPENPI_PYTHON:-/opt/venv/openpi/bin/python}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/mnt/project_rlinf/jzn/workspace/openpi/checkpoints/sft/pi05_simpler_widowx_bridge/simpler_widowx_bridge_full_sft}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-55000 50000 45000 40000}"
PORT_BASE="${PORT_BASE:-20000}"
PORT_STRIDE="${PORT_STRIDE:-100}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
RUN_DETECT="${RUN_DETECT:-0}"
DETECT_SCRIPT="${DETECT_SCRIPT:-/mnt/project_rlinf/jzn/workspace/detect.py}"
STOP_DETECT_ON_EXIT="${STOP_DETECT_ON_EXIT:-0}"
DETECT_PATTERNS="${DETECT_PATTERNS:-${DETECT_SCRIPT} detect.py zk.py}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT:-${RLINF_ROOT}/logs/simpler_widowx_visual_matching_sweep_fullsft/}"
mkdir -p "${SWEEP_OUTPUT_ROOT}"

if [[ -z "${VK_ICD_FILENAMES:-}" && -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

run_detect_once() {
  if [[ "${RUN_DETECT}" == "1" && -f "${DETECT_SCRIPT}" ]]; then
    if pgrep -f "${DETECT_SCRIPT}" >/dev/null 2>&1; then
      echo "detect.py is already running" >> "${SWEEP_OUTPUT_ROOT}/detect.log"
    else
      nohup "${OPENPI_PYTHON}" "${DETECT_SCRIPT}" >> "${SWEEP_OUTPUT_ROOT}/detect.log" 2>&1 &
      echo "Started detect.py in background with pid $!" >> "${SWEEP_OUTPUT_ROOT}/detect.log"
    fi
  fi
}

stop_detect_once() {
  local pattern pid pids unique_pids remaining_pids
  pids=""
  for pattern in ${DETECT_PATTERNS}; do
    while read -r pid; do
      [[ -n "${pid}" && "${pid}" != "$$" ]] || continue
      pids+="${pid} "
    done < <(pgrep -f "${pattern}" || true)
  done

  unique_pids="$(tr ' ' '\n' <<< "${pids}" | awk 'NF' | sort -u | tr '\n' ' ')"
  if [[ -z "${unique_pids}" ]]; then
    echo "No detect/zk.py process found" >> "${SWEEP_OUTPUT_ROOT}/detect.log"
    return
  fi

  echo "Stopping detect/zk.py pids: ${unique_pids}" >> "${SWEEP_OUTPUT_ROOT}/detect.log"
  kill ${unique_pids} >/dev/null 2>&1 || true
  sleep 2
  remaining_pids=""
  for pid in ${unique_pids}; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      remaining_pids+="${pid} "
    fi
  done
  if [[ -n "${remaining_pids}" ]]; then
    echo "Force-stopping detect/zk.py pids: ${remaining_pids}" >> "${SWEEP_OUTPUT_ROOT}/detect.log"
    kill -9 ${remaining_pids} >/dev/null 2>&1 || true
  fi
}

finish() {
  if [[ "${STOP_DETECT_ON_EXIT}" == "1" ]]; then
    stop_detect_once
  else
    run_detect_once
  fi
}
trap finish EXIT

{
  echo "timestamp=${TIMESTAMP}"
  echo "checkpoint_root=${CHECKPOINT_ROOT}"
  echo "checkpoint_steps=${CHECKPOINT_STEPS}"
  echo "sweep_output_root=${SWEEP_OUTPUT_ROOT}"
  echo "port_base=${PORT_BASE}"
  echo "port_stride=${PORT_STRIDE}"
  echo "continue_on_error=${CONTINUE_ON_ERROR}"
  echo "run_detect=${RUN_DETECT}"
  echo "stop_detect_on_exit=${STOP_DETECT_ON_EXIT}"
  echo "detect_patterns=${DETECT_PATTERNS}"
  echo "vk_icd_filenames=${VK_ICD_FILENAMES:-}"
} > "${SWEEP_OUTPUT_ROOT}/sweep_manifest.txt"

STATUS_FILE="${SWEEP_OUTPUT_ROOT}/sweep_status.csv"
write_csv_row() {
  local first=1
  local value escaped
  for value in "$@"; do
    escaped="${value//\"/\"\"}"
    if (( first == 0 )); then
      printf ","
    fi
    printf '"%s"' "${escaped}"
    first=0
  done
  printf "\n"
}
write_csv_row "step" "status" "output_dir" "metrics" > "${STATUS_FILE}"

idx=0
failed=0
for step in ${CHECKPOINT_STEPS}; do
  model_path="${CHECKPOINT_ROOT}/${step}"
  output_dir="${SWEEP_OUTPUT_ROOT}/ckpt_${step}"
  port=$((PORT_BASE + idx * PORT_STRIDE))
  idx=$((idx + 1))

  echo "================================================================"
  echo "Evaluating checkpoint ${step}"
  echo "MODEL_PATH=${model_path}"
  echo "OUTPUT_DIR=${output_dir}"
  echo "PORT=${port}"

  if [[ ! -d "${model_path}" ]]; then
    echo "Missing checkpoint directory: ${model_path}" >&2
    write_csv_row "${step}" "missing" "${output_dir}" "" >> "${STATUS_FILE}"
    failed=1
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      exit 1
    fi
    continue
  fi

  if MODEL_PATH="${model_path}" \
    OUTPUT_DIR="${output_dir}" \
    PORT="${port}" \
    RUN_DETECT=0 \
    bash "${RLINF_ROOT}/examples/embodiment/run_simpler_widowx_vm_eval.sh"; then
    metrics=""
    if [[ -f "${output_dir}/full/metrics.json" ]]; then
      metrics="${output_dir}/full/metrics.json"
    elif [[ -f "${output_dir}/smoke/metrics.json" ]]; then
      metrics="${output_dir}/smoke/metrics.json"
    fi
    write_csv_row "${step}" "ok" "${output_dir}" "${metrics}" >> "${STATUS_FILE}"
  else
    write_csv_row "${step}" "failed" "${output_dir}" "" >> "${STATUS_FILE}"
    failed=1
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      exit 1
    fi
  fi
done

echo "================================================================"
echo "Sweep complete. Status: ${STATUS_FILE}"
cat "${STATUS_FILE}"

if (( failed != 0 )); then
  exit 1
fi
