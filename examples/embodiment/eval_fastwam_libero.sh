#!/usr/bin/env bash
set -euo pipefail

EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$(dirname "$EMBODIED_PATH")")"
SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"

source /opt/venv/openpi/bin/activate

export EMBODIED_PATH
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-LIBERO}"
export LIBERO_TYPE="${LIBERO_TYPE:-standard}"

export FASTWAM_ROOT="${FASTWAM_ROOT:-/mnt/project_rlinf/jzn/workspace/FastWAM}"
export FASTWAM_MODEL_PATH="${FASTWAM_MODEL_PATH:-/mnt/project_rlinf/jlchen/code/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
export FASTWAM_DATASET_STATS_PATH="${FASTWAM_DATASET_STATS_PATH:-/mnt/project_rlinf/jlchen/code/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
export FASTWAM_ACTION_DIT_PATH="${FASTWAM_ACTION_DIT_PATH:-$FASTWAM_ROOT/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$FASTWAM_ROOT/checkpoints}"
export PYTHONPATH="$FASTWAM_ROOT:$FASTWAM_ROOT/src:$REPO_PATH:${PYTHONPATH:-}"

TOTAL_NUM_ENVS="${TOTAL_NUM_ENVS:-8}"
TIMESTAMP="$(date +'%Y%m%d-%H%M%S')"
BASE_LOG_DIR="${BASE_LOG_DIR:-$REPO_PATH/logs/fastwam/$TIMESTAMP}"
DEFAULT_CONFIGS=(
  "libero_spatial_eval_fastwam"
  # "libero_goal_eval_fastwam"
  # "libero_object_eval_fastwam"
  # "libero_10_eval_fastwam"
)

required_paths=(
  "$FASTWAM_ROOT"
  "$FASTWAM_MODEL_PATH"
  "$FASTWAM_DATASET_STATS_PATH"
  "$FASTWAM_ACTION_DIT_PATH"
  "$DIFFSYNTH_MODEL_BASE_PATH"
)
for path in "${required_paths[@]}"; do
  if [ ! -e "$path" ]; then
    echo "Required path not found: $path" >&2
    exit 1
  fi
done

if [ -n "${FASTWAM_EVAL_CONFIGS:-}" ]; then
  read -r -a CONFIGS <<< "${FASTWAM_EVAL_CONFIGS}"
else
  CONFIGS=("${DEFAULT_CONFIGS[@]}")
fi

mkdir -p "$BASE_LOG_DIR"

run_eval() {
  local config_name="$1"
  shift

  local suite_name="${config_name%_eval_fastwam}"
  local suite_short="${suite_name#libero_}"
  local log_dir="$BASE_LOG_DIR/${suite_short}"
  local log_file="$log_dir/eval_embodiment.log"
  local gpu_monitor_csv="$log_dir/gpu_monitor.csv"
  local gpu_peak_summary="$log_dir/gpu_peak_summary.txt"
  local monitor_pid=""

  mkdir -p "$log_dir"

  stop_gpu_monitor() {
    if [ -n "$monitor_pid" ] && kill -0 "$monitor_pid" 2>/dev/null; then
      kill "$monitor_pid" 2>/dev/null || true
      wait "$monitor_pid" 2>/dev/null || true
    fi

    if [ -s "$gpu_monitor_csv" ]; then
      awk -F',' '
      {
        gsub(/^[ \t]+|[ \t]+$/, "", $1);
        gsub(/^[ \t]+|[ \t]+$/, "", $2);
        gsub(/^[ \t]+|[ \t]+$/, "", $3);
        gpu=$1+0; util=$2+0; mem=$3+0;
        if (!(gpu in max_util) || util > max_util[gpu]) max_util[gpu]=util;
        if (!(gpu in max_mem) || mem > max_mem[gpu]) max_mem[gpu]=mem;
        if (util > global_max_util) global_max_util=util;
        if (mem > global_max_mem) global_max_mem=mem;
      }
      END {
        printf("global_max_gpu_util=%d%%\n", global_max_util);
        printf("global_max_mem_used=%d MiB\n", global_max_mem);
        for (gpu in max_util) {
          printf("gpu_%d_max_util=%d%% gpu_%d_max_mem=%d MiB\n", gpu, max_util[gpu], gpu, max_mem[gpu]);
        }
      }' "$gpu_monitor_csv" > "$gpu_peak_summary"
      echo "GPU peak summary saved to: $gpu_peak_summary"
    else
      echo "No GPU monitor samples were collected." > "$gpu_peak_summary"
      echo "GPU peak summary saved to: $gpu_peak_summary"
    fi
  }

  trap stop_gpu_monitor RETURN

  if command -v nvidia-smi >/dev/null 2>&1; then
    # Sample once per second: gpu_index, gpu_util, mem_used.
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > "$gpu_monitor_csv" &
    monitor_pid="$!"
    echo "Started GPU monitor (pid=${monitor_pid}), writing samples to: $gpu_monitor_csv"
  else
    echo "Warning: nvidia-smi not found, skip GPU peak monitoring." | tee -a "$log_file"
  fi

  local cmd=(
    python "$SRC_FILE"
    --config-path "$EMBODIED_PATH/config"
    --config-name "$config_name"
    "runner.logger.log_path=$log_dir"
    "env.train.total_num_envs=$TOTAL_NUM_ENVS"
    "env.eval.total_num_envs=$TOTAL_NUM_ENVS"
  )

  if [ "$#" -gt 0 ]; then
    cmd+=("$@")
  fi

  echo "Evaluation Mode: ${LIBERO_TYPE}"
  echo "Using ROBOT_PLATFORM=${ROBOT_PLATFORM}"
  echo "Config: ${config_name}"
  echo "Log directory: ${log_dir}"
  printf 'Running command:\n%s\n' "${cmd[*]}"
  "${cmd[@]}" 2>&1 | tee "$log_file"
}

for config_name in "${CONFIGS[@]}"; do
  run_eval "$config_name" "$@"
done
