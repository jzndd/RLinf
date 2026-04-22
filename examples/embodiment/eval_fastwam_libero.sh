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
export FASTWAM_CKPT_PATH="${FASTWAM_CKPT_PATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$FASTWAM_ROOT/checkpoints}"
export PYTHONPATH="$FASTWAM_ROOT:$FASTWAM_ROOT/src:$REPO_PATH:${PYTHONPATH:-}"

TOTAL_NUM_ENVS="${TOTAL_NUM_ENVS:-8}"
TIMESTAMP="$(date +'%Y%m%d-%H%M%S')"
BASE_LOG_DIR="${BASE_LOG_DIR:-$REPO_PATH/logs/fastwam/$TIMESTAMP}"
DEFAULT_CONFIGS=(
  "libero_spatial_eval_fastwam"
  "libero_goal_eval_fastwam"
  "libero_object_eval_fastwam"
  "libero_10_eval_fastwam"
)

resolve_fastwam_ckpt_path() {
  local raw_path="$1"
  if [ -z "$raw_path" ]; then
    return 0
  fi

  if [ -d "$raw_path" ]; then
    local base_name
    base_name="$(basename "$raw_path")"
    local parent_dir
    parent_dir="$(dirname "$(dirname "$raw_path")")"
    local resolved_path="$parent_dir/weights/${base_name}.pt"
    if [ ! -f "$resolved_path" ]; then
      echo "Unable to resolve FastWAM weight file from state dir: $raw_path" >&2
      return 1
    fi
    printf '%s\n' "$resolved_path"
    return 0
  fi

  printf '%s\n' "$raw_path"
}

RESOLVED_FASTWAM_CKPT_PATH="$(resolve_fastwam_ckpt_path "$FASTWAM_CKPT_PATH")"

required_paths=(
  "$FASTWAM_ROOT"
  "$FASTWAM_MODEL_PATH"
  "$FASTWAM_DATASET_STATS_PATH"
  "$FASTWAM_ACTION_DIT_PATH"
  "$DIFFSYNTH_MODEL_BASE_PATH"
)
if [ -n "$RESOLVED_FASTWAM_CKPT_PATH" ]; then
  required_paths+=("$RESOLVED_FASTWAM_CKPT_PATH")
fi
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

  mkdir -p "$log_dir"

  local cmd=(
    python "$SRC_FILE"
    --config-path "$EMBODIED_PATH/config"
    --config-name "$config_name"
    "runner.logger.log_path=$log_dir"
    "env.train.total_num_envs=$TOTAL_NUM_ENVS"
    "env.eval.total_num_envs=$TOTAL_NUM_ENVS"
  )

  if [ -n "$RESOLVED_FASTWAM_CKPT_PATH" ]; then
    cmd+=("runner.ckpt_path=$RESOLVED_FASTWAM_CKPT_PATH")
  fi

  if [ "$#" -gt 0 ]; then
    cmd+=("$@")
  fi

  echo "Evaluation Mode: ${LIBERO_TYPE}"
  echo "Using ROBOT_PLATFORM=${ROBOT_PLATFORM}"
  echo "Config: ${config_name}"
  echo "Log directory: ${log_dir}"
  if [ -n "$RESOLVED_FASTWAM_CKPT_PATH" ]; then
    echo "Resolved runner.ckpt_path=${RESOLVED_FASTWAM_CKPT_PATH}"
  fi
  printf 'Running command:\n%s\n' "${cmd[*]}"
  "${cmd[@]}" 2>&1 | tee "$log_file"
}

for config_name in "${CONFIGS[@]}"; do
  run_eval "$config_name" "$@"
done
