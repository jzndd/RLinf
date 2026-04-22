#!/usr/bin/env bash
set -euo pipefail

EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$(dirname "$EMBODIED_PATH")")"

source /opt/venv/openpi/bin/activate

export EMBODIED_PATH
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"

export FASTWAM_ROOT="${FASTWAM_ROOT:-/mnt/project_rlinf/jzn/workspace/FastWAM}"
export FASTWAM_MODEL_PATH="${FASTWAM_MODEL_PATH:-/mnt/project_rlinf/jlchen/code/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
export FASTWAM_DATASET_STATS_PATH="${FASTWAM_DATASET_STATS_PATH:-/mnt/project_rlinf/jlchen/code/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
export FASTWAM_ACTION_DIT_PATH="${FASTWAM_ACTION_DIT_PATH:-$FASTWAM_ROOT/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$FASTWAM_ROOT/checkpoints}"
export PYTHONPATH="$FASTWAM_ROOT:$FASTWAM_ROOT/src:$REPO_PATH:${PYTHONPATH:-}"

CONFIG_NAME="${CONFIG_NAME:-libero_spatial_eval_fastwam}"
TOTAL_NUM_ENVS="${TOTAL_NUM_ENVS:-8}"
LOG_DIR="${LOG_DIR:-$REPO_PATH/logs/fastwam/libero_spatial/$(date +'%Y%m%d-%H%M%S')}"
LOG_FILE="$LOG_DIR/eval_fastwam_libero.log"

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

mkdir -p "$LOG_DIR"

CMD=(
  python "$EMBODIED_PATH/eval_embodied_agent.py"
  --config-path "$EMBODIED_PATH/config"
  --config-name "$CONFIG_NAME"
  "runner.logger.log_path=$LOG_DIR"
  "env.train.total_num_envs=$TOTAL_NUM_ENVS"
  "env.eval.total_num_envs=$TOTAL_NUM_ENVS"
)

if [ "$#" -gt 0 ]; then
  CMD+=("$@")
fi

printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
