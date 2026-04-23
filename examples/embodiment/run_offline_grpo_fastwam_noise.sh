#! /bin/bash
set -euo pipefail

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH="$(dirname "$(dirname "$EMBODIED_PATH")")"
export SRC_FILE="${EMBODIED_PATH}/train_offline_grpo_fastwam.py"

source /opt/venv/openpi/bin/activate

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

CONFIG_NAME="${CONFIG_NAME:-libero_spatial_offline_grpo_fastwam_noise}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-20}"

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

echo "Using Python at $(which python)"

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}-step${NUM_INFERENCE_STEPS}"
MEGA_LOG_FILE="${LOG_DIR}/run_offline_grpo_fastwam.log"
mkdir -p "${LOG_DIR}"

CMD=(
  python "${SRC_FILE}"
  --config-path "${EMBODIED_PATH}/config/"
  --config-name "${CONFIG_NAME}"
  "runner.logger.log_path=${LOG_DIR}"
  "actor.model.num_inference_steps=${NUM_INFERENCE_STEPS}"
)

if [ "$#" -gt 0 ]; then
  CMD+=("$@")
fi

printf '%q ' "${CMD[@]}" > "${MEGA_LOG_FILE}"
printf '\n' >> "${MEGA_LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${MEGA_LOG_FILE}"
