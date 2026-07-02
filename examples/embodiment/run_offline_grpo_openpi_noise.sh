#! /bin/bash
set -euo pipefail

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_offline_grpo_openpi.py"

source /opt/venv/openpi/bin/activate

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false

CONFIG_NAME="libero_spatial_offline_grpo_openpi_pi05_noise"

# CONFIG_NAME="libero_130_offline_grpo_openpi_pi05"

# RESUME_CKPT_PATH="/mnt/project_rlinf/jzn/workspace/continual_learning/RLinf/checkpoints/pi05_libero_almost_zero/full_weights.pt"
NUM_STEPS=5

echo "Using Python at $(which python)"

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}_wo_vlm_step${NUM_STEPS}"
MEGA_LOG_FILE="${LOG_DIR}/run_offline_grpo_openpi.log"
mkdir -p "${LOG_DIR}"

CMD=(
    python "${SRC_FILE}"
    --config-path "${EMBODIED_PATH}/config/"
    --config-name "${CONFIG_NAME}"
    runner.logger.log_path="${LOG_DIR}"
    actor.model.num_steps="${NUM_STEPS}"
)

if [ "$#" -gt 0 ]; then
    CMD+=("$@")
fi

printf '%q ' "${CMD[@]}" > "${MEGA_LOG_FILE}"
printf '\n' >> "${MEGA_LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${MEGA_LOG_FILE}"
