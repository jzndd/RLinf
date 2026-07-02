#!/bin/bash

set -euo pipefail

export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"

EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$(dirname "${EMBODIED_PATH}")")"
EVAL_SCRIPT="${EMBODIED_PATH}/eval_libero_spatial_subset_tasks.sh"

CONFIG_NAME="${1:-libero_spatial_grpo_openpi_pi05_noise}"
WORLD_SIZE="${2:-8}"
BASE_LOG_DIR="${3:-${REPO_PATH}/logs/20260422-16:03:37-libero_spatial_offline_grpo_openpi_pi05_flow_noise_subset_chain/}"

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
    echo "Missing eval script: ${EVAL_SCRIPT}"
    exit 1
fi

SUBSETS=(
    # "subset1"
    # "subset2"
    # "subset3"
    "subset4"
    "subset5"
)

CKPT_PATHS=(
    # "/mnt/project_rlinf/jzn/workspace/continual_learning/RLinf/logs/20260422-16:03:37-libero_spatial_offline_grpo_openpi_pi05_flow_noise_subset_chain/libero_spatial_offline_grpo_openpi_pi05_flow_noise_subset1/libero_spatial_offline_grpo_openpi_pi05_flow_noise_subset1/checkpoints/global_step_10000/actor/model_state_dict/full_weights.pt"
    # "/mnt/project_rlinf/jzn/workspace/continual_learning/RLinf/logs/20260422-16:03:37-libero_spatial_offline_grpo_openpi_pi05_flow_noise_subset_chain/libero_spatial_offline_grpo_openpi_pi05_flow_noise_subset2/libero_spatial_offline_grpo_openpi_pi05_flow_noise_subset2/checkpoints/global_step_3000/actor/model_state_dict/full_weights.pt"
    # "/mnt/project_rlinf/jzn/workspace/continual_learning/RLinf/logs/20260422-16:03:37-libero_spatial_offline_grpo_openpi_pi05_flow_noise_subset_chain/libero_spatial_offline_grpo_openpi_pi05_flow_noise_subset3/libero_spatial_offline_grpo_openpi_pi05_flow_noise_subset3/checkpoints/global_step_3000/actor/model_state_dict/full_weights.pt"
    "/mnt/project_rlinf/jzn/workspace/continual_learning/RLinf/logs/20260422-16:03:37-libero_spatial_offline_grpo_openpi_pi05_flow_noise_subset_chain/libero_spatial_offline_grpo_openpi_pi05_flow_noise_subset4/libero_spatial_offline_grpo_openpi_pi05_flow_noise_subset4/checkpoints/global_step_3000/actor/model_state_dict/full_weights.pt"
    "/mnt/project_rlinf/jzn/workspace/continual_learning/RLinf/logs/20260422-16:03:37-libero_spatial_offline_grpo_openpi_pi05_flow_noise_subset_chain/libero_spatial_offline_grpo_openpi_pi05_flow_noise_subset5/libero_spatial_offline_grpo_openpi_pi05_flow_noise_subset5/checkpoints/global_step_3000/actor/model_state_dict/full_weights.pt"
)

if [[ "${#SUBSETS[@]}" -ne "${#CKPT_PATHS[@]}" ]]; then
    echo "Configuration error: SUBSETS count != CKPT_PATHS count"
    exit 1
fi

mkdir -p "${BASE_LOG_DIR}"

echo "============================================================"
echo "Batch subset eval start"
echo "  config_name : ${CONFIG_NAME}"
echo "  world_size  : ${WORLD_SIZE}"
echo "  group_log   : ${BASE_LOG_DIR}"
echo "  run_count   : ${#SUBSETS[@]}"
echo "============================================================"

fail_count=0

for idx in "${!SUBSETS[@]}"; do
    subset="${SUBSETS[$idx]}"
    ckpt_path="${CKPT_PATHS[$idx]}"
    run_log_root="${BASE_LOG_DIR}/eval_logs_${subset}/$(date +'%Y%m%d-%H%M%S')_ckpt3000"

    mkdir -p "${run_log_root}"

    echo
    echo "------------------------------------------------------------"
    echo "[RUN] subset=${subset}"
    echo "  ckpt_path=${ckpt_path}"
    echo "  log_root =${run_log_root}"
    echo "------------------------------------------------------------"

    if [[ ! -f "${ckpt_path}" ]]; then
        echo "[FAIL] checkpoint not found: ${ckpt_path}"
        fail_count=$((fail_count + 1))
        continue
    fi

    if ! bash "${EVAL_SCRIPT}" \
        "${subset}" \
        "${ckpt_path}" \
        "${CONFIG_NAME}" \
        "${WORLD_SIZE}" \
        "${run_log_root}"; then
        echo "[FAIL] subset=${subset}"
        fail_count=$((fail_count + 1))
        continue
    fi

    echo "[DONE] subset=${subset}"
done

echo
if [[ "${fail_count}" -gt 0 ]]; then
    echo "Batch completed with ${fail_count} failure(s)."
    exit 1
fi
echo "Batch completed successfully."
