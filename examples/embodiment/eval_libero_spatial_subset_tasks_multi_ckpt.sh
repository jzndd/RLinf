#!/bin/bash

set -uo pipefail

EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$(dirname "${EMBODIED_PATH}")")"
EVAL_SCRIPT="${EMBODIED_PATH}/eval_libero_spatial_subset_tasks.sh"

SUBSET_NAME="${1:-subset1}"
CONFIG_NAME="${2:-libero_spatial_grpo_openpi_pi05_noise}"
WORLD_SIZE="${3:-8}"
BASE_LOG_DIR="${4:-${REPO_PATH}/logs/20260430-18:10:45-libero_spatial_offline_grpo_openpi_pi05_flow_noise_subset_chain/libero_spatial_offline_grpo_openpi_pi05_flow_noise_subset1/eval_logs_l1}"
DENOISE_STEP=${5:-3}
DENOISE
shift $(( $# >= 4 ? 4 : $# ))

DEFAULT_CKPTS=(
    "/mnt/project_rlinf/jzn/workspace/continual_learning/RLinf/logs/20260430-18:10:45-libero_spatial_offline_grpo_openpi_pi05_flow_noise_subset_chain/libero_spatial_offline_grpo_openpi_pi05_flow_noise_subset1/libero_spatial_offline_grpo_openpi_pi05_flow_noise_subset1/checkpoints/global_step_6000"
)

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
    echo "Missing eval script: ${EVAL_SCRIPT}"
    exit 1
fi

if [[ $# -gt 0 ]]; then
    CKPT_INPUTS=("$@")
else
    CKPT_INPUTS=("${DEFAULT_CKPTS[@]}")
fi

resolve_ckpt_file() {
    local input_path="$1"
    local candidate_file="${input_path}/actor/model_state_dict/full_weights.pt"
    if [[ -f "${input_path}" ]]; then
        echo "${input_path}"
        return 0
    fi
    if [[ -f "${candidate_file}" ]]; then
        echo "${candidate_file}"
        return 0
    fi
    return 1
}

extract_step_tag() {
    local ckpt_file="$1"
    local fallback_tag="unknown_step"
    if [[ "${ckpt_file}" =~ (global_step_[0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo "${fallback_tag}"
    fi
}

RUN_GROUP_TAG="$(date +'%Y%m%d-%H%M%S')_${SUBSET_NAME}"
mkdir -p "${BASE_LOG_DIR}"

echo "============================================================"
echo "Batch eval start"
echo "  subset_name : ${SUBSET_NAME}"
echo "  config_name : ${CONFIG_NAME}"
echo "  world_size  : ${WORLD_SIZE}"
echo "  base_log_dir: ${BASE_LOG_DIR}"
echo "  ckpt_count  : ${#CKPT_INPUTS[@]}"
echo "============================================================"

fail_count=0
for raw_ckpt in "${CKPT_INPUTS[@]}"; do
    if ! ckpt_file="$(resolve_ckpt_file "${raw_ckpt}")"; then
        echo
        echo "[SKIP] checkpoint not found:"
        echo "  input=${raw_ckpt}"
        echo "  expected_file=${raw_ckpt}/actor/model_state_dict/full_weights.pt"
        fail_count=$((fail_count + 1))
        continue
    fi

    step_tag="$(extract_step_tag "${ckpt_file}")"
    per_ckpt_log_root="${BASE_LOG_DIR}/${RUN_GROUP_TAG}_${step_tag}"

    echo
    echo "------------------------------------------------------------"
    echo "[RUN] ${step_tag}"
    echo "  ckpt_file=${ckpt_file}"
    echo "  log_root =${per_ckpt_log_root}"
    echo "------------------------------------------------------------"

    if ! bash "${EVAL_SCRIPT}" \
        "${SUBSET_NAME}" \
        "${ckpt_file}" \
        "${CONFIG_NAME}" \
        "${WORLD_SIZE}" \
        "${per_ckpt_log_root}" \
        "${DENOISE_STEP}"; then
        echo "[FAIL] ${step_tag}"
        fail_count=$((fail_count + 1))
        continue
    fi
    echo "[DONE] ${step_tag}"
done

echo
if [[ "${fail_count}" -gt 0 ]]; then
    echo "Batch completed with ${fail_count} failure(s)."
    exit 1
fi
echo "Batch completed successfully."
