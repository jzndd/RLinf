#!/bin/bash

set -uo pipefail

EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$(dirname "${EMBODIED_PATH}")")"
EVAL_SCRIPT="${EMBODIED_PATH}/eval_libero_spatial_subset_tasks.sh"

SUBSET_NAME="${1:-subset1}"
CONFIG_NAME="${2:-libero_spatial_grpo_openpi_pi05_noise_base90_oneshot}"
WORLD_SIZE="${3:-8}"
BASE_LOG_DIR="${4:-${REPO_PATH}/logs/libero_spatial_offline_grpo_openpi_pi05_flow_noise_subset_chain_wm-step2-cosinelr-subsetaware-base90_oneshot_new10_old5_wm/libero_spatial_offline_grpo_openpi_pi05_flow_noise_subset1_wm/eval_logs_l1}"
shift_count=$#
if (( shift_count > 4 )); then
    shift_count=4
fi
shift "${shift_count}"
DENOISE_STEP="${DENOISE_STEP:-2}"
if [[ $# -gt 0 && "${1}" =~ ^[0-9]+$ ]]; then
    DENOISE_STEP="${1}"
    shift
fi

BASE_CKPTPATH="${REPO_PATH}/logs/libero_spatial_offline_grpo_openpi_pi05_flow_noise_subset_chain_wm-step2-cosinelr-subsetaware-base90_oneshot_new10_old5_wm/libero_spatial_offline_grpo_openpi_pi05_flow_noise_subset1_wm/libero_spatial_offline_grpo_openpi_pi05_flow_noise_subset1_wm/checkpoints"

DEFAULT_CKPTS=(
    "${BASE_CKPTPATH}/global_step_11500"
    "${BASE_CKPTPATH}/global_step_11000"
    "${BASE_CKPTPATH}/global_step_10500"
    "${BASE_CKPTPATH}/global_step_10000"
    "${BASE_CKPTPATH}/global_step_9500"
    "${BASE_CKPTPATH}/global_step_9000"
    "${BASE_CKPTPATH}/global_step_8500"
)

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
    echo "Missing eval script: ${EVAL_SCRIPT}"
    exit 1
fi

EXTRA_EVAL_ARGS=()
CKPT_ARGS=()
while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--" ]]; then
        shift
        EXTRA_EVAL_ARGS=("$@")
        break
    fi
    CKPT_ARGS+=("$1")
    shift
done

if [[ ${#CKPT_ARGS[@]} -gt 0 ]]; then
    CKPT_INPUTS=("${CKPT_ARGS[@]}")
else
    CKPT_INPUTS=("${DEFAULT_CKPTS[@]}")
fi

resolve_ckpt_file() {
    local input_path="$1"
    if [[ "${input_path}" == "null" ]]; then
        echo "null"
        return 0
    fi
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
    if [[ "${ckpt_file}" == "null" ]]; then
        echo "sft_null"
        return 0
    fi
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
echo "  denoise_step: ${DENOISE_STEP}"
echo "  base_log_dir: ${BASE_LOG_DIR}"
echo "  ckpt_count  : ${#CKPT_INPUTS[@]}"
echo "  extra_args   : ${EXTRA_EVAL_ARGS[*]:-}"
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

    PER_CKPT_EXTRA_EVAL_ARGS=()
    for extra_arg in "${EXTRA_EVAL_ARGS[@]}"; do
        extra_arg="${extra_arg//\{step_tag\}/${step_tag}}"
        extra_arg="${extra_arg//\{run_group_tag\}/${RUN_GROUP_TAG}}"
        PER_CKPT_EXTRA_EVAL_ARGS+=("${extra_arg}")
    done

    echo
    echo "------------------------------------------------------------"
    echo "[RUN] ${step_tag}"
    echo "  ckpt_file=${ckpt_file}"
    echo "  log_root =${per_ckpt_log_root}"
    echo "  extra_args=${PER_CKPT_EXTRA_EVAL_ARGS[*]:-}"
    echo "------------------------------------------------------------"

    if ! bash "${EVAL_SCRIPT}" \
        "${SUBSET_NAME}" \
        "${ckpt_file}" \
        "${CONFIG_NAME}" \
        "${WORLD_SIZE}" \
        "${per_ckpt_log_root}" \
        "${DENOISE_STEP}" \
        "${PER_CKPT_EXTRA_EVAL_ARGS[@]}"; then
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
