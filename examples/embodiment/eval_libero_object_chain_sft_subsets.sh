#!/bin/bash

set -uo pipefail

EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$(dirname "${EMBODIED_PATH}")")"
EVAL_SCRIPT="${EMBODIED_PATH}/eval_libero_object_subset_tasks_use_model_path.sh"

CONFIG_NAME="${1:-libero_object_grpo_openpi_pi05_noise}"
WORLD_SIZE="${2:-8}"
BASE_LOG_DIR="${3:-${REPO_PATH}/logs/eval_libero_object_chain_sft}"
DENOISE_STEP="${4:-2}"

OPENPI_ROOT="${OPENPI_ROOT:-/mnt/project_rlinf/jzn/workspace/openpi}"
START_SUBSET="${START_SUBSET:-1}"
END_SUBSET="${END_SUBSET:-5}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_MISSING="${ALLOW_MISSING:-0}"

CONFIG_PREFIX="pi05_libero_object_subset"
EXP_PREFIX="libero_object_chain_subset_new10_old5_"

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
    echo "Missing eval script: ${EVAL_SCRIPT}"
    exit 1
fi

if [[ ! "${START_SUBSET}" =~ ^[1-5]$ || ! "${END_SUBSET}" =~ ^[1-5]$ || "${START_SUBSET}" -gt "${END_SUBSET}" ]]; then
    echo "Invalid subset range: START_SUBSET=${START_SUBSET}, END_SUBSET=${END_SUBSET}. Use 1..5."
    exit 1
fi

latest_chain_checkpoint() {
    local subset="$1"
    local exp_dir="${OPENPI_ROOT}/checkpoints/sft/${CONFIG_PREFIX}${subset}/${EXP_PREFIX}${subset}"
    local entry base latest=""

    [[ -d "${exp_dir}" ]] || return 1

    for entry in "${exp_dir}"/*; do
        [[ -d "${entry}" ]] || continue
        base="$(basename "${entry}")"
        [[ "${base}" =~ ^[0-9]+$ ]] || continue
        [[ -f "${entry}/model.safetensors" ]] || continue
        if [[ -z "${latest}" || "${base}" -gt "${latest}" ]]; then
            latest="${base}"
        fi
    done

    [[ -n "${latest}" ]] || return 1
    printf '%s/%s\n' "${exp_dir}" "${latest}"
}

mkdir -p "${BASE_LOG_DIR}"
RUN_GROUP_TAG="$(date +'%Y%m%d-%H%M%S')_libero_object_chain_sft"

echo "============================================================"
echo "LIBERO object chain SFT subset eval"
echo "  config_name : ${CONFIG_NAME}"
echo "  world_size  : ${WORLD_SIZE}"
echo "  denoise_step: ${DENOISE_STEP}"
echo "  openpi_root : ${OPENPI_ROOT}"
echo "  subset_range: ${START_SUBSET}..${END_SUBSET}"
echo "  base_log_dir: ${BASE_LOG_DIR}"
echo "  dry_run     : ${DRY_RUN}"
echo "============================================================"

fail_count=0
for subset in $(seq "${START_SUBSET}" "${END_SUBSET}"); do
    subset_name="subset${subset}"
    exp_name="${EXP_PREFIX}${subset}"

    if ! model_path="$(latest_chain_checkpoint "${subset}")"; then
        echo
        echo "[MISSING] ${exp_name}: no numeric checkpoint with model.safetensors found."
        echo "  expected under ${OPENPI_ROOT}/checkpoints/sft/${CONFIG_PREFIX}${subset}/${exp_name}"
        if [[ "${ALLOW_MISSING}" == "1" ]]; then
            continue
        fi
        fail_count=$((fail_count + 1))
        continue
    fi

    step_tag="step_$(basename "${model_path}")"
    per_subset_log_root="${BASE_LOG_DIR}/${RUN_GROUP_TAG}/${exp_name}_${step_tag}"

    echo
    echo "------------------------------------------------------------"
    echo "[RUN] ${exp_name} -> ${subset_name}"
    echo "  model_path=${model_path}"
    echo "  log_root  =${per_subset_log_root}"
    echo "------------------------------------------------------------"

    cmd=(
        bash "${EVAL_SCRIPT}"
        "${subset_name}"
        "${model_path}"
        "${CONFIG_NAME}"
        "${WORLD_SIZE}"
        "${per_subset_log_root}"
        "${DENOISE_STEP}"
    )

    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'command='
        printf '%q ' "${cmd[@]}"
        printf '\n'
        continue
    fi

    if ! "${cmd[@]}"; then
        echo "[FAIL] ${exp_name} -> ${subset_name}"
        fail_count=$((fail_count + 1))
        continue
    fi

    echo "[DONE] ${exp_name} -> ${subset_name}"
done

echo
if [[ "${fail_count}" -gt 0 ]]; then
    echo "Batch completed with ${fail_count} failure(s)."
    exit 1
fi

echo "Batch completed successfully."
