#! /bin/bash
set -euo pipefail

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_offline_grpo_openpi.py"

export PYTHONPATH="${PYTHONPATH:-}"  # activate reads PYTHONPATH; keep set -u safe.
source /opt/venv/openpi/bin/activate

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
unset RAY_ADDRESS RAY_TMPDIR
# Ray's address="auto" can otherwise pick up a dead /tmp/ray session and wait
# for many minutes before falling back to a local cluster.
ray stop --force >/dev/null 2>&1 || true
rm -rf /tmp/ray

CONFIG_NAME="libero_spatial_offline_grpo_openpi_pi05_noise_subset"
CHAIN_NAME="libero_spatial_offline_grpo_openpi_pi05_flow_noise_subset_chain"
FIRST_STAGE_MAX_EPOCHS=6000
LOG_ROOT="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CHAIN_NAME}-step2-lr2e-6"
CHAIN_LOG_FILE="${LOG_ROOT}/run_offline_grpo_openpi_noise_subset_chain.log"

denoise_step=2

DATA_PATHS=(
    "/mnt/project_rlinf/jzn/workspace/openpi/data/libero_spatial_fullshot_subset1"
    # "/mnt/project_rlinf/jzn/workspace/openpi/data/libero_spatial_fullshot_subset2"
    # "/mnt/project_rlinf/jzn/workspace/openpi/data/libero_spatial_fullshot_subset3"
    # "/mnt/project_rlinf/jzn/workspace/openpi/data/libero_spatial_fullshot_subset4"
    # "/mnt/project_rlinf/jzn/workspace/openpi/data/libero_spatial_fullshot_subset5"
)

EXPERIMENT_NAMES=(
    "libero_spatial_offline_grpo_openpi_pi05_flow_noise_subset1"
    # "libero_spatial_offline_grpo_openpi_pi05_flow_noise_subset2"
    # "libero_spatial_offline_grpo_openpi_pi05_flow_noise_subset3"
    # "libero_spatial_offline_grpo_openpi_pi05_flow_noise_subset4"
    # "libero_spatial_offline_grpo_openpi_pi05_flow_noise_subset5"
)

mkdir -p "${LOG_ROOT}"
touch "${CHAIN_LOG_FILE}"

log() {
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[${timestamp}] $*" | tee -a "${CHAIN_LOG_FILE}"
}

find_latest_checkpoint_path() {
    local experiment_name="$1"
    local checkpoints_dir="${LOG_ROOT}/${experiment_name}/${experiment_name}/checkpoints"
    local latest_checkpoint_dir
    latest_checkpoint_dir="$(find "${checkpoints_dir}" -maxdepth 1 -mindepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)"

    if [[ -z "${latest_checkpoint_dir}" ]]; then
        log "Failed to find any checkpoint directory under ${checkpoints_dir}."
        return 1
    fi

    local checkpoint_path="${latest_checkpoint_dir}/actor/model_state_dict/full_weights.pt"
    if [[ ! -f "${checkpoint_path}" ]]; then
        log "Checkpoint file does not exist: ${checkpoint_path}"
        return 1
    fi

    printf '%s\n' "${checkpoint_path}"
}

run_one_stage() {
    local stage_index="$1"
    local experiment_name="$2"
    local data_path="$3"
    local ckpt_path="$4"
    local stage_log_file="${LOG_ROOT}/${experiment_name}.stdout.log"
    local stage_log_root="${LOG_ROOT}/${experiment_name}"

    log "Starting stage ${stage_index}: experiment_name=${experiment_name} data.train_data_paths=${data_path} runner.ckpt_path=${ckpt_path}"

    local -a cmd=(
        python "${SRC_FILE}"
        --config-path "${EMBODIED_PATH}/config/"
        --config-name "${CONFIG_NAME}"
        runner.logger.log_path="${stage_log_root}"
        runner.logger.experiment_name="${experiment_name}"
        data.train_data_paths="${data_path}"
        runner.ckpt_path="${ckpt_path}"
	    actor.model.num_steps="${denoise_step}"
    )

    if [[ "$#" -gt 4 ]]; then
        cmd+=("${@:5}")
    fi

    printf '%q ' "${cmd[@]}" | tee -a "${CHAIN_LOG_FILE}"
    printf '\n' | tee -a "${CHAIN_LOG_FILE}"

    "${cmd[@]}" 2>&1 | tee -a "${stage_log_file}"
}

log "Using Python at $(which python)"
log "Logs will be written to ${LOG_ROOT}"

prev_ckpt_path=null
for idx in "${!DATA_PATHS[@]}"; do
    stage_num=$((idx + 1))
    experiment_name="${EXPERIMENT_NAMES[idx]}"
    data_path="${DATA_PATHS[idx]}"

    if [[ "${stage_num}" -eq 1 ]]; then
        run_one_stage \
            "${stage_num}" \
            "${experiment_name}" \
            "${data_path}" \
            "${prev_ckpt_path}" \
            "$@" \
            runner.max_epochs="${FIRST_STAGE_MAX_EPOCHS}"
    else
        run_one_stage "${stage_num}" "${experiment_name}" "${data_path}" "${prev_ckpt_path}" "$@"
    fi

    prev_ckpt_path="$(find_latest_checkpoint_path "${experiment_name}")"
    log "Stage ${stage_num} finished. Next runner.ckpt_path will use ${prev_ckpt_path}"
done

log "All ${#DATA_PATHS[@]} subset stage(s) finished successfully. Final checkpoint: ${prev_ckpt_path}"
