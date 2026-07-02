#! /bin/bash
set -euo pipefail

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_offline_grpo_openpi.py"

source /opt/venv/openpi/bin/activate

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false

CONFIG_NAME="libero_10_offline_grpo_openpi_pi05_noise_subset_base90_oneshot"
CHAIN_NAME="libero_10_offline_grpo_openpi_pi05_flow_noise_subset_chain"
FIRST_STAGE_MAX_EPOCHS=12000
LOG_ROOT="${LOG_ROOT:-${REPO_PATH}/logs_long/${CHAIN_NAME}-step2-cosinelr5e-6_1e-6-base90_oneshot_new10_old5}"
CHAIN_LOG_FILE="${LOG_ROOT}/run_offline_grpo_openpi_noise_subset_chain.log"

denoise_step=2
FULL_DATA_PATH="/mnt/project_rlinf/jzn/workspace/openpi/data/libero_long_fullshot"
NEW_TASK_TRAJ="${NEW_TASK_TRAJ:-${new_task_traj:-10}}"
OLD_TASK_TRAJ="${OLD_TASK_TRAJ:-${old_task_traj:-5}}"
CONTINUAL_REPLAY_SEED="${CONTINUAL_REPLAY_SEED:-${continual_replay_seed:-0}}"
IF_OFFLINE_EVAL="${IF_OFFLINE_EVAL:-${if_offline_eval:-True}}"
EVAL_STEP="${EVAL_STEP:-${eval_step:-null}}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${eval_batch_size:-32}}"

DATA_PATHS=(
    "${FULL_DATA_PATH}"
)

EXPERIMENT_NAMES=(
    # "libero_10_offline_grpo_openpi_pi05_flow_noise_subset1"
    # "libero_10_offline_grpo_openpi_pi05_flow_noise_subset2"
    # "libero_10_offline_grpo_openpi_pi05_flow_noise_subset3"
    # "libero_10_offline_grpo_openpi_pi05_flow_noise_subset4"
    "libero_10_offline_grpo_openpi_pi05_flow_noise_subset5"
)

SUBSET_IDS=(
    # 1
    # 2
    # 3
    # 4
    5
)

mkdir -p "${LOG_ROOT}"
touch "${CHAIN_LOG_FILE}"

SCRIPT_EXTRA_ARGS=()
for arg in "$@"; do
    case "${arg}" in
        if_offline_eval=*) IF_OFFLINE_EVAL="${arg#*=}" ;;
        eval_step=*) EVAL_STEP="${arg#*=}" ;;
        eval_batch_size=*) EVAL_BATCH_SIZE="${arg#*=}" ;;
        *) SCRIPT_EXTRA_ARGS+=("${arg}") ;;
    esac
done
set -- "${SCRIPT_EXTRA_ARGS[@]}"

log() {
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[${timestamp}] $*" | tee -a "${CHAIN_LOG_FILE}"
}

find_latest_resume_dir() {
    local experiment_name="$1"
    local run_dir="${LOG_ROOT}/${experiment_name}/${experiment_name}"
    local checkpoints_dir="${run_dir}/checkpoints"
    local latest_resume_dir

    if [[ ! -d "${run_dir}" ]]; then
        return 1
    fi

    if [[ ! -d "${checkpoints_dir}" ]]; then
        return 1
    fi

    latest_resume_dir="$(find "${checkpoints_dir}" -maxdepth 1 -mindepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)"
    if [[ -z "${latest_resume_dir}" ]]; then
        return 1
    fi

    printf '%s\n' "${latest_resume_dir}"
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
    local subset_id="$4"
    local ckpt_path="$5"
    local stage_log_file="${LOG_ROOT}/${experiment_name}.stdout.log"
    local stage_log_root="${LOG_ROOT}/${experiment_name}"
    local resume_dir=""

    if resume_dir="$(find_latest_resume_dir "${experiment_name}")"; then
        if [[ -d "${resume_dir}/actor" ]]; then
            log "Found existing checkpoint for stage ${stage_index}; runner.resume_dir=${resume_dir}"
        else
            log "Latest resume_dir ${resume_dir} is missing actor checkpoint directory; skip resume."
            resume_dir=""
        fi
    else
        log "No existing checkpoint found for stage ${stage_index}; starting from runner.ckpt_path=${ckpt_path}"
    fi

    log "Starting stage ${stage_index}: experiment_name=${experiment_name} data.train_data_paths=${data_path} continual_subset_id=${subset_id} new_task_traj=${NEW_TASK_TRAJ} old_task_traj=${OLD_TASK_TRAJ} runner.ckpt_path=${ckpt_path} runner.resume_dir=${resume_dir:-null}"

    local -a cmd=(
        python "${SRC_FILE}"
        --config-path "${EMBODIED_PATH}/config/"
        --config-name "${CONFIG_NAME}"
        runner.logger.log_path="${stage_log_root}"
        runner.logger.experiment_name="${experiment_name}"
        data.train_data_paths="${data_path}"
        data.use_continual_learning=True
        data.continual_subset_id="${subset_id}"
        data.new_task_traj="${NEW_TASK_TRAJ}"
        data.old_task_traj="${OLD_TASK_TRAJ}"
        data.continual_replay_seed="${CONTINUAL_REPLAY_SEED}"
        data.if_offline_eval="${IF_OFFLINE_EVAL}"
        data.eval_step="${EVAL_STEP}"
        data.eval_batch_size="${EVAL_BATCH_SIZE}"
        runner.ckpt_path="${ckpt_path}"
        actor.model.num_steps="${denoise_step}"
    )

    if [[ -n "${resume_dir}" ]]; then
        cmd+=(runner.resume_dir="${resume_dir}")
    fi

    if [[ "$#" -gt 5 ]]; then
        cmd+=("${@:6}")
    fi

    printf '%q ' "${cmd[@]}" | tee -a "${CHAIN_LOG_FILE}"
    printf '\n' | tee -a "${CHAIN_LOG_FILE}"

    "${cmd[@]}" 2>&1 | tee -a "${stage_log_file}"
}

log "Using Python at $(which python)"
log "Logs will be written to ${LOG_ROOT}"
log "Continual data root: ${FULL_DATA_PATH}"
log "Continual selection: new_task_traj=${NEW_TASK_TRAJ}, old_task_traj=${OLD_TASK_TRAJ}, replay_seed=${CONTINUAL_REPLAY_SEED}"
log "Offline eval: if_offline_eval=${IF_OFFLINE_EVAL}, eval_step=${EVAL_STEP}, eval_batch_size=${EVAL_BATCH_SIZE}"

prev_ckpt_path="/mnt/project_rlinf/jzn/workspace/continual_learning/RLinf/logs_long/libero_10_offline_grpo_openpi_pi05_flow_noise_subset_chain-step2-cosinelr5e-6_1e-6-base90_oneshot_new10_old5/libero_10_offline_grpo_openpi_pi05_flow_noise_subset4/libero_10_offline_grpo_openpi_pi05_flow_noise_subset4/checkpoints/global_step_6500/actor/model_state_dict/full_weights.pt"
for idx in "${!DATA_PATHS[@]}"; do
    stage_num=$((idx + 1))
    experiment_name="${EXPERIMENT_NAMES[idx]}"
    data_path="${DATA_PATHS[idx]}"
    subset_id="${SUBSET_IDS[idx]}"

    if [[ "${stage_num}" -eq 1 ]]; then
        run_one_stage \
            "${stage_num}" \
            "${experiment_name}" \
            "${data_path}" \
            "${subset_id}" \
            "${prev_ckpt_path}" \
            "$@" \
            runner.max_epochs="${FIRST_STAGE_MAX_EPOCHS}"
    else
        run_one_stage "${stage_num}" "${experiment_name}" "${data_path}" "${subset_id}" "${prev_ckpt_path}" "$@"
    fi

    prev_ckpt_path="$(find_latest_checkpoint_path "${experiment_name}")"
    log "Stage ${stage_num} finished. Next runner.ckpt_path will use ${prev_ckpt_path}"
done

log "Selected subset stages finished successfully. Final checkpoint: ${prev_ckpt_path}"
