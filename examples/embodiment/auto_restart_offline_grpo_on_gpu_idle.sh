#! /bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
TARGET_SCRIPT="${SCRIPT_DIR}/run_offline_grpo_openpi.sh"
WATCHDOG_LOG_FILE="${WATCHDOG_LOG_FILE:-${SCRIPT_DIR}/gpu_idle_watchdog.log}"
MONITORED_GPUS="${MONITORED_GPUS:-0,1,2,3}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
IDLE_THRESHOLD_SECONDS="${IDLE_THRESHOLD_SECONDS:-1800}"
GPU_UTIL_THRESHOLD="${GPU_UTIL_THRESHOLD:-5}"
STARTUP_GRACE_SECONDS="${STARTUP_GRACE_SECONDS:-600}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -x "${TARGET_SCRIPT}" ]]; then
    echo "Target script is not executable: ${TARGET_SCRIPT}" >&2
    exit 1
fi

declare -A MONITORED_GPU_SET=()
IFS=',' read -r -a GPU_ID_ARRAY <<< "${MONITORED_GPUS}"
for raw_gpu_id in "${GPU_ID_ARRAY[@]}"; do
    gpu_id="${raw_gpu_id// /}"
    if [[ -n "${gpu_id}" ]]; then
        MONITORED_GPU_SET["${gpu_id}"]=1
    fi
done

if [[ "${#MONITORED_GPU_SET[@]}" -eq 0 ]]; then
    echo "No GPUs configured in MONITORED_GPUS." >&2
    exit 1
fi

mkdir -p "$(dirname "${WATCHDOG_LOG_FILE}")"
touch "${WATCHDOG_LOG_FILE}"

log() {
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[${timestamp}] $*" | tee -a "${WATCHDOG_LOG_FILE}"
}

is_target_run_active() {
    if pgrep -af "bash ${TARGET_SCRIPT}" >/dev/null; then
        return 0
    fi

    if pgrep -af "python .*train_offline_grpo_openpi.py" >/dev/null; then
        return 0
    fi

    return 1
}

has_any_compute_app() {
    local line
    while IFS= read -r line; do
        if [[ -n "${line// /}" ]]; then
            return 0
        fi
    done < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)

    return 1
}

all_monitored_gpus_idle() {
    local line
    local seen_gpu=0

    while IFS=',' read -r raw_index raw_util _; do
        local gpu_index="${raw_index// /}"
        local gpu_util="${raw_util// /}"

        if [[ -z "${MONITORED_GPU_SET[${gpu_index}]+x}" ]]; then
            continue
        fi

        seen_gpu=1
        if (( gpu_util > GPU_UTIL_THRESHOLD )); then
            return 1
        fi
    done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits)

    if (( seen_gpu == 0 )); then
        log "No monitored GPUs were found in nvidia-smi output."
        return 1
    fi

    return 0
}

launch_target_run() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        log "DRY_RUN=1, skipping launch: bash ${TARGET_SCRIPT}"
        return 0
    fi

    nohup bash "${TARGET_SCRIPT}" >> "${WATCHDOG_LOG_FILE}" 2>&1 &
    log "Started offline GRPO launcher with PID $!."
}

idle_seconds=0
last_launch_epoch=0

log "GPU idle watchdog started."
log "MONITORED_GPUS=${MONITORED_GPUS} CHECK_INTERVAL_SECONDS=${CHECK_INTERVAL_SECONDS} IDLE_THRESHOLD_SECONDS=${IDLE_THRESHOLD_SECONDS} GPU_UTIL_THRESHOLD=${GPU_UTIL_THRESHOLD} STARTUP_GRACE_SECONDS=${STARTUP_GRACE_SECONDS}"

while true; do
    current_epoch="$(date +%s)"

    if is_target_run_active; then
        if (( idle_seconds != 0 )); then
            log "Detected active offline GRPO run, resetting idle timer."
        fi
        idle_seconds=0
        sleep "${CHECK_INTERVAL_SECONDS}"
        continue
    fi

    if (( current_epoch - last_launch_epoch < STARTUP_GRACE_SECONDS )); then
        sleep "${CHECK_INTERVAL_SECONDS}"
        continue
    fi

    if has_any_compute_app; then
        if (( idle_seconds != 0 )); then
            log "Detected compute apps on GPU, resetting idle timer."
        fi
        idle_seconds=0
        sleep "${CHECK_INTERVAL_SECONDS}"
        continue
    fi

    if all_monitored_gpus_idle; then
        idle_seconds=$((idle_seconds + CHECK_INTERVAL_SECONDS))
        log "All monitored GPUs are idle for ${idle_seconds}/${IDLE_THRESHOLD_SECONDS} seconds."

        if (( idle_seconds >= IDLE_THRESHOLD_SECONDS )); then
            launch_target_run
            last_launch_epoch="$(date +%s)"
            idle_seconds=0
        fi
    else
        if (( idle_seconds != 0 )); then
            log "GPU utilization exceeded threshold, resetting idle timer."
        fi
        idle_seconds=0
    fi

    sleep "${CHECK_INTERVAL_SECONDS}"
done
