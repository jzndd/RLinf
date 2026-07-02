#! /bin/bash

set -euo pipefail
SCRIPT_VERSION="2026-04-22-r2"

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH="$(dirname "$(dirname "$EMBODIED_PATH")")"
export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"
export CALC_SR_SCRIPT="${EMBODIED_PATH}/calc_sr_from_flags.py"

export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

SUBSET_NAME="${1:-subset1}"
CKPT_PATH="${2:-null}"
CONFIG_NAME="${3:-libero_spatial_grpo_openpi_pi05_noise}"
WORLD_SIZE="${4:-8}"
BASE_LOG_DIR="${5:-${REPO_PATH}/logs/eval_libero_spatial_subset_tasks}"

DENOISE_STEP=${6:-3}

# Canonical user-facing task order for libero_spatial black-bowl tasks.
ORDERED_TASK_DESCS=(
    "pick up the black bowl between the plate and the ramekin and place it on the plate"
    "pick up the black bowl next to the ramekin and place it on the plate"
    "pick up the black bowl from table center and place it on the plate"
    "pick up the black bowl on the cookie box and place it on the plate"
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate"
    "pick up the black bowl on the ramekin and place it on the plate"
    "pick up the black bowl next to the cookie box and place it on the plate"
    "pick up the black bowl on the stove and place it on the plate"
    "pick up the black bowl next to the plate and place it on the plate"
    "pick up the black bowl on the wooden cabinet and place it on the plate"
)

if [[ -z "${SUBSET_NAME}" || -z "${CKPT_PATH}" ]]; then
    echo "Usage: $0 <subset_name> <checkpoint_path> [config_name] [world_size] [base_log_dir]"
    echo "Example:"
    echo "  $0 subset1 /path/to/full_weights.pt libero_spatial_grpo_openpi_pi05_noise 8"
    exit 1
fi

if [[ "${CKPT_PATH}" != "null" && ! -f "${CKPT_PATH}" ]]; then
    echo "Checkpoint not found: ${CKPT_PATH}"
    exit 1
fi

case "${SUBSET_NAME}" in
    subset1|libero_spatial_fullshot_subset1) MAX_TASK_ID=5 ;;
    subset2|libero_spatial_fullshot_subset2) MAX_TASK_ID=6 ;;
    subset3|libero_spatial_fullshot_subset3) MAX_TASK_ID=7 ;;
    subset4|libero_spatial_fullshot_subset4) MAX_TASK_ID=8 ;;
    subset5|libero_spatial_fullshot_subset5) MAX_TASK_ID=9 ;;
    *)
        echo "Unsupported subset '${SUBSET_NAME}'. Use subset1..subset5."
        exit 1
        ;;
esac

RUN_TAG="$(date +'%Y%m%d-%H%M%S')_${SUBSET_NAME}"
LOG_ROOT="${BASE_LOG_DIR}/${RUN_TAG}"
mkdir -p "${LOG_ROOT}"

echo "============================================================"
echo "Subset eval config"
echo "  script_ver  : ${SCRIPT_VERSION}"
echo "  subset_name : ${SUBSET_NAME}"
echo "  max_user_task_id : ${MAX_TASK_ID}"
echo "  config_name : ${CONFIG_NAME}"
echo "  ckpt_path   : ${CKPT_PATH}"
echo "  world_size  : ${WORLD_SIZE}"
echo "  log_root    : ${LOG_ROOT}"
echo "============================================================"

# Step 1: Build user_task_id -> (benchmark_task_id, reset_state_ids) mapping.
TASK_RECORDS="$(
python - <<'PY' "${MAX_TASK_ID}" "${ORDERED_TASK_DESCS[@]}"
import json
import sys
import io
import numpy as np
from contextlib import redirect_stdout

from rlinf.envs.libero.utils import get_benchmark_overridden

max_user_task_id = int(sys.argv[1])
ordered_task_descs = sys.argv[2:]
with redirect_stdout(io.StringIO()):
    suite = get_benchmark_overridden("libero_spatial")()

if max_user_task_id >= len(ordered_task_descs):
    raise ValueError(
        f"max_user_task_id={max_user_task_id} exceeds provided ordered_task_descs size={len(ordered_task_descs)}"
    )

trial_bins = []
for task_id in range(suite.get_num_tasks()):
    trial_bins.append(len(suite.get_task_init_states(task_id)))

cumsum = np.cumsum(trial_bins)
desc_to_task_id = {
    suite.get_task(task_id).language: task_id for task_id in range(suite.get_num_tasks())
}
for user_task_id in range(max_user_task_id + 1):
    task_desc = ordered_task_descs[user_task_id]
    if task_desc not in desc_to_task_id:
        raise KeyError(f"Task description not found in libero_spatial benchmark: {task_desc}")
    benchmark_task_id = int(desc_to_task_id[task_desc])
    start = 0 if benchmark_task_id == 0 else int(cumsum[benchmark_task_id - 1])
    end = int(cumsum[benchmark_task_id])
    reset_ids = list(range(start, end))
    # TSV: user_task_id, benchmark_task_id, task_desc, reset_ids_json, reset_count
    print(
        f"{user_task_id}\t{benchmark_task_id}\t{task_desc}\t{json.dumps(reset_ids, separators=(',', ':'))}\t{len(reset_ids)}"
    )
PY
)"

TASK_RECORDS="$(printf '%s\n' "${TASK_RECORDS}" | awk -F '\t' '$1 ~ /^[0-9]+$/')"
if [[ -z "${TASK_RECORDS}" ]]; then
    echo "No valid task mapping records were produced. Abort."
    exit 1
fi

echo
echo "User task to benchmark task mapping (from libero_spatial benchmark):"
echo "${TASK_RECORDS}" | while IFS=$'\t' read -r user_task_id benchmark_task_id task_desc reset_ids_json reset_count; do
    if [[ ! "${user_task_id}" =~ ^[0-9]+$ ]]; then
        continue
    fi
    echo "  user_task ${user_task_id} -> benchmark_task ${benchmark_task_id} (${reset_count} ids): ${task_desc}"
    echo "    reset_state_ids=${reset_ids_json}"
done
echo

SUMMARY_FILE="${LOG_ROOT}/per_task_sr_summary.csv"
echo "user_task_id,benchmark_task_id,eval_envs,reset_id_count,success_rate,log_path,task_description" > "${SUMMARY_FILE}"

# Step 2: Evaluate each task with only its reset_state_ids.
while IFS=$'\t' read -r user_task_id benchmark_task_id task_desc reset_ids_json reset_count; do
    if [[ ! "${user_task_id}" =~ ^[0-9]+$ ]]; then
        continue
    fi
    if [[ ! "${reset_count}" =~ ^[0-9]+$ ]]; then
        echo "Skip malformed record: task_id=${task_id}, reset_count=${reset_count}"
        continue
    fi
    eval_envs=$(( ((reset_count + WORLD_SIZE - 1) / WORLD_SIZE) * WORLD_SIZE ))
    task_log_dir="${LOG_ROOT}/task_${user_task_id}"
    task_log_file="${task_log_dir}/eval_embodiment.log"
    mkdir -p "${task_log_dir}"

    echo "------------------------------------------------------------"
    echo "Evaluating user_task ${user_task_id} (benchmark_task ${benchmark_task_id})"
    echo "  desc         : ${task_desc}"
    echo "  reset_ids    : ${reset_count}"
    echo "  total_num_envs(rounded to /${WORLD_SIZE}): ${eval_envs}"
    echo "  log          : ${task_log_file}"

    python "${SRC_FILE}" \
        --config-path "${EMBODIED_PATH}/config/" \
        --config-name "${CONFIG_NAME}" \
        runner.logger.log_path="${task_log_dir}" \
        runner.ckpt_path="${CKPT_PATH}" \
        actor.model.num_steps="${DENOISE_STEP}" \
        env.eval.total_num_envs="${eval_envs}" \
        env.eval.use_ordered_reset_state_ids=True \
        env.eval.eval_reset_state_ids="${reset_ids_json}" \
        2>&1 | tee "${task_log_file}"

    flags_csv="${task_log_dir}/per_episode_flags.csv"
    if [[ -f "${flags_csv}" && -f "${CALC_SR_SCRIPT}" ]]; then
        task_sr="$(
        python "${CALC_SR_SCRIPT}" \
            --csv "${flags_csv}" \
            --target-ids-json "${reset_ids_json}" | awk -F '=' '/^sr=/{print $2}'
        )"
        if [[ -z "${task_sr}" ]]; then
            task_sr="nan"
        fi
    else
        task_sr="$(
    python - <<'PY' "${task_log_file}"
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
text = log_path.read_text(encoding="utf-8", errors="ignore")
matches = re.findall(r"(?:'|\")eval/success_once(?:'|\")\s*:\s*([0-9eE+.\-]+)", text)
if not matches:
    print("nan")
else:
    print(matches[-1])
PY
        )"
    fi

    echo "  task_${user_task_id} success_rate=${task_sr}"
    echo "${user_task_id},${benchmark_task_id},${eval_envs},${reset_count},${task_sr},${task_log_file},\"${task_desc}\"" >> "${SUMMARY_FILE}"
done <<< "${TASK_RECORDS}"

echo
echo "Per-task SR summary:"
cat "${SUMMARY_FILE}"
echo
echo "Done. Summary saved to: ${SUMMARY_FILE}"
