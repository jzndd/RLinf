#! /bin/bash

set -euo pipefail
SCRIPT_VERSION="2026-07-04-ckpt-single-eval-per-subset"

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH="$(dirname "$(dirname "$EMBODIED_PATH")")"
export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"
export CALC_SR_SCRIPT="${EMBODIED_PATH}/calc_sr_from_flags.py"

export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

BENCHMARK_NAME="${LIBERO_BENCHMARK_NAME:-}"
SUBSET_NAME="${1:-subset1}"
CKPT_PATH="${2:-null}"
default_config_name() {
    case "${BENCHMARK_NAME}" in
        libero_spatial) echo "libero_spatial_grpo_openpi_pi05_noise" ;;
        libero_goal) echo "libero_goal_grpo_openpi_pi05_noise_base90_oneshot" ;;
        libero_object) echo "libero_object_grpo_openpi_pi05_noise" ;;
        libero_10) echo "libero_10_grpo_openpi_pi05_noise" ;;
        *) echo "" ;;
    esac
}

CONFIG_NAME="${3:-$(default_config_name)}"
WORLD_SIZE="${4:-8}"
BASE_LOG_DIR="${5:-${REPO_PATH}/logs/eval_${BENCHMARK_NAME}_subset_tasks}"
DENOISE_STEP="${6:-${LIBERO_DEFAULT_DENOISE_STEP:-2}}"
EXTRA_HYDRA_ARGS=("${@:7}")

if [[ -z "${BENCHMARK_NAME}" ]]; then
    echo "LIBERO_BENCHMARK_NAME is required. Use a benchmark wrapper script."
    exit 1
fi

if [[ -z "${SUBSET_NAME}" || -z "${CKPT_PATH}" || -z "${CONFIG_NAME}" ]]; then
    echo "Usage: $0 <subset_name> <checkpoint_path> [config_name] [world_size] [base_log_dir] [denoise_step]"
    echo "Example:"
    echo "  $0 subset1 /path/to/full_weights.pt ${BENCHMARK_NAME}_grpo_openpi_pi05_noise 8"
    exit 1
fi

if [[ ! "${WORLD_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "world_size must be a positive integer, got: ${WORLD_SIZE}"
    exit 1
fi

if [[ "${CKPT_PATH}" != "null" && ! -f "${CKPT_PATH}" ]]; then
    echo "Checkpoint not found: ${CKPT_PATH}"
    exit 1
fi

ordered_task_descs() {
    case "${BENCHMARK_NAME}" in
        libero_spatial)
            cat <<'EOF_TASKS'
pick up the black bowl between the plate and the ramekin and place it on the plate
pick up the black bowl next to the ramekin and place it on the plate
pick up the black bowl from table center and place it on the plate
pick up the black bowl on the cookie box and place it on the plate
pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate
pick up the black bowl on the ramekin and place it on the plate
pick up the black bowl next to the cookie box and place it on the plate
pick up the black bowl on the stove and place it on the plate
pick up the black bowl next to the plate and place it on the plate
pick up the black bowl on the wooden cabinet and place it on the plate
EOF_TASKS
            ;;
        libero_goal)
            cat <<'EOF_TASKS'
open the middle drawer of the cabinet
put the bowl on the stove
put the wine bottle on top of the cabinet
open the top drawer and put the bowl inside
put the bowl on top of the cabinet
push the plate to the front of the stove
put the cream cheese in the bowl
turn on the stove
put the bowl on the plate
put the wine bottle on the rack
EOF_TASKS
            ;;
        libero_object)
            cat <<'EOF_TASKS'
pick up the alphabet soup and place it in the basket
pick up the cream cheese and place it in the basket
pick up the salad dressing and place it in the basket
pick up the bbq sauce and place it in the basket
pick up the ketchup and place it in the basket
pick up the tomato sauce and place it in the basket
pick up the butter and place it in the basket
pick up the milk and place it in the basket
pick up the chocolate pudding and place it in the basket
pick up the orange juice and place it in the basket
EOF_TASKS
            ;;
        libero_10)
            cat <<'EOF_TASKS'
put both the alphabet soup and the tomato sauce in the basket
put both the cream cheese box and the butter in the basket
turn on the stove and put the moka pot on it
put the black bowl in the bottom drawer of the cabinet and close it
put the white mug on the left plate and put the yellow and white mug on the right plate
pick up the book and place it in the back compartment of the caddy
put the white mug on the plate and put the chocolate pudding to the right of the plate
put both the alphabet soup and the cream cheese box in the basket
put both moka pots on the stove
put the yellow and white mug in the microwave and close it
EOF_TASKS
            ;;
        *)
            echo "Unsupported LIBERO_BENCHMARK_NAME='${BENCHMARK_NAME}'" >&2
            return 1
            ;;
    esac
}

mapfile -t ORDERED_TASK_DESCS < <(ordered_task_descs)

case "${SUBSET_NAME}" in
    subset1|${BENCHMARK_NAME}_fullshot_subset1) MAX_TASK_ID=5 ;;
    subset2|${BENCHMARK_NAME}_fullshot_subset2) MAX_TASK_ID=6 ;;
    subset3|${BENCHMARK_NAME}_fullshot_subset3) MAX_TASK_ID=7 ;;
    subset4|${BENCHMARK_NAME}_fullshot_subset4) MAX_TASK_ID=8 ;;
    subset5|${BENCHMARK_NAME}_fullshot_subset5) MAX_TASK_ID=9 ;;
    libero_long_fullshot_subset1) [[ "${BENCHMARK_NAME}" == "libero_10" ]] && MAX_TASK_ID=5 || MAX_TASK_ID="" ;;
    libero_long_fullshot_subset2) [[ "${BENCHMARK_NAME}" == "libero_10" ]] && MAX_TASK_ID=6 || MAX_TASK_ID="" ;;
    libero_long_fullshot_subset3) [[ "${BENCHMARK_NAME}" == "libero_10" ]] && MAX_TASK_ID=7 || MAX_TASK_ID="" ;;
    libero_long_fullshot_subset4) [[ "${BENCHMARK_NAME}" == "libero_10" ]] && MAX_TASK_ID=8 || MAX_TASK_ID="" ;;
    libero_long_fullshot_subset5) [[ "${BENCHMARK_NAME}" == "libero_10" ]] && MAX_TASK_ID=9 || MAX_TASK_ID="" ;;
    *) MAX_TASK_ID="" ;;
esac

if [[ -z "${MAX_TASK_ID}" ]]; then
    echo "Unsupported subset '${SUBSET_NAME}' for ${BENCHMARK_NAME}. Use subset1..subset5."
    exit 1
fi

RUN_TAG="$(date +'%Y%m%d-%H%M%S')_${SUBSET_NAME}"
LOG_ROOT="${BASE_LOG_DIR}/${RUN_TAG}"
mkdir -p "${LOG_ROOT}"

echo "============================================================"
echo "Subset eval config"
echo "  script_ver       : ${SCRIPT_VERSION}"
echo "  benchmark_name   : ${BENCHMARK_NAME}"
echo "  subset_name      : ${SUBSET_NAME}"
echo "  max_user_task_id : ${MAX_TASK_ID}"
echo "  config_name      : ${CONFIG_NAME}"
echo "  ckpt_path        : ${CKPT_PATH}"
echo "  denoise_step     : ${DENOISE_STEP}"
echo "  extra_overrides  : ${EXTRA_HYDRA_ARGS[*]:-}"
echo "  world_size       : ${WORLD_SIZE}"
echo "  log_root         : ${LOG_ROOT}"
echo "============================================================"

TASK_RECORDS="$(
python - <<'PY' "${BENCHMARK_NAME}" "${MAX_TASK_ID}" "${ORDERED_TASK_DESCS[@]}"
import json
import os
import sys
import io
import numpy as np
from contextlib import redirect_stdout

from rlinf.envs.libero.utils import get_benchmark_overridden

benchmark_name = sys.argv[1]
max_user_task_id = int(sys.argv[2])
ordered_task_descs = sys.argv[3:]
reset_ids_per_task = int(os.environ.get("RESET_IDS_PER_TASK", "0") or "0")
with redirect_stdout(io.StringIO()):
    suite = get_benchmark_overridden(benchmark_name)()

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
        raise KeyError(f"Task description not found in {benchmark_name} benchmark: {task_desc}")
    benchmark_task_id = int(desc_to_task_id[task_desc])
    start = 0 if benchmark_task_id == 0 else int(cumsum[benchmark_task_id - 1])
    end = int(cumsum[benchmark_task_id])
    reset_ids = list(range(start, end))
    if reset_ids_per_task > 0:
        reset_ids = reset_ids[:reset_ids_per_task]
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
echo "User task to benchmark task mapping (from ${BENCHMARK_NAME} benchmark):"
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

ALL_RESET_INFO="$(
TASK_RECORDS="${TASK_RECORDS}" python - <<'PY'
import json
import os

all_reset_ids = []
for line in os.environ["TASK_RECORDS"].splitlines():
    fields = line.split("\t")
    if len(fields) < 5 or not fields[0].isdigit():
        continue
    all_reset_ids.extend(int(x) for x in json.loads(fields[3]))

print(json.dumps(all_reset_ids, separators=(",", ":")))
print(len(all_reset_ids))
PY
)"
ALL_RESET_IDS_JSON="$(printf '%s\n' "${ALL_RESET_INFO}" | sed -n '1p')"
TOTAL_RESET_COUNT="$(printf '%s\n' "${ALL_RESET_INFO}" | sed -n '2p')"

if [[ -z "${ALL_RESET_IDS_JSON}" || ! "${TOTAL_RESET_COUNT}" =~ ^[0-9]+$ || "${TOTAL_RESET_COUNT}" -eq 0 ]]; then
    echo "No valid reset_state_ids were produced. Abort."
    exit 1
fi

eval_envs=$(( ((TOTAL_RESET_COUNT + WORLD_SIZE - 1) / WORLD_SIZE) * WORLD_SIZE ))
eval_log_dir="${LOG_ROOT}/all_tasks"
eval_log_file="${eval_log_dir}/eval_embodiment.log"
mkdir -p "${eval_log_dir}"

echo "------------------------------------------------------------"
echo "Evaluating ${SUBSET_NAME} once for user_task 0..${MAX_TASK_ID}"
echo "  total reset_ids: ${TOTAL_RESET_COUNT}"
echo "  total_num_envs(rounded to /${WORLD_SIZE}): ${eval_envs}"
echo "  log            : ${eval_log_file}"

python "${SRC_FILE}" \
    --config-path "${EMBODIED_PATH}/config/" \
    --config-name "${CONFIG_NAME}" \
    runner.logger.log_path="${eval_log_dir}" \
    runner.ckpt_path="${CKPT_PATH}" \
    actor.model.num_steps="${DENOISE_STEP}" \
    env.eval.total_num_envs="${eval_envs}" \
    env.eval.use_ordered_reset_state_ids=True \
    env.eval.eval_reset_state_ids="${ALL_RESET_IDS_JSON}" \
    "${EXTRA_HYDRA_ARGS[@]}" \
    2>&1 | tee "${eval_log_file}"

flags_csv="${eval_log_dir}/per_episode_flags.csv"
while IFS=$'\t' read -r user_task_id benchmark_task_id task_desc reset_ids_json reset_count; do
    if [[ ! "${user_task_id}" =~ ^[0-9]+$ ]]; then
        continue
    fi
    if [[ ! "${reset_count}" =~ ^[0-9]+$ ]]; then
        echo "Skip malformed record: user_task_id=${user_task_id}, reset_count=${reset_count}"
        continue
    fi

    echo "------------------------------------------------------------"
    echo "Computing SR for user_task ${user_task_id} (benchmark_task ${benchmark_task_id})"
    echo "  desc      : ${task_desc}"
    echo "  reset_ids : ${reset_count}"

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
        echo "  Missing ${flags_csv} or ${CALC_SR_SCRIPT}; cannot compute per-task SR from a joint eval."
        task_sr="nan"
    fi

    echo "  task_${user_task_id} success_rate=${task_sr}"
    echo "${user_task_id},${benchmark_task_id},${eval_envs},${reset_count},${task_sr},${eval_log_file},\"${task_desc}\"" >> "${SUMMARY_FILE}"
done <<< "${TASK_RECORDS}"

echo
echo "Per-task SR summary:"
cat "${SUMMARY_FILE}"
echo
echo "Done. Summary saved to: ${SUMMARY_FILE}"
