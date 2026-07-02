#! /bin/bash

set -euo pipefail
SCRIPT_VERSION="2026-05-12-model-path-single-eval-per-subset"

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH="$(dirname "$(dirname "$EMBODIED_PATH")")"
export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"
export CALC_SR_SCRIPT="${EMBODIED_PATH}/calc_sr_from_flags.py"

export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

SUBSET_NAME="${1:-subset1}"
MODEL_PATH="${2:-/mnt/project_rlinf/jzn/workspace/openpi/checkpoints/torch/pi05_base}"
CONFIG_NAME="${3:-libero_object_grpo_openpi_pi05_noise}"
WORLD_SIZE="${4:-8}"
BASE_LOG_DIR="${5:-${REPO_PATH}/logs/eval_libero_object_subset_model_path_base_pi05}"
DENOISE_STEP=${6:-2}

# Canonical user-facing task order for libero_object basket tasks.
ORDERED_TASK_DESCS=(
    "pick up the alphabet soup and place it in the basket"
    "pick up the cream cheese and place it in the basket"
    "pick up the salad dressing and place it in the basket"
    "pick up the bbq sauce and place it in the basket"
    "pick up the ketchup and place it in the basket"
    "pick up the tomato sauce and place it in the basket"
    "pick up the butter and place it in the basket"
    "pick up the milk and place it in the basket"
    "pick up the chocolate pudding and place it in the basket"
    "pick up the orange juice and place it in the basket"
)

if [[ -z "${SUBSET_NAME}" || -z "${MODEL_PATH}" ]]; then
    echo "Usage: $0 <subset_name> <model_path> [config_name] [world_size] [base_log_dir] [denoise_step]"
    echo "Example:"
    echo "  $0 subset1 /path/to/openpi/model libero_object_offline_grpo_openpi_pi05_noise_subset 8"
    exit 1
fi

if [[ ! -e "${MODEL_PATH}" ]]; then
    echo "Model path not found: ${MODEL_PATH}"
    exit 1
fi

case "${SUBSET_NAME}" in
    subset1|libero_object_fullshot_subset1) MAX_TASK_ID=5 ;;
    subset2|libero_object_fullshot_subset2) MAX_TASK_ID=6 ;;
    subset3|libero_object_fullshot_subset3) MAX_TASK_ID=7 ;;
    subset4|libero_object_fullshot_subset4) MAX_TASK_ID=8 ;;
    subset5|libero_object_fullshot_subset5) MAX_TASK_ID=9 ;;
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
echo "  model_path  : ${MODEL_PATH}"
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
    suite = get_benchmark_overridden("libero_object")()

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
        raise KeyError(f"Task description not found in libero_object benchmark: {task_desc}")
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
echo "User task to benchmark task mapping (from libero_object benchmark):"
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

# Step 2: Evaluate the whole subset once, then split SR by each task's reset_state_ids.
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
    runner.ckpt_path=null \
    actor.model.model_path="${MODEL_PATH}" \
    rollout.model.model_path="${MODEL_PATH}" \
    actor.model.num_steps="${DENOISE_STEP}" \
    env.eval.total_num_envs="${eval_envs}" \
    env.eval.use_ordered_reset_state_ids=True \
    env.eval.eval_reset_state_ids="${ALL_RESET_IDS_JSON}" \
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
    echo "  desc         : ${task_desc}"
    echo "  reset_ids    : ${reset_count}"

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
