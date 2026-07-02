#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"

export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

# Base path to the BEHAVIOR dataset, which is the BEHAVIOR-1k repo's dataset folder
# Only required when running the behavior experiment.
export OMNIGIBSON_DATA_PATH=$OMNIGIBSON_DATA_PATH
export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATASET_PATH:-$OMNIGIBSON_DATA_PATH/behavior-1k-assets/}
export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_KEY_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson.key}
export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_ASSET_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets/}
export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}
# Base path to Isaac Sim, only required when running the behavior experiment.
export ISAAC_PATH=${ISAAC_PATH:-/path/to/isaac-sim}
export EXP_PATH=${EXP_PATH:-$ISAAC_PATH/apps}
export CARB_APP_PATH=${CARB_APP_PATH:-$ISAAC_PATH/kit}

export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:$PYTHONPATH

export DREAMZERO_PATH=${DREAMZERO_PATH:-"/path/to/DreamZero"}
export PYTHONPATH=${DREAMZERO_PATH}:$PYTHONPATH

export HYDRA_FULL_ERROR=1

if [ -z "$1" ]; then
    CONFIG_NAME="maniskill_ppo_openvlaoft"
else
    CONFIG_NAME=$1
fi

# NOTE: Set the active robot platform (required for correct action dimension and normalization), supported platforms are LIBERO, ALOHA, BRIDGE, default is LIBERO
ROBOT_PLATFORM=${2:-${ROBOT_PLATFORM:-"LIBERO"}}

export ROBOT_PLATFORM

# Libero variant: standard, pro, plus
export LIBERO_TYPE=${LIBERO_TYPE:-"standard"}
if [ "$LIBERO_TYPE" == "pro" ]; then
    export LIBERO_PERTURBATION="all"  # all,swap,object,lan
    echo "Evaluation Mode: LIBERO-PRO | Perturbation: $LIBERO_PERTURBATION"
elif [ "$LIBERO_TYPE" == "plus" ]; then
    export LIBERO_SUFFIX="all"
    echo "Evaluation Mode: LIBERO-PLUS | Suffix: $LIBERO_SUFFIX"
else
    echo "Evaluation Mode: Standard LIBERO"
fi

echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM"

# LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')" #/$(date +'%Y%m%d-%H:%M:%S')"
# MEGA_LOG_FILE="${LOG_DIR}/eval_embodiment.log"
# mkdir -p "${LOG_DIR}"
# CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"
# echo ${CMD}
# ${CMD} 2>&1 | tee ${MEGA_LOG_FILE}

BASE_LOG_DIR="${REPO_PATH}/logs/20260420-17:59:35-libero_spatial_offline_grpo_openpi_pi05_noise_wo_vlm_step5/" #/$(date +'%Y%m%d-%H:%M:%S')"

MODEL_PATH_DIR="/mnt/project_rlinf/jzn/workspace/openpi/checkpoints/sft/pi05_libero_spatial_fullshot/libero_spatial_fullshot/14000"
NUM_STEPS=5
NUM_ACTION_CHUNK=5

# base policy testing
CONFIG_NAME="libero_spatial_grpo_openpi_pi05_noise"
LOG_DIR="${BASE_LOG_DIR}/sft_eval_logs/step_14000_spatial"
MEGA_LOG_FILE="${LOG_DIR}/eval_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} \
runner.logger.log_path=${LOG_DIR} \
actor.model.model_path=${MODEL_PATH_DIR} \
rollout.model.model_path=${MODEL_PATH_DIR} \
actor.model.num_steps=${NUM_STEPS} \
actor.model.num_action_chunks=${NUM_ACTION_CHUNK}"
echo ${CMD}
${CMD} 2>&1 | tee ${MEGA_LOG_FILE}

CONFIG_NAME="libero_goal_grpo_openpi_pi05_noise"
LOG_DIR="${BASE_LOG_DIR}/sft_eval_logs/step_14000_goal"
MEGA_LOG_FILE="${LOG_DIR}/eval_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} \
runner.logger.log_path=${LOG_DIR} \
actor.model.model_path=${MODEL_PATH_DIR} \
rollout.model.model_path=${MODEL_PATH_DIR} \
actor.model.num_steps=${NUM_STEPS} \
actor.model.num_action_chunks=${NUM_ACTION_CHUNK}"
echo ${CMD}
${CMD} 2>&1 | tee ${MEGA_LOG_FILE}


CONFIG_NAME="libero_object_grpo_openpi_pi05_noise"
LOG_DIR="${BASE_LOG_DIR}/sft_eval_logs/step_14000_object"
MEGA_LOG_FILE="${LOG_DIR}/eval_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} \
runner.logger.log_path=${LOG_DIR} \
actor.model.model_path=${MODEL_PATH_DIR} \
rollout.model.model_path=${MODEL_PATH_DIR} \
actor.model.num_steps=${NUM_STEPS} \
actor.model.num_action_chunks=${NUM_ACTION_CHUNK}"
echo ${CMD}
${CMD} 2>&1 | tee ${MEGA_LOG_FILE}


CONFIG_NAME="libero_10_grpo_openpi_pi05_noise"
LOG_DIR="${BASE_LOG_DIR}/sft_eval_logs/step_14000_long"
MEGA_LOG_FILE="${LOG_DIR}/eval_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} \
runner.logger.log_path=${LOG_DIR} \
actor.model.model_path=${MODEL_PATH_DIR} \
rollout.model.model_path=${MODEL_PATH_DIR} \
actor.model.num_steps=${NUM_STEPS} \
actor.model.num_action_chunks=${NUM_ACTION_CHUNK}"
echo ${CMD}
${CMD} 2>&1 | tee ${MEGA_LOG_FILE}