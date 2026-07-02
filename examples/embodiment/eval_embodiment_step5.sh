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

# continual learning testing
# BASE_CHECKPONIT_PATH="/mnt/project_rlinf/jzn/workspace/continual_learning/RLinf/logs/20260415-13:18:48-libero_spatial_offline_grpo_openpi_pi05/libero_spatial_offline_grpo_openpi_pi05/checkpoints"
BASE_DIR="${REPO_PATH}/logs/20260420-17:45:11-libero_spatial_offline_grpo_openpi_pi05_wo_vlm_step5"
BASE_CHECKPONIT_PATH="${BASE_DIR}/libero_spatial_offline_grpo_openpi_pi05/checkpoints"
BASE_LOG_DIR="${BASE_DIR}/eval_logs/"
STEPS=(1000 2000 3000 4000 5000 6000)
NUM_STEPS=5
NUM_ACTION_CHUNK=5

for STEP in ${STEPS[@]}; do

    # EVAL 500 STEP performance
    CONFIG_NAME="libero_spatial_grpo_openpi_pi05"
    LOG_DIR="${BASE_LOG_DIR}/step_${STEP}" #/$(date +'%Y%m%d-%H:%M:%S')"
    MEGA_LOG_FILE="${LOG_DIR}/eval_embodiment.log"
    CHECKPONIT_PATH="${BASE_CHECKPONIT_PATH}/global_step_${STEP}/actor/model_state_dict/full_weights.pt"
    mkdir -p "${LOG_DIR}"
    CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR} runner.ckpt_path=${CHECKPONIT_PATH} actor.model.num_steps=${NUM_STEPS} actor.model.num_action_chunks=${NUM_ACTION_CHUNK}"
    echo ${CMD}
    ${CMD} 2>&1 | tee ${MEGA_LOG_FILE}

    # EVAL 500 STEP performance
    # CONFIG_NAME="libero_goal_grpo_openpi_pi05"
    # LOG_DIR="${BASE_LOG_DIR}/libero_spatial_base_130_posttrain_spatial_step2/step_${STEP}_test_goal_cl" #/$(date +'%Y%m%d-%H:%M:%S')"
    # MEGA_LOG_FILE="${LOG_DIR}/eval_embodiment.log"
    # CHECKPONIT_PATH="${BASE_CHECKPONIT_PATH}/global_step_${STEP}/actor/model_state_dict/full_weights.pt"
    # mkdir -p "${LOG_DIR}"
    # CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR} runner.ckpt_path=${CHECKPONIT_PATH} actor.model.num_steps=${NUM_STEPS}"
    # echo ${CMD}
    # ${CMD} 2>&1 | tee ${MEGA_LOG_FILE}

    # EVAL 500 STEP performance
    # CONFIG_NAME="libero_object_grpo_openpi_pi05"
    # LOG_DIR="${BASE_LOG_DIR}/libero_spatial_base_130_step2/step_${STEP}_test_object_cl" #/$(date +'%Y%m%d-%H:%M:%S')"
    # MEGA_LOG_FILE="${LOG_DIR}/eval_embodiment.log"
    # CHECKPONIT_PATH="${BASE_CHECKPONIT_PATH}/global_step_${STEP}/actor/model_state_dict/full_weights.pt"
    # mkdir -p "${LOG_DIR}"
    # CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR} runner.ckpt_path=${CHECKPONIT_PATH} actor.model.num_steps=${NUM_STEPS}"
    # echo ${CMD}
    # ${CMD} 2>&1 | tee ${MEGA_LOG_FILE}

    # EVAL 500 STEP performance
    # CONFIG_NAME="libero_10_grpo_openpi_pi05"
    # LOG_DIR="${BASE_LOG_DIR}/libero_spatial_base_130_step2/step_${STEP}_test_long_cl" #/$(date +'%Y%m%d-%H:%M:%S')"
    # MEGA_LOG_FILE="${LOG_DIR}/eval_embodiment.log"
    # CHECKPONIT_PATH="${BASE_CHECKPONIT_PATH}/global_step_${STEP}/actor/model_state_dict/full_weights.pt"
    # mkdir -p "${LOG_DIR}"
    # CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR} runner.ckpt_path=${CHECKPONIT_PATH} actor.model.num_steps=${NUM_STEPS}"
    # echo ${CMD}
    # ${CMD} 2>&1 | tee ${MEGA_LOG_FILE}
done
