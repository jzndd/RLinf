#! /bin/bash
set -euo pipefail

export CONFIG_NAME="${CONFIG_NAME:-libero_spatial_offline_grpo_fastwam_noise_lora}"

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
bash "${SCRIPT_DIR}/run_offline_grpo_fastwam_noise.sh" "$@"
