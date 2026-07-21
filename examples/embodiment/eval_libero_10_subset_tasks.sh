#! /bin/bash

set -euo pipefail

EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export LIBERO_BENCHMARK_NAME="libero_10"

exec bash "${EMBODIED_PATH}/eval_libero_subset_tasks_once.sh" "$@"
