#!/usr/bin/env bash
set -euo pipefail

SIMPLERENV_PATH="${SIMPLERENV_PATH:-/mnt/project_rlinf/jzn/workspace/third_party/SimplerEnv}"
SIMPLERENV_REPO="${SIMPLERENV_REPO:-https://github.com/simpler-env/SimplerEnv.git}"
SIMPLERENV_REF="${SIMPLERENV_REF:-06accaca93535902d408da4855f21cece12bceb7}"
SIMPLER_ENV_PREFIX="${SIMPLER_ENV_PREFIX:-/mnt/project_rlinf/jzn/conda_envs/simpler_env}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
ENV_MANAGER="${ENV_MANAGER:-auto}"
BASE_PYTHON="${BASE_PYTHON:-$(command -v python3 || true)}"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"

RLINF_ROOT="${RLINF_ROOT:-/mnt/project_rlinf/jzn/workspace/continual_learning/RLinf}"
LOG_ROOT="${LOG_ROOT:-${RLINF_ROOT}/logs/simpler_env_setup}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/${TIMESTAMP}}"
mkdir -p "${LOG_DIR}"

find_conda() {
  if [[ -n "${CONDA_BIN:-}" && -x "${CONDA_BIN}" ]]; then
    echo "${CONDA_BIN}"
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return
  fi
  for candidate in /opt/conda/bin/conda /root/miniconda3/bin/conda /root/anaconda3/bin/conda; do
    if [[ -x "${candidate}" ]]; then
      echo "${candidate}"
      return
    fi
  done
  return 1
}

CONDA_BIN="$(find_conda || true)"
if [[ "${ENV_MANAGER}" == "auto" ]]; then
  if [[ -n "${CONDA_BIN}" ]]; then
    ENV_MANAGER="conda"
  else
    ENV_MANAGER="venv"
  fi
fi
if [[ "${ENV_MANAGER}" == "conda" && -z "${CONDA_BIN}" ]]; then
  echo "Could not find conda. Set CONDA_BIN or use ENV_MANAGER=venv. This script does not modify /opt/venv/openpi." >&2
  exit 1
fi
if [[ "${ENV_MANAGER}" == "venv" && -z "${BASE_PYTHON}" ]]; then
  echo "Could not find python3 for venv fallback. Set BASE_PYTHON or install conda." >&2
  exit 1
fi
if [[ "${ENV_MANAGER}" != "conda" && "${ENV_MANAGER}" != "venv" ]]; then
  echo "Unsupported ENV_MANAGER=${ENV_MANAGER}; expected auto, conda, or venv." >&2
  exit 1
fi

PY="${SIMPLER_ENV_PREFIX}/bin/python"
INSTALL_STEPS="${LOG_DIR}/install_steps.txt"
PIP_CONSTRAINT_FILE="${LOG_DIR}/pip_constraints.txt"

{
  echo "SIMPLERENV_PATH=${SIMPLERENV_PATH}"
  echo "SIMPLERENV_REPO=${SIMPLERENV_REPO}"
  echo "SIMPLERENV_REF=${SIMPLERENV_REF}"
  echo "SIMPLER_ENV_PREFIX=${SIMPLER_ENV_PREFIX}"
  echo "PYTHON_VERSION=${PYTHON_VERSION}"
  echo "ENV_MANAGER=${ENV_MANAGER}"
  echo "CONDA_BIN=${CONDA_BIN}"
  echo "BASE_PYTHON=${BASE_PYTHON}"
  echo "UV_BIN=${UV_BIN}"
  echo
} > "${INSTALL_STEPS}"

cat > "${PIP_CONSTRAINT_FILE}" <<'EOF'
numpy==1.24.4
opencv-python==4.9.0.80
scikit-build-core<0.10
setuptools<70
EOF
export PIP_CONSTRAINT="${PIP_CONSTRAINT:-${PIP_CONSTRAINT_FILE}}"
echo "PIP_CONSTRAINT=${PIP_CONSTRAINT}" >> "${INSTALL_STEPS}"
echo >> "${INSTALL_STEPS}"

log_step() {
  echo "+ $*" | tee -a "${INSTALL_STEPS}"
}

if [[ ! -d "${SIMPLERENV_PATH}/.git" ]]; then
  mkdir -p "$(dirname "${SIMPLERENV_PATH}")"
  log_step git clone --recurse-submodules "${SIMPLERENV_REPO}" "${SIMPLERENV_PATH}"
  git clone --recurse-submodules "${SIMPLERENV_REPO}" "${SIMPLERENV_PATH}"
else
  log_step git -C "${SIMPLERENV_PATH}" fetch origin
  git -C "${SIMPLERENV_PATH}" fetch origin
fi

log_step git -C "${SIMPLERENV_PATH}" checkout "${SIMPLERENV_REF}"
git -C "${SIMPLERENV_PATH}" checkout "${SIMPLERENV_REF}"

log_step git -C "${SIMPLERENV_PATH}" submodule update --init --recursive
git -C "${SIMPLERENV_PATH}" submodule update --init --recursive

if [[ -x "${PY}" ]] && ! "${PY}" -m pip --version >/dev/null 2>&1; then
  case "${SIMPLER_ENV_PREFIX}" in
    /mnt/project_rlinf/jzn/conda_envs/*)
      log_step rm -rf "${SIMPLER_ENV_PREFIX}"
      rm -rf "${SIMPLER_ENV_PREFIX}"
      ;;
    *)
      echo "Existing env lacks pip, but refusing to remove unexpected path: ${SIMPLER_ENV_PREFIX}" >&2
      exit 1
      ;;
  esac
fi

if [[ ! -x "${PY}" ]]; then
  if [[ "${ENV_MANAGER}" == "conda" ]]; then
    log_step "${CONDA_BIN}" create -y -p "${SIMPLER_ENV_PREFIX}" python="${PYTHON_VERSION}"
    "${CONDA_BIN}" create -y -p "${SIMPLER_ENV_PREFIX}" python="${PYTHON_VERSION}"
  elif [[ -n "${UV_BIN}" ]]; then
    mkdir -p "$(dirname "${SIMPLER_ENV_PREFIX}")"
    log_step "${UV_BIN}" venv --seed --python "${BASE_PYTHON}" "${SIMPLER_ENV_PREFIX}"
    "${UV_BIN}" venv --seed --python "${BASE_PYTHON}" "${SIMPLER_ENV_PREFIX}"
  else
    mkdir -p "$(dirname "${SIMPLER_ENV_PREFIX}")"
    log_step "${BASE_PYTHON}" -m venv "${SIMPLER_ENV_PREFIX}"
    "${BASE_PYTHON}" -m venv "${SIMPLER_ENV_PREFIX}"
  fi
fi

log_step "${PY}" -m pip install --upgrade pip
"${PY}" -m pip install --upgrade pip

log_step "${PY}" -m pip install "setuptools<70"
"${PY}" -m pip install "setuptools<70"

log_step "${PY}" -m pip install numpy==1.24.4
"${PY}" -m pip install numpy==1.24.4

log_step "${PY}" -m pip install tyro matplotlib mediapy websockets msgpack
"${PY}" -m pip install tyro matplotlib mediapy websockets msgpack

log_step "${PY}" -m pip install -e "${SIMPLERENV_PATH}/ManiSkill2_real2sim"
"${PY}" -m pip install -e "${SIMPLERENV_PATH}/ManiSkill2_real2sim"

log_step "${PY}" -m pip install -e "${SIMPLERENV_PATH}"
"${PY}" -m pip install -e "${SIMPLERENV_PATH}"

{
  echo "SimplerEnv HEAD:"
  git -C "${SIMPLERENV_PATH}" rev-parse HEAD
  echo
  echo "SimplerEnv status:"
  git -C "${SIMPLERENV_PATH}" status --short
  echo
  echo "Submodule status:"
  git -C "${SIMPLERENV_PATH}" submodule status --recursive
} > "${LOG_DIR}/source_manifest.txt"

"${PY}" -m pip freeze > "${LOG_DIR}/pip_freeze.txt"

PYTHONPATH="${SIMPLERENV_PATH}:${PYTHONPATH:-}" "${PY}" - <<'PY'
import simpler_env
from simpler_env import ENVIRONMENTS

required = {
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "widowx_put_eggplant_in_basket",
}
missing = sorted(required - set(ENVIRONMENTS))
if missing:
    raise SystemExit(f"Missing required SimplerEnv tasks: {missing}")
print("simpler_env import ok")
print("widowx visual matching tasks ok")
PY

echo "SimplerEnv environment is ready."
echo "Install log: ${LOG_DIR}"
echo "Python: ${PY}"
