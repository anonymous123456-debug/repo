#!/usr/bin/env bash
set -euo pipefail

# Generic runner for Raw / CoT / CoT-SC experiments.
# Usage:
#   bash sh/run_experiment.sh <dataset> <method> [model_path]
#
# Examples:
#   bash sh/run_experiment.sh hotpotqa raw Qwen/Qwen2.5-1.5B-Instruct
#   MODEL_PATH=/path/to/model bash sh/run_experiment.sh commonsense cotsc
#
# Supported datasets:
#   2wiki, commonsense, cosmos, hotpotqa, logicbench, medqa, sciq, squad, winograd
#
# Supported methods:
#   raw, cot, cotsc

DATASET="${1:-hotpotqa}"
METHOD="${2:-raw}"
MODEL_PATH_ARG="${3:-${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}}"
PYTHON_BIN="${PYTHON:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

case "${DATASET}" in
  2wiki|commonsense|cosmos|hotpotqa|logicbench|medqa|sciq|squad|winograd)
    ;;
  *)
    echo "Unsupported dataset: ${DATASET}" >&2
    exit 1
    ;;
esac

case "${METHOD}" in
  raw|cot|cotsc)
    ;;
  *)
    echo "Unsupported method: ${METHOD}" >&2
    exit 1
    ;;
esac

cd "${PROJECT_DIR}"

if [[ "${DATASET}" == "commonsense" || "${DATASET}" == "cosmos" ]]; then
  "${PYTHON_BIN}" "${DATASET}.py" --method "${METHOD}" --dataset "${DATASET}" --model_path "${MODEL_PATH_ARG}"
else
  "${PYTHON_BIN}" "${DATASET}.py" --method "${METHOD}" --model_path "${MODEL_PATH_ARG}"
fi
