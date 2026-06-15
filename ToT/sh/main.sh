#!/usr/bin/env bash
set -euo pipefail

# Generic local runner for the Tree-of-Thought baseline.
# Configure values through environment variables before running this script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TASK="${TASK:-winograd}"
MODE="${MODE:-tot}"
TASK_START_INDEX="${TASK_START_INDEX:-0}"
TASK_END_INDEX="${TASK_END_INDEX:-200}"
BACKEND="${BACKEND:-local}"
TEMPERATURE="${TEMPERATURE:-0.7}"
STEP="${STEP:-3}"

METHOD_GENERATE="${METHOD_GENERATE:-propose}"
METHOD_EVALUATE="${METHOD_EVALUATE:-value}"
METHOD_SELECT="${METHOD_SELECT:-greedy}"
N_GENERATE_SAMPLE="${N_GENERATE_SAMPLE:-20}"
N_EVALUATE_SAMPLE="${N_EVALUATE_SAMPLE:-3}"
N_SELECT_SAMPLE="${N_SELECT_SAMPLE:-3}"

export TOT_MODEL_PATH="${TOT_MODEL_PATH:-HuggingFaceTB/SmolLM2-1.7B-Instruct}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p "${PROJECT_ROOT}/logs/${TASK}"
cd "${PROJECT_ROOT}"

case "${MODE}" in
    tot|bfs)
        "${PYTHON_BIN}" -u run.py \
            --task "${TASK}" \
            --task_start_index "${TASK_START_INDEX}" \
            --task_end_index "${TASK_END_INDEX}" \
            --backend "${BACKEND}" \
            --temperature "${TEMPERATURE}" \
            --method_generate "${METHOD_GENERATE}" \
            --method_evaluate "${METHOD_EVALUATE}" \
            --method_select "${METHOD_SELECT}" \
            --n_generate_sample "${N_GENERATE_SAMPLE}" \
            --n_evaluate_sample "${N_EVALUATE_SAMPLE}" \
            --n_select_sample "${N_SELECT_SAMPLE}" \
            --step "${STEP}" \
            "$@"
        ;;
    cot|standard)
        "${PYTHON_BIN}" -u run.py \
            --task "${TASK}" \
            --task_start_index "${TASK_START_INDEX}" \
            --task_end_index "${TASK_END_INDEX}" \
            --backend "${BACKEND}" \
            --temperature "${TEMPERATURE}" \
            --naive_run \
            --prompt_sample "${MODE}" \
            --n_generate_sample "${N_GENERATE_SAMPLE}" \
            --step "${STEP}" \
            "$@"
        ;;
    *)
        echo "Unsupported MODE='${MODE}'. Use tot, bfs, cot, or standard." >&2
        exit 1
        ;;
esac
