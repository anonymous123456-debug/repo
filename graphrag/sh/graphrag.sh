#!/usr/bin/env bash
set -euo pipefail

# Generic local runner for GraphRAG 0.5.0.
# Usage:
#   bash sh/graphrag.sh build
#   bash sh/graphrag.sh query commonsenseqa
#   bash sh/graphrag.sh all commonsenseqa

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${1:-query}"
TASK="${2:-commonsenseqa}"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.env"
  set +a
fi

if ! command -v graphrag >/dev/null 2>&1; then
  echo "GraphRAG is not installed. Install it with:"
  echo "  ${PYTHON_BIN} -m pip install graphrag==0.5.0"
  exit 1
fi

build_index() {
  echo "[GraphRAG] Building index from ${PROJECT_ROOT}/input"
  (cd "${PROJECT_ROOT}" && graphrag index --root "${PROJECT_ROOT}")
}

run_query_task() {
  local task_file="${PROJECT_ROOT}/code/${TASK}.py"
  if [[ ! -f "${task_file}" ]]; then
    echo "Task script not found: ${task_file}"
    echo "Available tasks:"
    find "${PROJECT_ROOT}/code" -maxdepth 1 -name "*.py" ! -name "metric.py" -printf "  %f\n" | sed 's/\.py$//'
    exit 1
  fi

  export GRAPHRAG_ROOT="${PROJECT_ROOT}"
  echo "[GraphRAG] Running task: ${TASK}"
  (cd "${PROJECT_ROOT}/code" && "${PYTHON_BIN}" -u "${task_file}")
}

case "${MODE}" in
  build)
    build_index
    ;;
  query)
    run_query_task
    ;;
  all)
    build_index
    run_query_task
    ;;
  *)
    echo "Unknown mode: ${MODE}"
    echo "Expected one of: build, query, all"
    exit 1
    ;;
esac
