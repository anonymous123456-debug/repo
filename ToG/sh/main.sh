#!/usr/bin/env bash
set -euo pipefail

# Generic local runner for the ToG Neo4j baseline.
# Environment variables can override every default below.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

DATASET="${DATASET:-2multiwiki}"
MAX_LENGTH="${MAX_LENGTH:-256}"
TEMPERATURE_EXPLORATION="${TEMPERATURE_EXPLORATION:-0.4}"
TEMPERATURE_REASONING="${TEMPERATURE_REASONING:-0}"
WIDTH="${WIDTH:-3}"
DEPTH="${DEPTH:-3}"
REMOVE_UNNECESSARY_REL="${REMOVE_UNNECESSARY_REL:-True}"
LLM_TYPE="${LLM_TYPE:-phi}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
NUM_RETAIN_ENTITY="${NUM_RETAIN_ENTITY:-5}"
PRUNE_TOOLS="${PRUNE_TOOLS:-llm}"

export NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
export NEO4J_USER="${NEO4J_USER:-neo4j}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-}"

mkdir -p "${PROJECT_ROOT}/ToG/misral"

echo "Running ToG locally"
echo "Project root resolved"
echo "Dataset: ${DATASET}"
echo "Neo4j URI: ${NEO4J_URI}"

cd "${PROJECT_ROOT}/ToG"

"${PYTHON_BIN}" -u main_wiki.py \
  --dataset "${DATASET}" \
  --max_length "${MAX_LENGTH}" \
  --temperature_exploration "${TEMPERATURE_EXPLORATION}" \
  --temperature_reasoning "${TEMPERATURE_REASONING}" \
  --width "${WIDTH}" \
  --depth "${DEPTH}" \
  --remove_unnecessary_rel "${REMOVE_UNNECESSARY_REL}" \
  --LLM_type "${LLM_TYPE}" \
  --opeani_api_keys "${OPENAI_API_KEY}" \
  --num_retain_entity "${NUM_RETAIN_ENTITY}" \
  --prune_tools "${PRUNE_TOOLS}"
