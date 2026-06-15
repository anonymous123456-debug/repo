#!/usr/bin/env bash
set -euo pipefail

# Generic local runner for LongRAG and vanilla RAG inference.
# Override these values from the shell when needed.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET="${DATASET:-winograd}"
MODEL="${MODEL:-misral}"
LRAG_MODEL="${LRAG_MODEL:-}"
SAMPLE_SIZE="${SAMPLE_SIZE:-200}"
TOP_K1="${TOP_K1:-100}"
TOP_K2="${TOP_K2:-20}"
CHUNK_SIZE="${CHUNK_SIZE:-200}"
MIN_SENTENCE="${MIN_SENTENCE:-2}"
OVERLAP="${OVERLAP:-2}"
MODE="${MODE:-all}"
BUILD_INDEX="${BUILD_INDEX:-auto}"

R_PATH="${R_PATH:-../data/corpus/processed/${CHUNK_SIZE}_${MIN_SENTENCE}_${OVERLAP}}"
INDEX_DIR="${PROJECT_ROOT}/data/corpus/processed/${CHUNK_SIZE}_${MIN_SENTENCE}_${OVERLAP}/${DATASET}"
RAW_JSON="${PROJECT_ROOT}/data/corpus/raw/${DATASET}.json"

if [[ "${BUILD_INDEX}" == "1" || "${BUILD_INDEX}" == "true" || ( "${BUILD_INDEX}" == "auto" && ! -f "${INDEX_DIR}/vector.index" ) ]]; then
  if [[ ! -f "${RAW_JSON}" ]]; then
    echo "Raw corpus file not found: ${RAW_JSON}" >&2
    exit 1
  fi
  echo "Building FAISS index for ${DATASET}..."
  (
    cd "${PROJECT_ROOT}/src"
    "${PYTHON_BIN}" gen_index.py \
      --dataset "${DATASET}" \
      --chunk_size "${CHUNK_SIZE}" \
      --min_sentence "${MIN_SENTENCE}" \
      --overlap "${OVERLAP}"
  )
fi

COMMON_ARGS=(
  --dataset "${DATASET}"
  --model "${MODEL}"
  --top_k1 "${TOP_K1}"
  --top_k2 "${TOP_K2}"
  --r_path "${R_PATH}"
  --sample_size "${SAMPLE_SIZE}"
)

if [[ -n "${LRAG_MODEL}" ]]; then
  COMMON_ARGS+=(--lrag_model "${LRAG_MODEL}")
fi

run_mode() {
  local method="$1"
  echo "Running ${method} on ${DATASET} with ${MODEL}..."
  (
    cd "${PROJECT_ROOT}/src"
    "${PYTHON_BIN}" main.py "${COMMON_ARGS[@]}" "--${method}"
  )
}

case "${MODE}" in
  all)
    run_mode ext_fil
    run_mode rb
    ;;
  ext_fil|rb)
    run_mode "${MODE}"
    ;;
  *)
    echo "Unsupported MODE='${MODE}'. Use all, ext_fil, or rb." >&2
    exit 1
    ;;
esac
