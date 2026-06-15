#!/usr/bin/env bash
#SBATCH --job-name=mindmap
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=A40

set -euo pipefail

# 用法示例：
#   sbatch sh/run_mindmap.sh csv hotpotqa
#   sbatch sh/run_mindmap.sh keywords hotpotqa
#   sbatch sh/run_mindmap.sh embed hotpotqa
#   sbatch sh/run_mindmap.sh mindmap hotpotqa qwen
#   sbatch sh/run_mindmap.sh all hotpotqa qwen
#   sbatch sh/run_mindmap.sh mindmap hotpotqa phi 200
#
# 脱敏说明：原脚本中的个人路径、模型路径和数据库凭据已用“”替代。
# 请在提交任务前通过环境变量注入真实值，不要把真实隐私信息写回脚本：
#   export MINDMAP_PROJECT_DIR="“”"          # 可选；默认自动定位到本脚本上级目录
#   export MINDMAP_PYTHON="“”"               # 可选；默认使用当前环境中的 python
#   export MINDMAP_CONDA_ENV="“”"            # 可选；需要 conda 环境时填写
#   export MINDMAP_LLM_MODEL_PATH="“”"       # 必填；实体/关系/关键词抽取模型
#   export MINDMAP_EMBEDDING_MODEL_PATH="“”" # 必填；SentenceTransformer embedding 模型
#   export MINDMAP_NEO4J_URI="“”"            # mindmap 阶段必填
#   export MINDMAP_NEO4J_USER="“”"           # mindmap 阶段必填
#   export MINDMAP_NEO4J_PASSWORD="“”"       # mindmap 阶段必填

STAGE="${1:-mindmap}"
DATASET="${2:-hotpotqa}"
MODEL_VARIANT="${3:-qwen}"
EVAL_LIMIT="${4:-${MINDMAP_EVAL_LIMIT:-200}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${MINDMAP_PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${MINDMAP_PYTHON:-python}"
CONDA_ENV="${MINDMAP_CONDA_ENV:-}"
LOG_DIR="${PROJECT_DIR}/logs"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_file() {
  local file="$1"
  [[ -f "$file" ]] || die "required file not found: $file"
}

load_conda_env() {
  [[ -n "$CONDA_ENV" ]] || return 0
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  else
    die "MINDMAP_CONDA_ENV is set, but conda is not available"
  fi
}

run_dataset_script() {
  local script_name="$1"
  local dataset_dir="${PROJECT_DIR}/dataset/${DATASET}"
  require_file "${dataset_dir}/${script_name}"
  cd "$dataset_dir"
  "$PYTHON_BIN" -u "$script_name"
}

run_mindmap() {
  case "$MODEL_VARIANT" in
    qwen|phi|smol|default|base|llama) ;;
    *) die "unknown model variant: ${MODEL_VARIANT}; use qwen, phi, smol, or default" ;;
  esac
  local normalized_model="$MODEL_VARIANT"
  case "$normalized_model" in
    base|llama) normalized_model="default" ;;
  esac

  require_file "${PROJECT_DIR}/mindmap.py"
  cd "$PROJECT_DIR"
  "$PYTHON_BIN" -u mindmap.py --dataset "$DATASET" --model "$normalized_model" --limit "$EVAL_LIMIT"
}

run_preprocess_all() {
  local dataset_dir="${PROJECT_DIR}/dataset/${DATASET}"
  [[ -d "$dataset_dir" ]] || die "unknown dataset: ${DATASET}"

  if [[ -f "${dataset_dir}/gen.py" ]]; then
    run_dataset_script "gen.py"
    return 0
  fi

  run_dataset_script "gen_csv.py"
  run_dataset_script "gen_keywords.py"
  run_dataset_script "encode_keyword_entity.py"
}

mkdir -p "$LOG_DIR"
load_conda_env

echo "Stage: ${STAGE}"
echo "Dataset: ${DATASET}"
echo "Model variant: ${MODEL_VARIANT}"
echo "Eval limit: ${EVAL_LIMIT}"
echo "Project dir: ${PROJECT_DIR}"
echo "Python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
echo "Node: $(hostname)"
date

case "$STAGE" in
  text)
    run_dataset_script "gen_text.py"
    ;;
  csv)
    run_dataset_script "gen_csv.py"
    ;;
  keywords)
    run_dataset_script "gen_keywords.py"
    ;;
  embed)
    run_dataset_script "encode_keyword_entity.py"
    ;;
  preprocess|preprocess-all)
    run_preprocess_all
    ;;
  mindmap)
    run_mindmap
    ;;
  all)
    run_preprocess_all
    run_mindmap
    ;;
  *)
    die "unknown stage: ${STAGE}; use text, csv, keywords, embed, preprocess, mindmap, or all"
    ;;
esac

date
echo "Done."
