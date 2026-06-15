# ToG Neo4j Baseline

This document provides an example run script and explanation for the ToG baseline.

## 1. Folder Structure

- `ToG/main_wiki.py`: Main entry point for running ToG on a selected dataset.
- `ToG/wiki_func.py`: Neo4j-based graph search functions, including relation retrieval, entity expansion, pruning, and reasoning.
- `ToG/utils.py`: Dataset loading, LLM calls, BM25/SentenceTransformer helper functions, output writing, and common utility functions.
- `ToG/prompt_list.py`: Prompt templates used for relation selection, entity scoring, reasoning, and answer generation.
- `ToG/metric.py`: F1 scoring utilities for QA-style datasets.
- `ToG/server_urls.txt`: Placeholder endpoint list for the original Wikidata service mode. The current workflow uses local Neo4j.
- `data/`: Dataset files used by the scripts.
- `eval/`: Evaluation tools for JSON output files.
- `tools/`: Small helper scripts for data conversion and preprocessing.
- `sh/main.sh`: General local script for running `ToG/main_wiki.py`.

## 2. Environment Setup

It is recommended to create an isolated Python environment and install the dependencies:

```bash
cd /path/to/ToG-main
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install neo4j openai tqdm rank-bm25 sentence-transformers transformers datasets torch
```

Start a local Neo4j service and load the graph data required by the current dataset. The current code expects nodes to contain an integer `id` field and a `name` field. Dataset-specific node labels follow this format:

```text
Entity_<dataset>
```

For example, `commonsenseqa` is mapped to `Entity_commonsense`.

## 3. Running With `sh/main.sh`

This run script is designed for normal local execution. Cluster scheduling directives, node exclusion settings, and private Python environment commands have been removed.

Basic usage:

```bash
cd /path/to/ToG-main
bash sh/main.sh
```

Common parameter overrides:

```bash
DATASET=hotpotqa \
LLM_TYPE=phi \
TOG_MODEL_PATH="YOUR_LOCAL_OR_HF_MODEL_PATH" \
NEO4J_URI="bolt://localhost:7687" \
NEO4J_USER="neo4j" \
NEO4J_PASSWORD="YOUR_NEO4J_PASSWORD" \
bash sh/main.sh
```

The script performs the following steps:

1. Resolves the project root based on the location of `sh/main.sh`.
2. Reads runtime parameters from environment variables and provides local defaults for unset parameters.
3. Exports Neo4j connection settings for the Python code.
4. Creates `ToG/misral/` if it does not already exist.
5. Changes into the `ToG/` directory.
6. Runs `main_wiki.py` with the selected dataset, search depth, search width, pruning tool, LLM type, and other parameters.

## 4. Main Runtime Parameters

- `DATASET`: Dataset name. Default: `2multiwiki`.
- `MAX_LENGTH`: Maximum LLM output length. Default: `256`.
- `TEMPERATURE_EXPLORATION`: LLM temperature during the relation/entity exploration stage. Default: `0.4`.
- `TEMPERATURE_REASONING`: LLM temperature during the final reasoning stage. Default: `0`.
- `WIDTH`: ToG search width. Default: `3`.
- `DEPTH`: ToG search depth. Default: `3`.
- `REMOVE_UNNECESSARY_REL`: Whether to filter unhelpful relations. Default: `True`.
- `LLM_TYPE`: LLM type passed to the code. Default: `phi`.
- `OPENAI_API_KEY`: Optional key used when `LLM_TYPE` is an OpenAI-compatible model.
- `NUM_RETAIN_ENTITY`: Number of entities retained during entity search. Default: `5`.
- `PRUNE_TOOLS`: Relation pruning backend. Default: `llm`.
- `PYTHON_BIN`: Python executable. Default: `python3`.

## 5. ToG Execution Flow

1. `main_wiki.py` calls `prepare_dataset()` from `utils.py` to load the selected dataset from `data/`.
2. For each sample, the script reads the question and `qid_topic_entity`.
3. If no topic entity exists, `generate_without_explored_paths()` directly calls the LLM to answer.
4. Otherwise, ToG expands graph paths layer by layer according to the search depth.
5. `relation_search_prune()` in `wiki_func.py` queries Neo4j to get outgoing and incoming relations of the current entity.
6. The LLM scores candidate relations through the prompt templates in `prompt_list.py`.
7. `entity_search()` retrieves candidate neighboring entities from Neo4j.
8. `entity_score()` uses the LLM to score candidate entities.
9. `entity_prune()` keeps the best candidates and builds the reasoning chain.
10. `reasoning()` asks the LLM whether the current graph reasoning chain is sufficient to answer the question.
11. Results are appended to `ToG/misral/ToG_<dataset>.jsonl`.
12. After inference, `main_wiki.py` computes accuracy or F1 depending on the dataset type.

## 6. Output Files

The main output file is:

```text
ToG/misral/ToG_<dataset>.jsonl
```

Each JSONL line contains:

- `question`: The input question.
- `results`: The model output.
- `reasoning_chains`: The graph reasoning chain selected by ToG.

## 7. Notes

- The current code uses Neo4j as the graph backend instead of the original Wikidata service client.
- If using a local Hugging Face model, make sure `TOG_MODEL_PATH` points to a valid local model directory or public model ID.
- If using an OpenAI-compatible model, set `OPENAI_API_KEY` and confirm that the model path behavior in `utils.py` matches your environment.
