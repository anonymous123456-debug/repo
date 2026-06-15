# GraphRAG Baseline Workflow

This document describes the configuration of the `graphrag` paradigm, the GraphRAG knowledge graph construction workflow, and the inference workflow for each dataset.

## 1. Folder Structure

- `settings.yaml`: The main configuration file for GraphRAG 0.5.0. It contains settings for the LLM, embeddings, input corpus, cache, logs, output, entity extraction, community reports, and query prompts.
- `.env`: An example file for local environment variables.
- `input/`: The local document corpus read during GraphRAG construction. The current configuration reads `input/*.txt`.
- `prompts/`: Prompts used for GraphRAG construction and querying, including entity extraction, description summarization, community reports, local/global/drift search, and related prompts.
- `code/`: GraphRAG query and evaluation scripts for each dataset.
- `sh/graphrag.sh`: A general local execution script.
- `cache/`: The cache directory used during GraphRAG construction.
- `logs/`: The log directory used during GraphRAG construction.
- `output/`: The directory for GraphRAG indexes and intermediate artifacts generated after construction.

## 2. Environment Setup

It is recommended to use an isolated Python environment:

```bash
cd /path/to/graphrag
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install graphrag==0.5.0 datasets tqdm
```

Configure `.env`:

```bash
GRAPHRAG_API_KEY=YOUR_API_KEY
```

If you use the public OpenAI API, you usually do not need to set `api_base`. If you use a local or private compatible endpoint, run the model locally and create a server for request forwarding.

## 4. GraphRAG Build Flow

The local GraphRAG construction workflow is as follows:

1. Prepare the environment and install `graphrag==0.5.0`.
2. Put the document corpus used for graph construction into `input/`. The file format should be `.txt`.
3. Check `settings.yaml`:
   - `input.base_dir: "input"` means the corpus is read from `input/`.
   - `file_pattern: ".*\\.txt$"` means only text files are read.
   - `chunks.size: 1200` and `chunks.overlap: 100` control the chunk size and overlap length.
   - `entity_extraction.prompt` points to `prompts/entity_extraction.txt`.
   - `community_reports.prompt` points to `prompts/community_report.txt`.
   - `local_search.prompt` points to `prompts/local_search_system_prompt.txt`.
4. Run local index construction:

```bash
bash sh/graphrag.sh build
```

This command actually runs:

```bash
graphrag index --root /path/to/graphrag
```

The following files and directories are generated during construction:

- `cache/`: Cache for LLM calls and the construction process.
- `logs/`: GraphRAG construction logs.
- `output/`: Intermediate results such as graph indexes, entities, relationships, text chunks, communities, and community reports.
- `output/lancedb/`: The LanceDB vector database used for vector retrieval during GraphRAG querying.

## 5. Inference Flow

Inference is performed by the scripts in `code/*.py`. Each script:

1. Reads the corresponding dataset.
2. Formats the question, options, or context into a query prompt.
3. Calls:

```bash
graphrag query --root "$GRAPHRAG_ROOT" --method local --query "..."
```

4. Parses the answer from the GraphRAG output.
5. Calculates accuracy or F1 score.

Available task scripts include:

- `bqa.py`
- `commonsenseqa.py`
- `cosmosqa.py`
- `hotpotqa.py`
- `mcqa.py`
- `medqa.py`
- `sciq.py`
- `squad.py`
- `wiki.py`
- `winograd.py`

General execution:

```bash
bash sh/graphrag.sh query commonsenseqa
```

To build the index first and then run a task:

```bash
bash sh/graphrag.sh all commonsenseqa
```

Here, `commonsenseqa` can be replaced with any task name listed above, without the `.py` suffix.

## 6. Prompt Locations

GraphRAG construction and query prompts are located in `prompts/`:

- `entity_extraction.txt`: Prompt for entity and relationship extraction.
- `summarize_descriptions.txt`: Prompt for summarizing entity/relationship descriptions.
- `community_report.txt`: Prompt for generating community reports.
- `local_search_system_prompt.txt`: Prompt for local search queries.
- `global_search_map_system_prompt.txt`: Prompt for the global search map stage.
- `global_search_reduce_system_prompt.txt`: Prompt for the global search reduce stage.
- `global_search_knowledge_system_prompt.txt`: Prompt for global search knowledge organization.
- `drift_search_system_prompt.txt`: Prompt for drift search queries.
- `question_gen_system_prompt.txt`: Prompt for question generation.
- `claim_extraction.txt`: Prompt for claim extraction. In the current `settings.yaml`, `claim_extraction.enabled: false`, so it is disabled by default.

## 7. Notes

- The current query scripts use `--method local` by default, which means retrieval-augmented answering is performed based on the locally constructed graph and vector database.
- If you modify the `input/` corpus, prompts, or core configuration, it is recommended to rerun `bash sh/graphrag.sh build`.
