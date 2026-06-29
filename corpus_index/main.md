# RAG Corpus Generation and Vector Index Construction Guide

This file presents the scripts for RAG corpus generation, text chunking, and vector index construction for all ten datasets. The scripts are derived from the original processing workflow related to RAG corpus generation and vector encoding in `preprocess/raw`, and they have been unified into the same command-line format. The original core logic is preserved: first generate `data/corpus/raw/<dataset>.json`, then split sentences by Chinese and English punctuation, merge chunks with `chunk_size`, `min_sentence`, and `overlap`, and finally encode the chunks with SentenceTransformer and write them into a FAISS `IndexFlatIP` index.

## File Description

| File | Purpose |
| --- | --- |
| `corpus_index_utils.py` | Shared utility for reading JSON/JSONL/CSV/TXT files, generating the raw corpus, chunking text, saving mappings, and encoding vectors. |
| `build_2wikimultihopqa.py` | Corpus and index construction for 2WikiMultiHopQA. |
| `build_hotpotqa.py` | Corpus and index construction for HotpotQA. |
| `build_squad.py` | Corpus and index construction for SQuAD. |
| `build_medqa.py` | Corpus and index construction for MedQA. |
| `build_sciq.py` | Corpus and index construction for SciQ. |
| `build_commonsenseqa.py` | Corpus and index construction for CommonsenseQA. |
| `build_cosmosqa.py` | Corpus and index construction for CosmosQA. |
| `build_winograd.py` | Corpus and index construction for Winograd. |
| `build_bqa.py` | Corpus and index construction for LogicBench BQA. |
| `build_mcqa.py` | Corpus and index construction for LogicBench MCQA. |
| `build_all.py` | Batch execution entry point. |

## Dependencies

```bash
python -m pip install sentence-transformers faiss-cpu tqdm
```

If you use the GPU version of FAISS, install the corresponding version according to your local CUDA environment.

## Common Execution Format

```bash
python build_<dataset>.py \
  --input_path "" \
  --raw_output_dir data/corpus/raw \
  --processed_output_dir data/corpus/processed \
  --embedding_model llama3-8b
```

`--input_path` should point to a locally downloaded or preprocessed data file, which can be JSON, JSONL, CSV, or TXT. No private local path is written in the code. The default value of `--embedding_model` is `llama3-8b`; if your actual model directory or model name is different, replace it manually.

## Output Files

Each dataset generates:

```text
data/corpus/raw/<dataset>.json
data/corpus/processed/<chunk_size>_<min_sentence>_<overlap>/<dataset>/chunks.json
data/corpus/processed/<chunk_size>_<min_sentence>_<overlap>/<dataset>/id_to_rawid.json
data/corpus/processed/<chunk_size>_<min_sentence>_<overlap>/<dataset>/vector.index
```

Where:

- `raw/<dataset>.json` stores the extracted raw RAG corpus.
- `chunks.json` stores the list of chunked texts.
- `id_to_rawid.json` stores the mapping from chunk id to the row id in the raw corpus.
- `vector.index` is the FAISS vector index.

## Chunking Rules

The chunking logic is consistent with the original RAG index script:

1. Use `!`, `。`, `，`, `！`, `?`, `？`, `,`, `.`, and `;` as sentence split symbols.
2. Use the original word-count function to count English words and Chinese characters.
3. Merge sentences according to `chunk_size`.
4. Use `overlap` to preserve sentence overlap between adjacent chunks.
5. If the last chunk contains fewer sentences than `min_sentence`, merge it into the previous chunk.

## Default Chunking Parameters

| Dataset | chunk_size | min_sentence | overlap |
| --- | ---: | ---: | ---: |
| 2WikiMultiHopQA | 200 | 2 | 2 |
| HotpotQA | 200 | 2 | 2 |
| MedQA | 200 | 2 | 2 |
| SQuAD | 50 | 2 | 2 |
| SciQ | 50 | 1 | 1 |
| CommonsenseQA | 50 | 1 | 1 |
| CosmosQA | 50 | 1 | 1 |
| Winograd | 50 | 1 | 1 |
| LogicBench BQA | 50 | 1 | 1 |
| LogicBench MCQA | 50 | 1 | 1 |

## Corpus Field Rules

| Dataset | Raw corpus field | Extraction priority |
| --- | --- | --- |
| 2WikiMultiHopQA | `paragraph_text` | `paragraph_text`, `context`, `text` |
| HotpotQA | `paragraph_text` | `paragraph_text`, `context`, `text` |
| SQuAD | `paragraph_text` | `paragraph_text`, `context`, `text` |
| MedQA | `text` | `text`, `paragraph_text`, `context`, `support`, `question` |
| SciQ | `text` | `text`, `support`, `context` |
| CommonsenseQA | `text` | `text`, `context`, `support` |
| CosmosQA | `text` | `text`, `context`, `support` |
| Winograd | `text` | `text`, `content`, `context`, `support` |
| LogicBench BQA | `text` | `text`, `context`, `support` |
| LogicBench MCQA | `text` | `text`, `context`, `support` |

If the input file does not contain the fields above, the script concatenates fields such as question, options, and answer into a retrievable corpus text to complete the RAG corpus generation workflow for missing datasets.

## Examples

Generate only the corpus and chunks without vector encoding:

```bash
python build_sciq.py \
  --input_path "" \
  --skip_encoding
```

Generate the corpus, chunks, and vector index:

```bash
python build_hotpotqa.py \
  --input_path "" \
  --embedding_model llama3-8b
```

Batch execution:

```bash
python build_all.py \
  --input_root "" \
  --embedding_model llama3-8b
```

`build_all.py` searches under `--input_root` for files with the same names as the datasets, such as `hotpotqa.json`, `sciq.json`, and `cosmosqa.csv`.
