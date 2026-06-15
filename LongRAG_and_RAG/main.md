# LongRAG and RAG Baseline

This folder contains code for two retrieval-based QA baselines:

- `ext_fil`: LongRAG, which combines retrieval, extraction, and filtering.
- `rb`: vanilla RAG, which retrieves and reranks chunks before answer generation.

The original corpus files are stored under `data/corpus/raw/`.

## Embedding and Reranking Models

The embedding model is `multilingual-e5-large`. It is used in `src/gen_index.py` to encode corpus chunks, and in `src/main.py` or `src/server.py` to encode questions for vector retrieval.

The rerank model is stored under `rerank/`. It is a `BertForSequenceClassification` reranker based on the following configuration:

```text
_name_or_path: microsoft/MiniLM-L12-H384-uncased
model_type: bert
hidden_size: 384
num_hidden_layers: 12
num_attention_heads: 12
intermediate_size: 1536
max_position_embeddings: 512
vocab_size: 30522
transformers_version: 4.4.2
activation: torch.nn.modules.linear.Identity
```

Please download these two models yourself and place them in the project root before building indexes or running inference.

## Corpus Encoding Flow

The corpus indexing flow is implemented in `src/gen_index.py`.

1. Load the raw corpus file from `data/corpus/raw/<dataset>.json`.
2. Read each row's corpus field in this order: `paragraph_text`, `text`, then `ch_contenn`.
3. Split text into sentence-like spans using these separators: `!`, Chinese full stop and comma, Chinese exclamation mark, `?`, Chinese question mark, `,`, `.`, and `;`.
4. Merge sentence spans into chunks until each chunk reaches the target word count.
5. Use `get_word_count()` to count English word-like spans and CJK characters.
6. The default chunking parameters are:
   - `chunk_size=200`
   - `min_sentence=2`
   - `overlap=2`
7. Save processed files to `data/corpus/processed/<chunk_size>_<min_sentence>_<overlap>/<dataset>/`.
8. Write `chunks.json` to store processed chunks.
9. Write `id_to_rawid.json` to map each chunk id back to the original corpus row id.
10. Encode all chunks with `multilingual-e5-large`.
11. Build a FAISS `IndexFlatIP` index.
12. Save the FAISS index as `vector.index`.
13. The datasets using the `200_2_2` setting are HotpotQA, 2WikiMultiHopQA, and MedQA. The dataset using the `50_2_2` setting is SQuAD. Other datasets use the `50_1_1` setting.

Example:

```bash
cd src
python gen_index.py --dataset winograd --chunk_size 200 --min_sentence 2 --overlap 2
```

## Inference Flow

The inference flow is implemented in `src/main.py`.

1. Load the FAISS index from `--r_path/<dataset>/vector.index`.
2. Load `chunks.json` and `id_to_rawid.json`.
3. Load the first `--sample_size` evaluation samples. The default value is `200`.
4. Encode the question with `multilingual-e5-large`.
5. Retrieve `--top_k1` coarse-grained chunks from FAISS.
6. Rerank the retrieved chunks with the MiniLM/BERT sequence-classification rerank model and keep the top `--top_k2` results.
7. For `--rb`, generate the answer directly from the reranked chunks. This is vanilla RAG.
8. For `--ext_fil`, run the LongRAG extractor and filter pipeline before final answer generation.
9. Cache predictions under `src/log/`.
10. Compute F1 and accuracy-style summary fields.

## Quick Start

Install dependencies:

```bash
cd /path/to/LongRAG_and_RAG
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Place the retrieval models in the project root:

```text
multilingual-e5-large/
rerank/
```

Configure generation model paths in `config/config.yaml`, or override model-related behavior in your local environment.

Run both LongRAG and RAG:

```bash
bash sh/main.sh
```

Run only LongRAG:

```bash
MODE=ext_fil DATASET=winograd MODEL=misral bash sh/main.sh
```

Run only vanilla RAG:

```bash
MODE=rb DATASET=winograd MODEL=misral bash sh/main.sh
```

Default override parameters:

```bash
DATASET=hotpotqa \
MODEL=misral \
SAMPLE_SIZE=200 \
TOP_K1=100 \
TOP_K2=20 \
CHUNK_SIZE=200 \
MIN_SENTENCE=2 \
OVERLAP=2 \
bash sh/main.sh
```

When `vector.index` is missing, `sh/main.sh` automatically builds the FAISS index. To force rebuilding:

```bash
BUILD_INDEX=1 DATASET=winograd bash sh/main.sh
```

## Outputs

Index outputs:

```text
data/corpus/processed/200_2_2/<dataset>/chunks.json
data/corpus/processed/200_2_2/<dataset>/id_to_rawid.json
data/corpus/processed/200_2_2/<dataset>/vector.index
```

Inference logs and cached predictions:

```text
src/log/<index_setting>/<dataset>/<model>/<lrag_model_or_base>/<timestamp>/
```
