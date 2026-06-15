# MindMap Knowledge Graph Construction Flow

This document describes the knowledge graph construction, retrieval, and reasoning workflow in this directory.

## Directory Roles

- `dataset/<dataset>/data/`: Raw data files, such as `hotpotqa.json`, `sciq_2000.json`, parquet files, and so on.
- `dataset/<dataset>/<dataset>.txt`: The corpus file created by concatenating raw text, used for entity and relation extraction.
- `dataset/<dataset>/gen_text.py`: Exists only in `bqa`, `mcqa`, and `winograd`; used to concatenate the `text` fields in JSON files into a `.txt` file.
- `dataset/<dataset>/gen_csv.py`: The entity/relation extraction script used by most datasets; extracts entities and triples from `.txt` files.
- `dataset/<dataset>/gen_keywords.py`: The keyword extraction script used by most datasets; extracts keywords from questions and writes them to `keyword.json`.
- `dataset/<dataset>/gen.py`: The integrated preprocessing script for `bqa`, `mcqa`, and `winograd`; includes keyword extraction, entity/relation extraction, filtering, and embedding generation.
- `dataset/<dataset>/encode_keyword_entity.py`: Reads `entities.csv` and `keyword.json`, then generates entity and keyword embeddings.

## Main Intermediate Files

- `<dataset>.txt`: The concatenated knowledge corpus.
- `raw_entities.csv`: Raw entities extracted from the corpus by the LLM, with columns `idx,Entity`.
- `raw_relations.csv`: Raw relation triples extracted from the corpus by the LLM, with columns `Entity1,Relation,Entity2`.
- `keyword.json`: Keywords, questions, contexts, options, and answers for each question.
- `entities.csv`: The entity table used for database import and embedding.
- `relation.csv`: The relation triples used for writing to Neo4j.
- `entity_embeddings.pkl`: Entity vectors, containing `entities` and `embeddings`.
- `keyword_embeddings.pkl`: Keyword vectors, containing `keywords` and `embeddings`.

## Construction Flow

1. Text preparation
   For `bqa`, `mcqa`, and `winograd`, you can first run `dataset/<dataset>/gen_text.py` to concatenate the `text` fields in the JSON file into `<dataset>.txt`. Other dataset directories already contain the corresponding `.txt` corpus files.

2. Entity and relation extraction
   Most datasets use `dataset/<dataset>/gen_csv.py`. The script reads `<dataset>.txt`, prompts the model to extract entities in `extract_entities()`, prompts the model to extract `(Entity 1, Relation, Entity 2)` triples in `extract_relations_based_on_entities()`, and finally writes `raw_entities.csv` and `raw_relations.csv` through `save_to_csv()`. Some scripts may write `raw_entities_deepseek.csv`, `raw_relations_deepseek.csv`, or `entities_deepseek.csv`, `relations_deepseek.csv`; these are historical output names for the corresponding datasets.

3. Keyword extraction
   Most datasets use `dataset/<dataset>/gen_keywords.py`. The core prompt is in `extract_keywords()`, which asks the model to extract comma-separated keywords from the question. The result is written to `keyword.json`.

4. Entity/relation filtering and standardization
   In `bqa`, `mcqa`, and `winograd`, `gen.py` uses `keyword.json` in `filter_with_keywords()` to filter `raw_entities.csv` and `raw_relations.csv`, generating `entities.csv` and `relation.csv`. Other dataset directories already contain `entities.csv` and `relation.csv`; if raw files are regenerated, these two standard files should be produced using the same rules or through manual cleaning.

5. Vector generation
   `dataset/<dataset>/encode_keyword_entity.py` reads `entities.csv` and `keyword.json`, then uses a SentenceTransformer model to generate `entity_embeddings.pkl` and `keyword_embeddings.pkl`. The `gen.py` scripts in `bqa`, `mcqa`, and `winograd` also complete the same step in `encode_and_save_embeddings()`.

6. Writing to Neo4j
   In the main workflow, `mindmap*.py` reads `dataset/<dataset>/relation.csv`, connects to Neo4j through `GraphDatabase.driver()`, and executes `MERGE` for each relation row: the node label is `Entity_<dataset>`, and the relation type is `<Relation>_<dataset>`. The database used is Neo4j, and the connection parameters come from `MINDMAP_NEO4J_URI`, `MINDMAP_NEO4J_USER`, and `MINDMAP_NEO4J_PASSWORD`.

7. Graph retrieval
   `find_shortest_path()` uses Cypher `allShortestPaths` to find paths within 5 hops between entities; `get_entity_neighbors()` retrieves outgoing neighbors of matched entities. Entity matching first computes cosine similarity between `keyword_embeddings.pkl` and `entity_embeddings.pkl`, mapping question keywords to knowledge graph entities.

8. Prompt generation and final reasoning
   `prompt_path_finding()` converts graph paths into natural-language evidence; `prompt_neighbor()` converts neighbor relations into natural-language evidence; `final_answer_commonsense_knowledge()` combines the question, context, options, path evidence, and neighbor evidence for different datasets, asking the model to output the answer, reasoning process, and decision tree.

9. Embedding model
   multilingual-e5-large

## Execution Script

The new script is `sh/run_mindmap.sh`. Common commands:

```bash
sbatch sh/run_mindmap.sh csv hotpotqa
sbatch sh/run_mindmap.sh keywords hotpotqa
sbatch sh/run_mindmap.sh embed hotpotqa
sbatch sh/run_mindmap.sh mindmap hotpotqa qwen
sbatch sh/run_mindmap.sh all bqa qwen
```

Set the following as needed before running:

```bash
export MINDMAP_LLM_MODEL_PATH="“”"
export MINDMAP_EMBEDDING_MODEL_PATH="“”"
export MINDMAP_NEO4J_URI="“”"
export MINDMAP_NEO4J_USER="“”"
export MINDMAP_NEO4J_PASSWORD="“”"
```
