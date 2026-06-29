from corpus_index_utils import run_pipeline


CONFIG = {
    "dataset": "cosmosqa",
    "output_field": "text",
    "text_fields": ("text", "context", "support"),
    "fallback_fields": ("context", "question", "answer0", "answer1", "answer2", "answer3"),
    "chunk_size": 50,
    "min_sentence": 1,
    "overlap": 1,
}


if __name__ == "__main__":
    run_pipeline(CONFIG)
