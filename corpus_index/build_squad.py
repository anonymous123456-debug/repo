from corpus_index_utils import run_pipeline


CONFIG = {
    "dataset": "squad",
    "output_field": "paragraph_text",
    "text_fields": ("paragraph_text", "context", "text"),
    "fallback_fields": ("title", "context", "question"),
    "chunk_size": 50,
    "min_sentence": 2,
    "overlap": 2,
}


if __name__ == "__main__":
    run_pipeline(CONFIG)
