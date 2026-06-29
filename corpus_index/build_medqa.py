from corpus_index_utils import run_pipeline


CONFIG = {
    "dataset": "medqa",
    "output_field": "text",
    "text_fields": ("text", "paragraph_text", "context", "support", "question"),
    "fallback_fields": ("question", "options", "answer"),
    "chunk_size": 200,
    "min_sentence": 2,
    "overlap": 2,
}


if __name__ == "__main__":
    run_pipeline(CONFIG)
