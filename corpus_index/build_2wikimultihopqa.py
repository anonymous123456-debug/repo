from corpus_index_utils import run_pipeline


CONFIG = {
    "dataset": "2wikimultihopqa",
    "output_field": "paragraph_text",
    "text_fields": ("paragraph_text", "context", "text"),
    "fallback_fields": ("question", "context", "answer"),
    "chunk_size": 200,
    "min_sentence": 2,
    "overlap": 2,
}


if __name__ == "__main__":
    run_pipeline(CONFIG)
