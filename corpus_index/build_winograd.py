from corpus_index_utils import run_pipeline


CONFIG = {
    "dataset": "winograd",
    "output_field": "text",
    "text_fields": ("text", "content", "context", "support"),
    "fallback_fields": ("content", "pronoun", "options", "rendered_output"),
    "chunk_size": 50,
    "min_sentence": 1,
    "overlap": 1,
}


if __name__ == "__main__":
    run_pipeline(CONFIG)
