from corpus_index_utils import run_pipeline


CONFIG = {
    "dataset": "sciq",
    "output_field": "text",
    "text_fields": ("text", "support", "context"),
    "fallback_fields": ("support", "question", "choices_1", "choices_2", "choices_3", "choices_4", "correct_answer"),
    "chunk_size": 50,
    "min_sentence": 1,
    "overlap": 1,
}


if __name__ == "__main__":
    run_pipeline(CONFIG)
