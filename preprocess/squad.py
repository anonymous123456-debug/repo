from preprocess_utils import build_parser, load_hf_records, make_splits, print_summary, write_json_splits


DATASET_KEY = "squad"
DATASET_NAME = "rajpurkar/squad"


def format_record(row):
    return {
        "id": row.get("id", ""),
        "title": row.get("title", ""),
        "context": row.get("context", ""),
        "question": row.get("question", ""),
        "answers": row.get("answers", {}),
    }


def main():
    parser = build_parser("Preprocess SQuAD.")
    args = parser.parse_args()
    records = [format_record(row) for row in load_hf_records(DATASET_NAME, args.source_path, args.split)]
    sampled, eval_records = make_splits(records, args.sample_size, args.eval_size, args.seed)
    paths = write_json_splits(DATASET_KEY, sampled, eval_records, args.output_dir)
    print_summary(DATASET_KEY, sampled, eval_records, paths)


if __name__ == "__main__":
    main()
