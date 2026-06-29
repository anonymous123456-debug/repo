from preprocess_utils import build_parser, flatten_context, load_hf_records, make_splits, print_summary, write_json_splits


DATASET_KEY = "hotpotqa"
DATASET_NAME = "hotpotqa/hotpot_qa"
CONFIG_NAME = "distractor"


def format_record(row):
    return {
        "question_id": row.get("id", row.get("_id", row.get("question_id", ""))),
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "context": flatten_context(row.get("context", "")),
    }


def main():
    parser = build_parser("Preprocess HotpotQA.")
    args = parser.parse_args()
    records = [
        format_record(row)
        for row in load_hf_records(DATASET_NAME, args.source_path, args.split, config_name=CONFIG_NAME)
    ]
    sampled, eval_records = make_splits(records, args.sample_size, args.eval_size, args.seed)
    paths = write_json_splits(DATASET_KEY, sampled, eval_records, args.output_dir)
    print_summary(DATASET_KEY, sampled, eval_records, paths)


if __name__ == "__main__":
    main()
