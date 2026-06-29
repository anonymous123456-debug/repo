from preprocess_utils import (
    build_parser,
    load_hf_records,
    make_splits,
    print_summary,
    write_csv_splits,
    write_json_splits,
)


DATASET_KEY = "cosmosqa"
DATASET_NAME = "allenai/cosmos_qa"
FIELDNAMES = ["id", "context", "question", "answer0", "answer1", "answer2", "answer3", "label"]


def format_record(row):
    label = row.get("label", "")
    if isinstance(label, str) and label.isdigit():
        label = int(label)
    return {
        "id": row.get("id", ""),
        "context": row.get("context", ""),
        "question": row.get("question", ""),
        "answer0": row.get("answer0", ""),
        "answer1": row.get("answer1", ""),
        "answer2": row.get("answer2", ""),
        "answer3": row.get("answer3", ""),
        "label": label,
    }


def main():
    parser = build_parser("Preprocess CosmosQA.")
    args = parser.parse_args()
    records = [format_record(row) for row in load_hf_records(DATASET_NAME, args.source_path, args.split)]
    sampled, eval_records = make_splits(records, args.sample_size, args.eval_size, args.seed)
    json_paths = write_json_splits(DATASET_KEY, sampled, eval_records, args.output_dir)
    csv_paths = write_csv_splits(DATASET_KEY, sampled, eval_records, args.output_dir, FIELDNAMES)
    print_summary(DATASET_KEY, sampled, eval_records, json_paths + csv_paths)


if __name__ == "__main__":
    main()
