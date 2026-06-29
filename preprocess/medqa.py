from pathlib import Path

from preprocess_utils import build_parser, make_splits, print_summary, read_records_from_file, write_json_splits


DATASET_KEY = "medqa"


def find_source_file(source_path, split):
    if not source_path:
        raise ValueError("MedQA is distributed from GitHub. Pass --source_path after downloading it.")
    path = Path(source_path)
    if path.is_file():
        return path
    candidates = sorted(path.rglob(f"*{split}*.jsonl")) + sorted(path.rglob(f"*{split}*.json"))
    if not candidates:
        candidates = sorted(path.rglob("*.jsonl")) + sorted(path.rglob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No JSON or JSONL file found under {path}")
    return candidates[0]


def format_record(row):
    return {
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "options": row.get("options", {}),
        "meta_info": row.get("meta_info", ""),
        "answer_idx": row.get("answer_idx", row.get("answerKey", "")),
    }


def main():
    parser = build_parser("Preprocess MedQA.")
    args = parser.parse_args()
    source_file = find_source_file(args.source_path, args.split)
    records = [format_record(row) for row in read_records_from_file(source_file)]
    sampled, eval_records = make_splits(records, args.sample_size, args.eval_size, args.seed)
    paths = write_json_splits(DATASET_KEY, sampled, eval_records, args.output_dir)
    print_summary(DATASET_KEY, sampled, eval_records, paths)


if __name__ == "__main__":
    main()
