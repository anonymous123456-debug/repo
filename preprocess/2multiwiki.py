from preprocess_utils import build_parser, flatten_context, load_hf_records, make_splits, print_summary, write_json_splits


DATASET_KEY = "2multiwiki"
DATASET_NAME = "framolfese/2WikiMultihopQA"


def format_record(row):
    return {
        "question_id": row.get("_id", row.get("id", row.get("question_id", ""))),
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "context": flatten_context(row.get("context", "")),
    }


def main():
    parser = build_parser("Preprocess 2WikiMultiHopQA.")
    args = parser.parse_args()
    records = [format_record(row) for row in load_hf_records(DATASET_NAME, args.source_path, args.split)]
    sampled, eval_records = make_splits(records, args.sample_size, args.eval_size, args.seed)
    paths = write_json_splits(DATASET_KEY, sampled, eval_records, args.output_dir)
    print_summary(DATASET_KEY, sampled, eval_records, paths)


if __name__ == "__main__":
    main()
