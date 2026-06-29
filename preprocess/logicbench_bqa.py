from pathlib import Path

from preprocess_utils import build_parser, make_splits, print_summary, read_json, write_json_splits


DATASET_KEY = "logicbench_bqa"


def source_root(source_path):
    if not source_path:
        raise ValueError("LogicBench is distributed from GitHub. Pass --source_path after downloading it.")
    path = Path(source_path)
    candidate = path / "BQA"
    return candidate if candidate.exists() else path


def collect_samples(root):
    files = sorted(root.rglob("data_instances.json"))
    if root.is_file():
        files = [root]
    records = []
    for file_path in files:
        records.extend(read_json(file_path))
    return records


def format_records(samples):
    records = []
    for sample in samples:
        context = sample.get("context", "")
        qa_pairs = sample.get("qa_pairs")
        if isinstance(qa_pairs, list):
            for qa in qa_pairs:
                records.append(
                    {
                        "context": context,
                        "question": qa.get("question", ""),
                        "answer": str(qa.get("answer", "")).lower(),
                    }
                )
        elif sample.get("question") is not None:
            records.append(
                {
                    "context": context,
                    "question": sample.get("question", ""),
                    "answer": str(sample.get("answer", "")).lower(),
                }
            )
    return records


def main():
    parser = build_parser("Preprocess LogicBench BQA.", default_sample_size=200, default_eval_size=200)
    args = parser.parse_args()
    records = format_records(collect_samples(source_root(args.source_path)))
    sampled, eval_records = make_splits(records, args.sample_size, args.eval_size, args.seed)
    paths = write_json_splits(DATASET_KEY, sampled, eval_records, args.output_dir)
    print_summary(DATASET_KEY, sampled, eval_records, paths)


if __name__ == "__main__":
    main()
