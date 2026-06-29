from pathlib import Path

from preprocess_utils import build_parser, make_splits, print_summary, read_json, write_json_splits


DATASET_KEY = "logicbench_mcqa"
LETTER_TO_CHOICE = {"A": "choice_1", "B": "choice_2", "C": "choice_3", "D": "choice_4", "E": "choice_5"}


def source_root(source_path):
    if not source_path:
        raise ValueError("LogicBench is distributed from GitHub. Pass --source_path after downloading it.")
    path = Path(source_path)
    candidate = path / "MCQA"
    return candidate if candidate.exists() else path


def collect_samples(root):
    files = sorted(root.rglob("data_instances.json"))
    if root.is_file():
        files = [root]
    records = []
    for file_path in files:
        records.extend(read_json(file_path))
    return records


def normalize_choices(choices):
    if not isinstance(choices, dict):
        return {}
    if any(key.startswith("choice_") for key in choices):
        return choices
    normalized = {}
    for idx, key in enumerate(sorted(choices.keys()), 1):
        normalized[f"choice_{idx}"] = choices[key]
    return normalized


def normalize_answer(answer):
    if answer in LETTER_TO_CHOICE:
        return LETTER_TO_CHOICE[answer]
    return answer


def format_records(samples):
    records = []
    for idx, sample in enumerate(samples, 1):
        if sample.get("choices") is None:
            continue
        records.append(
            {
                "id": sample.get("id", idx),
                "context": sample.get("context", ""),
                "question": sample.get("question", ""),
                "choices": normalize_choices(sample.get("choices", {})),
                "answer": normalize_answer(sample.get("answer", "")),
            }
        )
    return records


def main():
    parser = build_parser("Preprocess LogicBench MCQA.", default_sample_size=200, default_eval_size=200)
    args = parser.parse_args()
    records = format_records(collect_samples(source_root(args.source_path)))
    sampled, eval_records = make_splits(records, args.sample_size, args.eval_size, args.seed)
    paths = write_json_splits(DATASET_KEY, sampled, eval_records, args.output_dir)
    print_summary(DATASET_KEY, sampled, eval_records, paths)


if __name__ == "__main__":
    main()
