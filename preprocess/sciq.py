import random

from preprocess_utils import build_parser, load_hf_records, make_splits, print_summary, write_json_splits


DATASET_KEY = "sciq"
DATASET_NAME = "allenai/sciq"


def format_records(records, seed):
    rng = random.Random(seed)
    formatted = []
    for idx, row in enumerate(records, 1):
        correct = row.get("correct_answer", "")
        choices = [
            correct,
            row.get("distractor1", ""),
            row.get("distractor2", ""),
            row.get("distractor3", ""),
        ]
        rng.shuffle(choices)
        formatted.append(
            {
                "id": f"sample_{idx}",
                "question": row.get("question", ""),
                "support": row.get("support", ""),
                "choices_1": choices[0],
                "choices_2": choices[1],
                "choices_3": choices[2],
                "choices_4": choices[3],
                "correct_answer": f"choices_{choices.index(correct) + 1}",
            }
        )
    return formatted


def main():
    parser = build_parser("Preprocess SciQ.")
    args = parser.parse_args()
    raw_records = load_hf_records(DATASET_NAME, args.source_path, args.split)
    sampled_raw, _ = make_splits(raw_records, args.sample_size, args.eval_size, args.seed)
    sampled = format_records(sampled_raw, args.seed)
    eval_records = sampled[: min(args.eval_size, len(sampled))]
    paths = write_json_splits(DATASET_KEY, sampled, eval_records, args.output_dir)
    print_summary(DATASET_KEY, sampled, eval_records, paths)


if __name__ == "__main__":
    main()
