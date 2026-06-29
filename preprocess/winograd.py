from preprocess_utils import build_parser, load_hf_records, make_splits, print_summary, write_json_splits


DATASET_KEY = "winograd"
DATASET_NAME = "marcov/winograd_wsc_wsc273_promptsource"


def pick_options(row):
    options = row.get("options")
    if isinstance(options, list):
        return options
    values = [row.get("option_0"), row.get("option_1")]
    if all(value is not None for value in values):
        return values
    values = [row.get("option0"), row.get("option1")]
    if all(value is not None for value in values):
        return values
    candidates = row.get("candidates")
    if isinstance(candidates, list):
        return candidates[:2]
    return ["", ""]


def pick_answer(row, options):
    if row.get("rendered_output") is not None:
        return row.get("rendered_output")
    if row.get("answer") is not None:
        return row.get("answer")
    label = row.get("label")
    if isinstance(label, str) and label.isdigit():
        label = int(label)
    if isinstance(label, int) and 0 <= label < len(options):
        return options[label]
    return ""


def format_record(row, idx):
    options = pick_options(row)
    return {
        "id": row.get("id", idx),
        "text": row.get("text", row.get("content", row.get("sentence", ""))),
        "pronoun": row.get("pronoun", row.get("target_pronoun", "")),
        "options": options,
        "rendered_output": pick_answer(row, options),
    }


def main():
    parser = build_parser("Preprocess Winograd.", default_split="test")
    args = parser.parse_args()
    records = [
        format_record(row, idx)
        for idx, row in enumerate(load_hf_records(DATASET_NAME, args.source_path, args.split))
    ]
    sampled, eval_records = make_splits(records, args.sample_size, args.eval_size, args.seed)
    paths = write_json_splits(DATASET_KEY, sampled, eval_records, args.output_dir)
    print_summary(DATASET_KEY, sampled, eval_records, paths)


if __name__ == "__main__":
    main()
