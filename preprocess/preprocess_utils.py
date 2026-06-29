import argparse
import csv
import json
import random
from pathlib import Path

def build_parser(description, default_sample_size=2000, default_eval_size=200, default_split="train"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--source_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--split", type=str, default=default_split)
    parser.add_argument("--seed", type=int, default=931)
    parser.add_argument("--sample_size", type=int, default=default_sample_size)
    parser.add_argument("--eval_size", type=int, default=default_eval_size)
    return parser


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    for key in ("samples", "data", "train", "validation", "test"):
        if isinstance(data.get(key), list):
            return data[key]
    return [data]


def read_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_records_from_file(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return read_json(path)
    if suffix == ".jsonl":
        return read_jsonl(path)
    if suffix == ".csv":
        return read_csv(path)
    raise ValueError(f"Unsupported input file type: {path}")


def load_hf_records(dataset_name, source_path, split, config_name=None):
    if source_path:
        path = Path(source_path)
        if path.is_file():
            return read_records_from_file(path)
        dataset_source = str(path)
    else:
        dataset_source = dataset_name
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install the datasets package or pass --source_path to a local JSON/JSONL/CSV file.") from exc
    dataset = load_dataset(dataset_source, config_name) if config_name else load_dataset(dataset_source)
    active_split = split if split in dataset else next(iter(dataset.keys()))
    return [jsonable(row) for row in dataset[active_split]]


def sample_records(records, sample_size, seed):
    records = list(records)
    sample_size = min(sample_size, len(records))
    rng = random.Random(seed)
    return rng.sample(records, sample_size)


def make_splits(records, sample_size, eval_size, seed):
    sampled = sample_records(records, sample_size, seed)
    eval_records = sampled[: min(eval_size, len(sampled))]
    return sampled, eval_records


def dataset_output_dir(output_dir, dataset_key):
    path = Path(output_dir) / dataset_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, records):
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def write_csv(path, records, fieldnames):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_json_splits(dataset_key, sampled, eval_records, output_dir):
    out_dir = dataset_output_dir(output_dir, dataset_key)
    sampled_path = out_dir / f"sampled_{len(sampled)}.json"
    eval_path = out_dir / f"eval_{len(eval_records)}.json"
    write_json(sampled_path, sampled)
    write_json(eval_path, eval_records)
    return sampled_path, eval_path


def write_csv_splits(dataset_key, sampled, eval_records, output_dir, fieldnames):
    out_dir = dataset_output_dir(output_dir, dataset_key)
    sampled_path = out_dir / f"sampled_{len(sampled)}.csv"
    eval_path = out_dir / f"eval_{len(eval_records)}.csv"
    write_csv(sampled_path, sampled, fieldnames)
    write_csv(eval_path, eval_records, fieldnames)
    return sampled_path, eval_path


def print_summary(dataset_key, sampled, eval_records, paths):
    payload = {
        "dataset": dataset_key,
        "sampled_size": len(sampled),
        "eval_size": len(eval_records),
        "outputs": [str(path) for path in paths],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def text_join(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(text_join(item) for item in value if item is not None).strip()
    if isinstance(value, dict):
        return " ".join(text_join(item) for item in value.values()).strip()
    return str(value)


def flatten_context(context):
    if isinstance(context, dict):
        titles = context.get("title") or context.get("titles") or []
        sentences = context.get("sentences") or context.get("context") or []
        if isinstance(titles, list) and isinstance(sentences, list) and len(titles) == len(sentences):
            chunks = []
            for title, sentence_group in zip(titles, sentences):
                body = text_join(sentence_group)
                chunks.append(f"{title}: {body}" if title else body)
            return " ".join(chunks).strip()
    if isinstance(context, list):
        chunks = []
        for item in context:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                chunks.append(f"{item[0]}: {text_join(item[1])}")
            else:
                chunks.append(text_join(item))
        return " ".join(chunks).strip()
    return text_join(context)
