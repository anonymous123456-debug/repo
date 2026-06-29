import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []


TEXT_FIELD_ORDER = ("paragraph_text", "text", "ch_contenn", "context", "support", "content")


def parse_args(config):
    parser = argparse.ArgumentParser(description=f"Build corpus and vector index for {config['dataset']}.")
    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--raw_output_dir", type=str, default="data/corpus/raw")
    parser.add_argument("--processed_output_dir", type=str, default="data/corpus/processed")
    parser.add_argument("--embedding_model", type=str, default="llama3-8b")
    parser.add_argument("--chunk_size", type=int, default=config.get("chunk_size", 200))
    parser.add_argument("--min_sentence", type=int, default=config.get("min_sentence", 2))
    parser.add_argument("--overlap", type=int, default=config.get("overlap", 2))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--skip_encoding", action="store_true")
    parser.add_argument("--only_index", action="store_true")
    return parser.parse_args()


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


def read_text(path):
    text = Path(path).read_text(encoding="utf-8")
    return [{"text": text}]


def read_records(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return read_json(path)
    if suffix == ".jsonl":
        return read_jsonl(path)
    if suffix == ".csv":
        return read_csv(path)
    if suffix == ".txt":
        return read_text(path)
    raise ValueError(f"Unsupported input file type: {path}")


def stringify(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(stringify(v) for v in value.values() if v is not None).strip()
    if isinstance(value, (list, tuple)):
        return " ".join(stringify(v) for v in value if v is not None).strip()
    return str(value)


def flatten_context(value):
    if isinstance(value, dict):
        titles = value.get("title") or value.get("titles") or []
        sentences = value.get("sentences") or value.get("context") or []
        if isinstance(titles, list) and isinstance(sentences, list) and len(titles) == len(sentences):
            merged = []
            for title, body in zip(titles, sentences):
                body_text = stringify(body)
                merged.append(f"{title}: {body_text}" if title else body_text)
            return " ".join(merged).strip()
    if isinstance(value, list):
        merged = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                merged.append(f"{item[0]}: {stringify(item[1])}")
            else:
                merged.append(stringify(item))
        return " ".join(merged).strip()
    return stringify(value)


def first_text(row, fields):
    for field in fields:
        value = row.get(field)
        if value:
            if field == "context":
                return flatten_context(value)
            return stringify(value)
    return ""


def combine_choices(value):
    if isinstance(value, dict):
        return " ".join(f"{key}: {text}" for key, text in value.items())
    if isinstance(value, list):
        return " ".join(stringify(item) for item in value)
    return stringify(value)


def build_default_text(row, fields):
    parts = []
    for field in fields:
        if field == "choices":
            text = combine_choices(row.get(field))
        elif field == "context":
            text = flatten_context(row.get(field))
        else:
            text = stringify(row.get(field))
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def normalize_records(records, config):
    normalized = []
    output_field = config.get("output_field", "text")
    text_fields = config.get("text_fields", TEXT_FIELD_ORDER)
    fallback_fields = config.get("fallback_fields", text_fields)
    for idx, row in enumerate(records):
        if not isinstance(row, dict):
            row = {"text": stringify(row)}
        text = first_text(row, text_fields) or build_default_text(row, fallback_fields)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            normalized.append({"id": row.get("id", row.get("question_id", idx)), output_field: text})
    return normalized


def raw_corpus_path(config, raw_output_dir):
    return Path(raw_output_dir) / f"{config['dataset']}.json"


def processed_dir(config, processed_output_dir, chunk_size, min_sentence, overlap):
    return Path(processed_output_dir) / f"{chunk_size}_{min_sentence}_{overlap}" / config["dataset"]


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def get_word_count(text):
    reg_ex = re.compile(r"[\W]")
    chinese_char_re = re.compile(r"([\u4e00-\u9fa5])")
    words = reg_ex.split(text.lower())
    word_list = []
    for word in words:
        if chinese_char_re.split(word):
            word_list.extend(chinese_char_re.split(word))
        else:
            word_list.append(word)
    return len([word for word in word_list if len(word.strip()) > 0])


def split_sentences(content, chunk_size, min_sentence, overlap):
    stop_list = ["!", "。", "，", "！", "?", "？", ",", ".", ";"]
    split_pattern = f"({'|'.join(map(re.escape, stop_list))})"
    sentences = re.split(split_pattern, content)
    if len(sentences) == 1:
        return sentences
    sentences = [sentences[i] + sentences[i + 1] for i in range(0, len(sentences) - 1, 2)]
    sentence_word_counts = [get_word_count(sentence) for sentence in sentences]
    chunks = []
    temp_text = ""
    temp_word_count = 0
    sentence_overlap_len = 0
    start_index = 0
    for i, sentence in enumerate(sentences):
        temp_text += sentence
        temp_word_count += sentence_word_counts[i]
        if temp_word_count >= chunk_size - sentence_overlap_len or i == len(sentences) - 1:
            if i + 1 > overlap:
                sentence_overlap_len = sum(sentence_word_counts[j] for j in range(i + 1 - overlap, i + 1))
            if chunks and start_index > overlap:
                start_index -= overlap
            chunk_text = "".join(sentences[start_index:i + 1])
            if not chunks:
                chunks.append(chunk_text)
            elif i == len(sentences) - 1 and (i - start_index + 1) < min_sentence:
                chunks[-1] += chunk_text
            else:
                chunks.append(chunk_text)
            temp_text = ""
            temp_word_count = 0
            start_index = i + 1
    return chunks


def content_from_raw_item(item):
    for field in TEXT_FIELD_ORDER:
        value = item.get(field)
        if value:
            return flatten_context(value) if field == "context" else stringify(value)
    return ""


def process_data(raw_records, chunk_size, min_sentence, overlap, save_path):
    id_to_rawid = {}
    processed_chunks = []
    for idx, item in tqdm(list(enumerate(raw_records)), total=len(raw_records), desc="Processing data"):
        content = content_from_raw_item(item)
        chunks = split_sentences(content, chunk_size, min_sentence, overlap) if content else []
        for i, _ in enumerate(chunks):
            id_to_rawid[len(processed_chunks) + i] = idx
        processed_chunks.extend(chunks)
    save_path.mkdir(parents=True, exist_ok=True)
    write_json(save_path / "chunks.json", processed_chunks)
    write_json(save_path / "id_to_rawid.json", id_to_rawid)
    return processed_chunks


def calculate_embeddings(content, model_path, vector_store_path, batch_size):
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    if not model_path:
        raise ValueError("Set --embedding_model before encoding.")
    model = SentenceTransformer(model_path)
    embeddings = model.encode(content, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(vector_store_path))


def run_pipeline(config):
    args = parse_args(config)
    raw_path = raw_corpus_path(config, args.raw_output_dir)
    if args.only_index:
        raw_records = read_json(raw_path)
    else:
        if not args.input_path:
            raise ValueError("Set --input_path to a local JSON, JSONL, CSV, or TXT corpus source.")
        source_records = read_records(args.input_path)
        raw_records = normalize_records(source_records, config)
        write_json(raw_path, raw_records)
    save_path = processed_dir(config, args.processed_output_dir, args.chunk_size, args.min_sentence, args.overlap)
    chunks = process_data(raw_records, args.chunk_size, args.min_sentence, args.overlap, save_path)
    if not args.skip_encoding:
        vector_path = save_path / "vector.index"
        start_time = time.time()
        calculate_embeddings(chunks, args.embedding_model, vector_path, args.batch_size)
        elapsed = time.time() - start_time
        print(json.dumps({"vector_index": str(vector_path), "encoding_seconds": round(elapsed, 2)}, ensure_ascii=False))
    print(json.dumps({"dataset": config["dataset"], "raw_corpus": str(raw_path), "processed_dir": str(save_path), "chunks": len(chunks)}, ensure_ascii=False, indent=2))
