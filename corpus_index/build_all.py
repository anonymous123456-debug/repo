import argparse
import subprocess
import sys
from pathlib import Path


DATASET_TO_SCRIPT = {
    "2wikimultihopqa": "build_2wikimultihopqa.py",
    "bqa": "build_bqa.py",
    "commonsenseqa": "build_commonsenseqa.py",
    "cosmosqa": "build_cosmosqa.py",
    "hotpotqa": "build_hotpotqa.py",
    "mcqa": "build_mcqa.py",
    "medqa": "build_medqa.py",
    "sciq": "build_sciq.py",
    "squad": "build_squad.py",
    "winograd": "build_winograd.py",
}


def main():
    parser = argparse.ArgumentParser(description="Run corpus and index builders for multiple datasets.")
    parser.add_argument("--input_root", type=str, default="")
    parser.add_argument("--raw_output_dir", type=str, default="data/corpus/raw")
    parser.add_argument("--processed_output_dir", type=str, default="data/corpus/processed")
    parser.add_argument("--embedding_model", type=str, default="llama3-8b")
    parser.add_argument("--skip_encoding", action="store_true")
    args = parser.parse_args()
    if not args.input_root:
        raise ValueError("Set --input_root to the folder containing dataset source files.")
    input_root = Path(args.input_root)
    base_dir = Path(__file__).resolve().parent
    for dataset, script in DATASET_TO_SCRIPT.items():
        candidates = sorted(input_root.glob(f"{dataset}.*"))
        if not candidates:
            raise FileNotFoundError(f"No source file found for {dataset} under {input_root}")
        command = [
            sys.executable,
            str(base_dir / script),
            "--input_path",
            str(candidates[0]),
            "--raw_output_dir",
            args.raw_output_dir,
            "--processed_output_dir",
            args.processed_output_dir,
            "--embedding_model",
            args.embedding_model,
        ]
        if args.skip_encoding:
            command.append("--skip_encoding")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
