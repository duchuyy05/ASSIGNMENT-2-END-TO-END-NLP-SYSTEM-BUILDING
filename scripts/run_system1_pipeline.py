"""Run system 1 (MiniLM + DistilBERT) and evaluate EM/F1/Recall."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "system_outputs" / "system_output_1.txt"
DEFAULT_TRACE = ROOT / "system_outputs" / "system_output_1_trace.jsonl"
DEFAULT_QUESTIONS = ROOT / "data" / "test" / "questions.txt"
DEFAULT_REFERENCES = ROOT / "data" / "test" / "reference_answers.txt"


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-train-facts", action="store_true")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rag_command = [
        sys.executable,
        str(ROOT / "rag" / "rag_minilm_distilbert.py"),
        "--questions",
        str(args.questions),
        "--output",
        str(args.output),
        "--trace",
        str(args.trace),
        "--top-k",
        str(args.top_k),
        "--batch-size",
        str(args.batch_size),
        "--device",
        str(device),
    ]
    if args.limit is not None:
        rag_command.extend(["--limit", str(args.limit)])
    if args.no_train_facts:
        rag_command.append("--no-train-facts")
    run_command(rag_command)

    eval_command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_qa.py"),
        "--predictions",
        str(args.output),
        "--references",
        str(args.references),
    ]
    run_command(eval_command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
