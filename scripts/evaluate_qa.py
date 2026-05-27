"""Evaluate QA outputs with SQuAD-style exact match and token F1."""

from __future__ import annotations

import argparse
import re
import string
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = ROOT / "system_outputs" / "system_output_1.txt"
DEFAULT_REFERENCES = ROOT / "data" / "test" / "reference_answers.txt"


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = {}
    for token in pred_tokens:
        common[token] = min(pred_tokens.count(token), gold_tokens.count(token))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_score(prediction: str, references: str, metric) -> float:
    answers = [answer.strip() for answer in references.split(";") if answer.strip()]
    return max(metric(prediction, answer) for answer in answers) if answers else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    args = parser.parse_args()

    predictions = read_lines(args.predictions)
    references = read_lines(args.references)
    if len(predictions) != len(references):
        raise ValueError(f"Line count mismatch: {len(predictions)} predictions vs {len(references)} references")

    em_scores = [best_score(pred, ref, exact_match) for pred, ref in zip(predictions, references)]
    f1_scores = [best_score(pred, ref, token_f1) for pred, ref in zip(predictions, references)]
    answer_recall = [float(score > 0.0) for score in f1_scores]

    total = len(references)
    print(f"Examples: {total}")
    print(f"Exact Match: {sum(em_scores) / total * 100:.2f}")
    print(f"Token F1: {sum(f1_scores) / total * 100:.2f}")
    print(f"Answer Recall: {sum(answer_recall) / total * 100:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
