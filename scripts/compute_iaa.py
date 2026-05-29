"""Compute inter-annotator agreement for free-text QA annotations.

The expected CSV schema is:

id,question,annotator_answer,notes

The script reports strict normalized exact agreement and soft agreement based
on token F1. Soft agreement is useful for free-text answers where two
annotators may use different but overlapping wording.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import string
import unicodedata
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATOR_A = ROOT / "data" / "iaaa" / "annotator_a.csv"
DEFAULT_ANNOTATOR_B = ROOT / "data" / "iaaa" / "annotator_b.csv"
DEFAULT_REPORT = ROOT / "data" / "iaaa" / "iaa_results.json"
DEFAULT_DISAGREEMENTS = ROOT / "data" / "iaaa" / "iaa_disagreements.csv"


REQUIRED_FIELDS = ["id", "question", "annotator_answer", "notes"]


def ascii_fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_answer(text: str) -> str:
    text = ascii_fold(text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"\b\d{1,3}(?:[.,]\d{3})+\b", lambda m: re.sub(r"[.,]", "", m.group(0)), text)
    text = re.sub(r"[/_-]+", " ", text)
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def tokens(text: str) -> list[str]:
    return normalize_answer(text).split()


def token_f1(answer_a: str, answer_b: str) -> float:
    a_tokens = tokens(answer_a)
    b_tokens = tokens(answer_b)
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0

    common = Counter(a_tokens) & Counter(b_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(a_tokens)
    recall = overlap / len(b_tokens)
    return 2 * precision * recall / (precision + recall)


def load_annotations(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != REQUIRED_FIELDS:
            raise ValueError(f"{path} must have columns {REQUIRED_FIELDS}; got {reader.fieldnames}")
        rows = []
        for line_no, row in enumerate(reader, start=2):
            if row.get(None):
                raise ValueError(f"{path}:{line_no} has extra CSV columns: {row[None]}")
            rows.append({field: (row.get(field) or "").strip() for field in REQUIRED_FIELDS})
    return rows


def align_rows(rows_a: list[dict[str, str]], rows_b: list[dict[str, str]]) -> list[tuple[dict[str, str], dict[str, str]]]:
    by_id_a = {row["id"]: row for row in rows_a}
    by_id_b = {row["id"]: row for row in rows_b}
    ids_a = set(by_id_a)
    ids_b = set(by_id_b)
    if ids_a != ids_b:
        missing_from_b = sorted(ids_a - ids_b)
        missing_from_a = sorted(ids_b - ids_a)
        raise ValueError(f"ID mismatch. Missing from B: {missing_from_b}; missing from A: {missing_from_a}")

    aligned = []
    for row_a in rows_a:
        row_b = by_id_b[row_a["id"]]
        if row_a["question"].strip() != row_b["question"].strip():
            raise ValueError(f"Question mismatch for {row_a['id']}")
        aligned.append((row_a, row_b))
    return aligned


def write_disagreements(path: Path, disagreements: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "question",
        "annotator_a_answer",
        "annotator_b_answer",
        "normalized_a",
        "normalized_b",
        "token_f1",
        "notes_a",
        "notes_b",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(disagreements)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator-a", type=Path, default=DEFAULT_ANNOTATOR_A)
    parser.add_argument("--annotator-b", type=Path, default=DEFAULT_ANNOTATOR_B)
    parser.add_argument("--soft-threshold", type=float, default=0.80)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--disagreements", type=Path, default=DEFAULT_DISAGREEMENTS)
    args = parser.parse_args()

    rows_a = load_annotations(args.annotator_a)
    rows_b = load_annotations(args.annotator_b)
    aligned = align_rows(rows_a, rows_b)

    exact_matches = 0
    soft_matches = 0
    f1_scores = []
    blank_count = 0
    disagreements = []

    for row_a, row_b in aligned:
        answer_a = row_a["annotator_answer"]
        answer_b = row_b["annotator_answer"]
        if not answer_a or not answer_b:
            blank_count += 1

        normalized_a = normalize_answer(answer_a)
        normalized_b = normalize_answer(answer_b)
        exact = normalized_a == normalized_b
        f1 = token_f1(answer_a, answer_b)
        soft = f1 >= args.soft_threshold

        exact_matches += int(exact)
        soft_matches += int(soft)
        f1_scores.append(f1)

        if not exact:
            disagreements.append(
                {
                    "id": row_a["id"],
                    "question": row_a["question"],
                    "annotator_a_answer": answer_a,
                    "annotator_b_answer": answer_b,
                    "normalized_a": normalized_a,
                    "normalized_b": normalized_b,
                    "token_f1": f"{f1:.4f}",
                    "notes_a": row_a["notes"],
                    "notes_b": row_b["notes"],
                }
            )

    total = len(aligned)
    report = {
        "annotator_a": str(args.annotator_a),
        "annotator_b": str(args.annotator_b),
        "item_count": total,
        "blank_answer_count": blank_count,
        "exact_agreement_count": exact_matches,
        "exact_agreement": exact_matches / total if total else 0.0,
        "soft_threshold": args.soft_threshold,
        "soft_agreement_count": soft_matches,
        "soft_agreement": soft_matches / total if total else 0.0,
        "mean_token_f1": sum(f1_scores) / total if total else 0.0,
        "disagreement_count": len(disagreements),
        "disagreements_path": str(args.disagreements),
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_disagreements(args.disagreements, disagreements)

    print(f"Items: {total}")
    print(f"Blank answers: {blank_count}")
    print(f"Exact agreement: {exact_matches}/{total} = {report['exact_agreement'] * 100:.2f}%")
    print(
        f"Soft agreement (token F1 >= {args.soft_threshold:.2f}): "
        f"{soft_matches}/{total} = {report['soft_agreement'] * 100:.2f}%"
    )
    print(f"Mean token F1: {report['mean_token_f1'] * 100:.2f}")
    print(f"Wrote report to {args.report}")
    print(f"Wrote disagreements to {args.disagreements}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
