"""Evaluate system outputs and plot a model comparison chart.

This script intentionally uses only the Python standard library. It writes:

- `results/model_comparison.csv`
- `results/model_comparison.svg`
- `results/model_comparison.json`
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import string
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCES = ROOT / "data" / "test" / "reference_answers.txt"
DEFAULT_RESULTS_DIR = ROOT / "results"


SYSTEMS = [
    ("System 1", "MiniLM + DistilBERT", ROOT / "system_outputs" / "system_output_1.txt"),
    ("System 2", "BGE + RoBERTa", ROOT / "system_outputs" / "system_output_2.txt"),
    ("System 3", "E5 + RoBERTa", ROOT / "system_outputs" / "system_output_3.txt"),
    ("System 4", "TF-IDF baseline", ROOT / "system_outputs" / "system_output_4.txt"),
]


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


def evaluate(predictions_path: Path, references: list[str]) -> dict[str, float | int]:
    predictions = read_lines(predictions_path)
    if len(predictions) != len(references):
        raise ValueError(
            f"{predictions_path} has {len(predictions)} predictions, "
            f"but references have {len(references)} lines"
        )
    em_scores = [best_score(pred, ref, exact_match) for pred, ref in zip(predictions, references)]
    f1_scores = [best_score(pred, ref, token_f1) for pred, ref in zip(predictions, references)]
    recall_scores = [float(score > 0.0) for score in f1_scores]
    total = len(references)
    return {
        "examples": total,
        "exact_match": sum(em_scores) / total * 100,
        "token_f1": sum(f1_scores) / total * 100,
        "answer_recall": sum(recall_scores) / total * 100,
    }


def write_csv(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    fields = ["system", "model", "examples", "exact_match", "token_f1", "answer_recall"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def svg_text(x: float, y: float, text: str, size: int = 13, anchor: str = "start", weight: str = "400") -> str:
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{escaped}</text>'


def write_svg(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    width = 980
    height = 520
    margin_left = 170
    margin_right = 40
    margin_top = 70
    margin_bottom = 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    colors = {
        "exact_match": "#2563eb",
        "token_f1": "#16a34a",
        "answer_recall": "#dc2626",
    }
    metrics = [
        ("exact_match", "Exact Match"),
        ("token_f1", "Token F1"),
        ("answer_recall", "Answer Recall"),
    ]
    group_height = plot_height / len(rows)
    bar_height = 20
    scale = plot_width / 100

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111827}.axis{stroke:#9ca3af;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.bar-label{fill:#111827}</style>',
        svg_text(width / 2, 34, "RAG Model Comparison on Local Test Set", size=20, anchor="middle", weight="700"),
    ]

    for tick in range(0, 101, 20):
        x = margin_left + tick * scale
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{height - margin_bottom}"/>')
        parts.append(svg_text(x, height - margin_bottom + 24, str(tick), size=11, anchor="middle"))
    parts.append(f'<line class="axis" x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}"/>')
    parts.append(svg_text(margin_left + plot_width / 2, height - 20, "Score (%)", size=13, anchor="middle"))

    for row_index, row in enumerate(rows):
        y_base = margin_top + row_index * group_height + 18
        label = f"{row['system']}: {row['model']}"
        parts.append(svg_text(margin_left - 12, y_base + 27, label, size=12, anchor="end", weight="700"))
        for metric_index, (metric_key, metric_label) in enumerate(metrics):
            score = float(row[metric_key])
            y = y_base + metric_index * (bar_height + 6)
            bar_width = score * scale
            parts.append(
                f'<rect x="{margin_left}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height}" '
                f'rx="2" fill="{colors[metric_key]}"/>'
            )
            parts.append(svg_text(margin_left + bar_width + 6, y + 15, f"{score:.2f}", size=11))

    legend_x = margin_left
    legend_y = 52
    for index, (metric_key, metric_label) in enumerate(metrics):
        x = legend_x + index * 150
        parts.append(f'<rect x="{x}" y="{legend_y - 12}" width="14" height="14" fill="{colors[metric_key]}"/>')
        parts.append(svg_text(x + 22, legend_y, metric_label, size=12))

    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    references = read_lines(args.references)
    rows: list[dict[str, str | float | int]] = []
    for system_name, model_name, output_path in SYSTEMS:
        if not output_path.exists():
            raise FileNotFoundError(f"Missing output file: {output_path}")
        metrics = evaluate(output_path, references)
        rows.append({"system": system_name, "model": model_name, **metrics})

    args.results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.results_dir / "model_comparison.csv"
    json_path = args.results_dir / "model_comparison.json"
    svg_path = args.results_dir / "model_comparison.svg"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_svg(svg_path, rows)

    for row in rows:
        print(
            f"{row['system']} ({row['model']}): "
            f"EM={float(row['exact_match']):.2f}, "
            f"F1={float(row['token_f1']):.2f}, "
            f"Recall={float(row['answer_recall']):.2f}"
        )
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
