"""Refine generated QA questions with OpenRouter.

This script rewrites the auto-generated QA candidates into more natural
English WH questions while preserving source/evidence metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ANNOTATION_DIR = DATA_DIR / "annotations"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"


BANNED_PHRASES = [
    "according to the source",
    "based on the source",
    "in the collected",
    "collected document",
    "collected material",
    "missing from this evidence",
    "from this evidence",
    "evidence:",
]


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def openrouter_chat(messages: list[dict], model: str, timeout: int) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing in .env")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 6000,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    site_url = os.environ.get("OPENROUTER_SITE_URL", "").strip()
    app_name = os.environ.get("OPENROUTER_APP_NAME", "").strip()
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name

    request = Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {error_body[:500]}") from exc
    except URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

    data = json.loads(body)
    return data["choices"][0]["message"]["content"]


def parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return json.loads(text)


def needs_rewrite(question: str) -> bool:
    lowered = question.lower()
    if any(phrase in lowered for phrase in BANNED_PHRASES):
        return True
    if " is associated with " in lowered or " is discussed in " in lowered:
        return True
    if "which acronym appears" in lowered:
        return True
    return False


def build_prompt(batch: list[dict]) -> list[dict]:
    compact_batch = []
    for row in batch:
        compact_batch.append(
            {
                "id": row["id"],
                "current_question": row["question"],
                "answer": row["answers"][0],
                "source_title": row.get("source_title", ""),
                "question_type": row.get("question_type", ""),
                "evidence": row.get("evidence", "")[:700],
            }
        )

    system = (
        "You rewrite factual QA dataset items. Return only valid JSON. "
        "Do not invent facts outside the evidence."
    )
    user = {
        "task": "Rewrite each item into a natural English factual question.",
        "rules": [
            "Use normal WH-question wording such as What, Who, When, Which, Where, or How many.",
            "Do not mention source, collected document, evidence, missing text, blank, or cloze.",
            "The answer must be concise and directly supported by the evidence.",
            "You may clean a malformed answer if the evidence clearly supports a shorter correct answer.",
            "Prefer questions similar to factual QA test questions, not annotation instructions.",
            "Return JSON with exactly this shape: {\"items\":[{\"id\":\"...\",\"question\":\"...\",\"answer\":\"...\"}]}",
        ],
        "items": compact_batch,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def refine_rows(
    split: str,
    rows: list[dict],
    model: str,
    batch_size: int,
    timeout: int,
    sleep: float,
    only_weak: bool,
) -> list[dict]:
    id_to_row = {row["id"]: row for row in rows}
    candidates = [row for row in rows if (needs_rewrite(row["question"]) or not only_weak)]
    total = len(candidates)
    print(f"Refining {total} QA items with {model}...")

    for start in range(0, total, batch_size):
        batch = candidates[start : start + batch_size]
        messages = build_prompt(batch)
        last_error: Exception | None = None
        parsed = None
        for attempt in range(3):
            try:
                content = openrouter_chat(messages, model=model, timeout=timeout)
                parsed = parse_json_object(content)
                break
            except Exception as exc:  # noqa: BLE001 - keep batch robust for data build.
                last_error = exc
                time.sleep(1.5 + attempt)
        if parsed is None:
            print(f"Batch {start // batch_size + 1} failed: {last_error}", file=sys.stderr)
            continue

        for item in parsed.get("items", []):
            row = id_to_row.get(str(item.get("id", "")))
            if not row:
                continue
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if question.endswith("?") and len(question) >= 12:
                row["question"] = question
            if answer:
                row["answers"] = [answer]
            row["annotation_status"] = "llm_refined_needs_human_review"

        print(f"Refined {min(start + batch_size, total)}/{total}", flush=True)
        write_split(split, rows)
        if sleep:
            time.sleep(sleep)
    return rows


def write_split(split: str, rows: list[dict]) -> None:
    folder = TRAIN_DIR if split == "train" else TEST_DIR
    (folder / "questions.txt").write_text(
        "\n".join(row["question"] for row in rows) + "\n",
        encoding="utf-8",
    )
    (folder / "reference_answers.txt").write_text(
        "\n".join(";".join(row["answers"]) for row in rows) + "\n",
        encoding="utf-8",
    )
    (ANNOTATION_DIR / f"{split}_qa.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--all", action="store_true", help="Rewrite all rows instead of only weak-looking rows.")
    args = parser.parse_args()

    load_env(ROOT / ".env")
    model = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.1-70b-instruct").strip()

    train = json.loads((ANNOTATION_DIR / "train_qa.json").read_text(encoding="utf-8"))
    test = json.loads((ANNOTATION_DIR / "test_qa.json").read_text(encoding="utf-8"))

    train = refine_rows("train", train, model, args.batch_size, args.timeout, args.sleep, only_weak=not args.all)
    test = refine_rows("test", test, model, args.batch_size, args.timeout, args.sleep, only_weak=not args.all)

    write_split("train", train)
    write_split("test", test)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
