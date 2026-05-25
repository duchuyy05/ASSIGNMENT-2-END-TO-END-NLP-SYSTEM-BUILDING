"""Generate natural English QA pairs from chunks with OpenRouter."""

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
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATION_DIR = DATA_DIR / "annotations"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
WORK_PATH = ANNOTATION_DIR / "openrouter_generated_qa_work.json"


BANNED = [
    "according to",
    "based on",
    "source",
    "evidence",
    "collected",
    "missing",
    "blank",
    "context",
]


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key.strip()] = value


def openrouter_chat(messages: list[dict], model: str, timeout: int) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing in .env")
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.25,
        "max_tokens": 7000,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if os.environ.get("OPENROUTER_SITE_URL"):
        headers["HTTP-Referer"] = os.environ["OPENROUTER_SITE_URL"]
    if os.environ.get("OPENROUTER_APP_NAME"):
        headers["X-Title"] = os.environ["OPENROUTER_APP_NAME"]
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


def clean_text(value: str) -> str:
    value = value.replace("�", "")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def good_chunk(chunk: dict) -> bool:
    text = chunk["text"]
    if chunk.get("source_language") != "en":
        return False
    if len(text.split()) < 65:
        return False
    lowered = text.lower()
    if "digital librarylibraryemail" in lowered:
        return False
    useful = ["vnu", "uet", "university", "program", "admission", "faculty", "student", "research", "training"]
    return any(term in lowered for term in useful)


def build_prompt(batch: list[dict], per_chunk: int) -> list[dict]:
    chunks = []
    for chunk in batch:
        chunks.append(
            {
                "chunk_id": chunk["chunk_id"],
                "title": clean_text(chunk["title"]),
                "url": chunk["url"],
                "text": clean_text(chunk["text"])[:1500],
            }
        )
    system = (
        "You create factual question-answer pairs for a RAG evaluation dataset. "
        "Return only valid JSON and do not invent facts."
    )
    user = {
        "task": f"Create up to {per_chunk} natural English QA pairs from each chunk.",
        "rules": [
            "Questions must look like real QA questions: What, Who, When, Where, Which, or How many.",
            "Do not write cloze questions and do not mention source, evidence, chunk, context, or document.",
            "Answers must be short, concise, and directly supported by the chunk text.",
            "Prefer facts about history, admissions, academic programs, organizational structure, rankings, people, dates, numbers, contacts, and regulations.",
            "Use the exact chunk_id from the input.",
            "Return JSON with this exact shape: {\"items\":[{\"chunk_id\":\"...\",\"question\":\"...\",\"answer\":\"...\",\"evidence\":\"short supporting sentence\"}]}",
        ],
        "chunks": chunks,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def valid_item(item: dict, chunk_ids: set[str]) -> bool:
    question = clean_text(str(item.get("question", "")))
    answer = clean_text(str(item.get("answer", "")))
    chunk_id = str(item.get("chunk_id", ""))
    if chunk_id not in chunk_ids:
        return False
    if not question.endswith("?") or len(question) < 12:
        return False
    if len(answer) < 1 or len(answer) > 120:
        return False
    lowered = question.lower()
    if any(term in lowered for term in BANNED):
        return False
    if not question.lower().startswith(("what", "who", "when", "where", "which", "how")):
        return False
    return True


def write_outputs(items: list[dict], chunks_by_id: dict[str, dict], train_target: int, test_target: int) -> None:
    rows = []
    seen = set()
    for item in items:
        chunk = chunks_by_id[item["chunk_id"]]
        question = clean_text(item["question"])
        answer = clean_text(item["answer"])
        key = (question.lower(), answer.lower())
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "id": f"qa_{len(rows):05d}",
                "question": question,
                "answers": [answer],
                "source_id": chunk["document_id"],
                "source_title": chunk["title"],
                "source_url": chunk["url"],
                "chunk_id": chunk["chunk_id"],
                "question_type": "openrouter_natural",
                "evidence": clean_text(str(item.get("evidence", "")))[:600],
                "source_language": chunk.get("source_language", "en"),
                "qa_language": "en",
                "annotation_status": "llm_generated_needs_human_review",
            }
        )
        if len(rows) >= train_target + test_target:
            break

    train = rows[:train_target]
    test = rows[train_target : train_target + test_target]
    for split, split_rows, folder in [("train", train, TRAIN_DIR), ("test", test, TEST_DIR)]:
        for index, row in enumerate(split_rows):
            row["id"] = f"{split}_{index:04d}"
            row["split"] = split
        (folder / "questions.txt").write_text("\n".join(row["question"] for row in split_rows) + "\n", encoding="utf-8")
        (folder / "reference_answers.txt").write_text(
            "\n".join(";".join(row["answers"]) for row in split_rows) + "\n",
            encoding="utf-8",
        )
        (ANNOTATION_DIR / f"{split}_qa.json").write_text(
            json.dumps(split_rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-target", type=int, default=600)
    parser.add_argument("--test-target", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--per-chunk", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    load_env(ROOT / ".env")
    model = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.1-70b-instruct").strip()

    chunks = json.loads((PROCESSED_DIR / "chunks.json").read_text(encoding="utf-8"))
    selected_chunks = [chunk for chunk in chunks if good_chunk(chunk)]
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    target_total = args.train_target + args.test_target

    items = []
    processed = set()
    if args.resume and WORK_PATH.exists():
        work = json.loads(WORK_PATH.read_text(encoding="utf-8"))
        items = work.get("items", [])
        processed = set(work.get("processed_chunk_ids", []))

    print(f"Generating natural QA with {model}; existing={len(items)} target={target_total}")
    for start in range(0, len(selected_chunks), args.batch_size):
        batch = [chunk for chunk in selected_chunks[start : start + args.batch_size] if chunk["chunk_id"] not in processed]
        if not batch:
            continue
        messages = build_prompt(batch, args.per_chunk)
        parsed = None
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                content = openrouter_chat(messages, model=model, timeout=args.timeout)
                parsed = parse_json_object(content)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(1.5 + attempt)
        if parsed is None:
            print(f"Batch starting at {start} failed: {last_error}", file=sys.stderr)
            continue
        chunk_ids = {chunk["chunk_id"] for chunk in batch}
        for item in parsed.get("items", []):
            if valid_item(item, chunk_ids):
                item["question"] = clean_text(item["question"])
                item["answer"] = clean_text(item["answer"])
                item["evidence"] = clean_text(str(item.get("evidence", "")))
                items.append(item)
        processed.update(chunk_ids)
        WORK_PATH.write_text(
            json.dumps({"items": items, "processed_chunk_ids": sorted(processed)}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Generated {len(items)} valid QA items", flush=True)
        write_outputs(items, chunks_by_id, args.train_target, args.test_target)
        if len({(item['question'].lower(), item['answer'].lower()) for item in items}) >= target_total:
            break
        if args.sleep:
            time.sleep(args.sleep)

    write_outputs(items, chunks_by_id, args.train_target, args.test_target)
    if len({(item['question'].lower(), item['answer'].lower()) for item in items}) < target_total:
        print("WARNING: target QA count not reached.", file=sys.stderr)
        return 2
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
