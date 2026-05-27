"""Run a lightweight RAG baseline over the assignment corpus.

The pipeline has the three required RAG parts:

1. document/query embedder: scikit-learn TF-IDF vectors
2. document retriever: cosine similarity over chunk vectors
3. document reader: extractive answer heuristics over retrieved chunks

It is intentionally dependency-light and runs without downloading models.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATION_DIR = DATA_DIR / "annotations"
DEFAULT_QUESTIONS = DATA_DIR / "test" / "questions.txt"
DEFAULT_OUTPUT = ROOT / "system_outputs" / "system_output_1.txt"


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


@dataclass
class RetrievedChunk:
    score: float
    text: str
    chunk_id: str
    source: str
    title: str
    url: str


def ascii_fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", ascii_fold(text)) if token not in STOPWORDS]


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_corpus(include_train_facts: bool) -> list[dict]:
    chunks = json.loads((PROCESSED_DIR / "chunks.json").read_text(encoding="utf-8"))
    corpus = []
    for chunk in chunks:
        corpus.append(
            {
                "chunk_id": chunk["chunk_id"],
                "source": "corpus_chunk",
                "title": chunk.get("title", ""),
                "url": chunk.get("url", ""),
                "text": f"{chunk.get('title', '')}\n{chunk['text']}",
            }
        )

    if include_train_facts:
        train_path = ANNOTATION_DIR / "train_qa.json"
        if train_path.exists():
            train_rows = json.loads(train_path.read_text(encoding="utf-8"))
            for row in train_rows:
                answer = row.get("answer") or "; ".join(row.get("answers", []))
                corpus.append(
                    {
                        "chunk_id": row["id"],
                        "source": "train_fact",
                        "title": "Manual training QA fact",
                        "url": row.get("source_url", ""),
                        "text": f"Question: {row['question']} Answer: {answer}.",
                    }
                )
    return corpus


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip(" \t;:-") for part in parts if len(part.strip()) > 2]


def extract_fact_answer(text: str) -> str | None:
    match = re.search(r"\bAnswer:\s*(.+?)(?:\.\s*$|\n|$)", text, flags=re.I | re.S)
    if not match:
        return None
    answer = clean_answer(match.group(1))
    return answer or None


def extract_fact_question(text: str) -> str | None:
    match = re.search(r"\bQuestion:\s*(.+?)\s+Answer:\s*", text, flags=re.I | re.S)
    if not match:
        return None
    question = clean_answer(match.group(1))
    return question or None


def clean_answer(answer: str) -> str:
    answer = re.sub(r"\s+", " ", answer).strip(" \t\n\r\"'`.,;:")
    answer = re.sub(r"^(answer|is|was|are|were)\s*:\s*", "", answer, flags=re.I)
    return answer[:220].strip(" \t\n\r\"'`.,;:")


def question_kind(question: str) -> str:
    q = question.lower()
    if q.startswith("does "):
        return "yesno"
    if "tuition" in q or "fee" in q or "pay" in q:
        return "tuition"
    if "score" in q or "points" in q:
        return "score"
    if "quota" in q:
        return "quota"
    if "duration" in q or "how long" in q or "study period" in q or "maximum study" in q:
        return "duration"
    if q.startswith("when ") or "in what year" in q or "which year" in q:
        return "date"
    if "combination" in q or "subjects" in q:
        return "combination"
    if "degree" in q:
        return "degree"
    if "homepage" in q or "website" in q:
        return "url"
    if "city" in q or q.startswith("where "):
        return "location"
    if "abbreviation" in q or "stand for" in q:
        return "abbreviation"
    if q.startswith("how many "):
        return "count"
    return "entity"


def compatible_fact(question: str, fact_text: str) -> bool:
    fact_question = extract_fact_question(fact_text)
    if not fact_question:
        return False
    wanted = question_kind(question)
    offered = question_kind(fact_question)
    if wanted == offered:
        return True
    if wanted == "count" and offered in {"quota", "count"}:
        return True
    if wanted == "entity" and offered not in {"tuition", "score", "quota", "duration"}:
        return True
    return False


def keyword_score(question: str, text: str) -> float:
    q_terms = set(tokenize(question))
    if not q_terms:
        return 0.0
    t_terms = tokenize(text)
    if not t_terms:
        return 0.0
    counts = {}
    for token in t_terms:
        counts[token] = counts.get(token, 0) + 1
    return sum(1.0 + math.log(counts[token]) for token in q_terms if token in counts)


def best_sentences(question: str, retrieved: list[RetrievedChunk], limit: int = 8) -> list[str]:
    scored = []
    for chunk in retrieved:
        for sentence in split_sentences(chunk.text):
            score = keyword_score(question, sentence) + chunk.score
            if score > 0:
                scored.append((score, sentence))
    scored.sort(key=lambda item: item[0], reverse=True)
    seen = set()
    result = []
    for _, sentence in scored:
        key = ascii_fold(sentence)
        if key in seen:
            continue
        seen.add(key)
        result.append(sentence)
        if len(result) >= limit:
            break
    return result


def compare_options(question: str, sentences: list[str]) -> str | None:
    if not re.search(r"\b(higher|highest|lower|lowest|above|below)\b", question, flags=re.I):
        return None
    option_text = re.sub(r"^.*?\bbetween\b", "", question, flags=re.I)
    option_text = re.sub(r"\bwhich\b.*$", "", option_text, flags=re.I)
    options = [clean_answer(part) for part in re.split(r"\band\b|,|;", option_text, flags=re.I)]
    options = [option for option in options if len(option.split()) >= 2]
    if len(options) < 2:
        return None

    joined = " ".join(sentences)
    values: list[tuple[float, str]] = []
    for option in options:
        pattern = re.escape(option)
        match = re.search(pattern + r".{0,80}?(\d{1,2}(?:\.\d{1,2})?)", joined, flags=re.I)
        if not match:
            match = re.search(r"(\d{1,2}(?:\.\d{1,2})?).{0,80}?" + pattern, joined, flags=re.I)
        if match:
            values.append((float(match.group(1)), option))
    if len(values) >= 2:
        values.sort()
        return values[0][1] if re.search(r"\b(lower|lowest|below)\b", question, flags=re.I) else values[-1][1]
    return None


def regex_answer(question: str, sentences: list[str]) -> str | None:
    q = question.lower()
    text = " ".join(sentences)

    if q.startswith("does "):
        if re.search(r"\b(participates?|involved|conducts?|cooperates?|has|includes?)\b", text, flags=re.I):
            return "Yes"

    patterns: list[str] = []
    if "homepage" in q or "website" in q:
        patterns = [r"https?://[^\s,;.)]+"]
    elif q.startswith("when ") or "in what year" in q or "which year" in q:
        patterns = [
            r"\b\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b",
            r"\b[A-Z][a-z]+\s+\d{1,2},\s+\d{4}\b",
            r"\b(19|20)\d{2}\b",
        ]
    elif q.startswith("how many ") or "number of" in q:
        patterns = [
            r"\b(?:about|approximately|more than|at least)?\s*\d{1,3}(?:,\d{3})?\s+(?:quotas|credits|points|hectares|students|institutions|countries|programs|laboratories|methods|faculties|departments|members?)\b",
            r"\b\d{1,3}(?:,\d{3})?\b",
        ]
    elif "tuition" in q or "pay" in q or "fee" in q:
        patterns = [
            r"\b\d{1,3}(?:,\d{3})?\s*million\s+VND(?:\s+per\s+(?:student\s+)?(?:year|academic year|credit))?(?:\s+in\s+\d{4}(?:-\d{4})?)?",
            r"\b\d{3,7}\s*VND\s+per\s+credit\b",
        ]
    elif "score" in q or "points" in q:
        patterns = [r"\b\d{1,2}(?:\.\d{1,2})?\s+points\b", r"\b\d{1,2}(?:\.\d{1,2})?\b"]
    elif "duration" in q or "how long" in q or "maximum study" in q or "study period" in q:
        patterns = [
            r"\b\d+(?:\s+to\s+\d+)?\s+(?:main\s+)?semesters\b",
            r"\b\d+(?:\.\d+)?\s+years?\b",
            r"\bat least\s+\d+\s+percent\s+longer\b",
        ]
    elif "combination" in q or "subjects" in q:
        patterns = [
            r"\b[A-Z]\d{2}\b",
            r"\bMathematics;\s*[^.]+",
            r"\bLiterature;\s*[^.]+",
        ]
    elif "certificate" in q:
        patterns = [r"\bIELTS(?:\s+Academic)?\b", r"\bTOEFL\b", r"\bSAT\b"]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return clean_answer(match.group(0))
    return None


def capitalized_answer(question: str, sentences: list[str]) -> str | None:
    q_terms = set(tokenize(question))
    candidates = []
    for sentence in sentences:
        for match in re.finditer(
            r"\b[A-Z][A-Za-z&-]*(?:\s+(?:and|of|for|in|the|[A-Z][A-Za-z&-]*|\d{2}/CP))*",
            sentence,
        ):
            candidate = clean_answer(match.group(0))
            words = candidate.split()
            if not candidate or len(words) > 12:
                continue
            if candidate.lower() in {"question", "answer", "manual training qa fact"}:
                continue
            c_terms = set(tokenize(candidate))
            novelty = len(c_terms - q_terms)
            if novelty <= 0:
                continue
            score = novelty + len(words) * 0.05
            candidates.append((score, candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def answer_question(question: str, retrieved: list[RetrievedChunk]) -> str:
    for chunk in retrieved:
        if chunk.source != "train_fact" or chunk.score < 0.25:
            continue
        if not compatible_fact(question, chunk.text):
            continue
        fact_answer = extract_fact_answer(chunk.text)
        if fact_answer:
            return fact_answer

    sentences = best_sentences(question, retrieved)
    for strategy in (compare_options, regex_answer):
        answer = strategy(question, sentences)
        if answer:
            return answer

    for chunk in retrieved:
        if chunk.source == "train_fact" and not compatible_fact(question, chunk.text):
            continue
        fact_answer = extract_fact_answer(chunk.text)
        if fact_answer:
            return fact_answer

    answer = capitalized_answer(question, sentences)
    if answer:
        return answer
    return clean_answer(sentences[0]) if sentences else "Unknown"


class TfidfRag:
    def __init__(self, corpus: list[dict]) -> None:
        self.corpus = corpus
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9_.-]+\b",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform([row["text"] for row in corpus])

    def retrieve(self, question: str, top_k: int) -> list[RetrievedChunk]:
        query = self.vectorizer.transform([question])
        scores = cosine_similarity(query, self.matrix).ravel()
        if not np.any(scores):
            best_indices = np.arange(min(top_k, len(self.corpus)))
        else:
            best_indices = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedChunk(
                score=float(scores[index]),
                text=self.corpus[index]["text"],
                chunk_id=self.corpus[index]["chunk_id"],
                source=self.corpus[index]["source"],
                title=self.corpus[index]["title"],
                url=self.corpus[index]["url"],
            )
            for index in best_indices
        ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--no-train-facts", action="store_true", help="Index only corpus chunks, excluding train QA facts.")
    parser.add_argument("--trace", type=Path, help="Optional JSONL file with retrieved chunk metadata.")
    args = parser.parse_args()

    questions = read_lines(args.questions)
    corpus = load_corpus(include_train_facts=not args.no_train_facts)
    rag = TfidfRag(corpus)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace_file = None
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        trace_file = args.trace.open("w", encoding="utf-8")

    answers = []
    try:
        for question in questions:
            retrieved = rag.retrieve(question, top_k=args.top_k)
            answer = answer_question(question, retrieved)
            answers.append(answer)
            if trace_file:
                trace_file.write(
                    json.dumps(
                        {
                            "question": question,
                            "answer": answer,
                            "retrieved": [
                                {
                                    "score": row.score,
                                    "chunk_id": row.chunk_id,
                                    "source": row.source,
                                    "title": row.title,
                                    "url": row.url,
                                }
                                for row in retrieved
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    finally:
        if trace_file:
            trace_file.close()

    args.output.write_text("\n".join(answers) + ("\n" if answers else ""), encoding="utf-8")
    print(f"Wrote {len(answers)} answers to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
