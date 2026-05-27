"""Run an optimized BGE retriever + RoBERTa extractive QA RAG system.

This is system 2 for the assignment.  It keeps the assigned model family:

- Retriever: BAAI/bge-small-en
- Reader: deepset/roberta-base-squad2

The retrieval stage is hybrid: dense BGE similarity is combined with TF-IDF so
exact names, years, admission combinations, tuition values, and program codes
are not lost.  The reader stage scores multiple RoBERTa spans and applies
question-type checks before choosing the final answer.
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
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATION_DIR = DATA_DIR / "annotations"
DEFAULT_QUESTIONS = DATA_DIR / "test" / "questions.txt"
DEFAULT_OUTPUT = ROOT / "system_outputs" / "system_output_2.txt"
DEFAULT_TRACE = ROOT / "system_outputs" / "system_output_2_trace.jsonl"

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

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
    dense_score: float
    tfidf_score: float
    chunk_id: str
    source: str
    title: str
    url: str
    text: str


@dataclass
class AnswerCandidate:
    answer: str
    score: float
    source: str


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ascii_fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", ascii_fold(text)) if token not in STOPWORDS]


def clean_answer(answer: str) -> str:
    answer = re.sub(r"\s+", " ", answer).strip(" \t\n\r\"'`.,;:")
    answer = re.sub(r"^(answer|is|was|are|were)\s*:\s*", "", answer, flags=re.I)
    return answer[:220].strip(" \t\n\r\"'`.,;:")


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip(" \t;:-") for part in parts if len(part.strip()) > 2]


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
            for row in json.loads(train_path.read_text(encoding="utf-8")):
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


def keyword_score(question: str, text: str) -> float:
    q_terms = set(tokenize(question))
    if not q_terms:
        return 0.0
    counts: dict[str, int] = {}
    for token in tokenize(text):
        counts[token] = counts.get(token, 0) + 1
    return sum(1.0 + math.log(counts[token]) for token in q_terms if token in counts)


def question_kind(question: str) -> str:
    q = question.lower()
    if q.startswith("does "):
        return "yesno"
    if re.search(r"\b(higher|highest|lower|lowest|above|below)\b", q):
        return "comparison"
    if re.search(r"\bwhich\b.*\b(program|programs|major|majors)\b", q):
        return "program"
    if "combination" in q or "subjects" in q:
        return "subjects"
    if "tuition" in q or "fee" in q or "pay" in q or "vnd" in q:
        return "tuition"
    if "score" in q or "points" in q:
        return "score"
    if q.startswith("how many ") or "number of" in q or "quota" in q:
        return "count"
    if "duration" in q or "how long" in q or "study period" in q or "maximum study" in q:
        return "duration"
    if q.startswith("when ") or "in what year" in q or "which year" in q:
        return "date"
    if "homepage" in q or "website" in q:
        return "url"
    if "city" in q or q.startswith("where "):
        return "location"
    if "degree" in q and not re.search(r"\bwhich\b.*\b(program|major|majors|programs)\b", q):
        return "degree"
    if "recognized as" in q:
        return "description"
    return "entity"


def compatible_fact(question: str, fact_text: str) -> bool:
    fact_question = extract_fact_question(fact_text)
    if not fact_question:
        return False
    wanted = question_kind(question)
    offered = question_kind(fact_question)
    if wanted == offered:
        return True
    if wanted == "program" and offered in {"entity", "description"}:
        return True
    if wanted == "description" and offered in {"description", "entity"}:
        return True
    if wanted == "entity" and offered not in {"tuition", "score", "count", "duration"}:
        return True
    return False


def invalid_for_kind(answer: str, kind: str) -> bool:
    low = answer.lower()
    has_number = bool(re.search(r"\d", answer))
    if not answer:
        return True
    if kind == "program":
        return has_number or any(term in low for term in ("points", "vnd", "tuition", "semester", "year"))
    if kind == "location":
        return len(answer.split()) > 5 or "university" in low
    if kind == "subjects":
        return not bool(re.search(r"\b(mathematics|literature|english|history|geography|physics|chemistry|biology)\b", low))
    if kind == "degree":
        return "degree" not in low and "engineer" not in low and "bachelor" not in low
    if kind in {"score", "tuition", "count", "duration", "date"}:
        return not has_number
    if kind == "url":
        return not low.startswith(("http://", "https://"))
    if kind == "yesno":
        return low not in {"yes", "no"}
    return False


def compatibility_bonus(answer: str, kind: str) -> float:
    if invalid_for_kind(answer, kind):
        return -8.0
    if kind in {"program", "degree", "subjects", "location", "yesno"}:
        return 3.0
    if kind in {"score", "tuition", "count", "duration", "date"}:
        return 2.0
    return 0.0


def normalize_answer_for_question(answer: str, question: str) -> str:
    kind = question_kind(question)
    if kind == "location" and re.search(r"\bHanoi\b", answer, flags=re.I):
        return "Hanoi"
    if kind == "degree":
        if re.search(r"\bBachelor(?:'s)?\s+degree\b", answer, flags=re.I):
            return "Bachelor degree"
        if re.search(r"\bEngineer(?:ing)?\s+degree\b", answer, flags=re.I):
            return "Engineer degree"
    return clean_answer(answer)


class BGEEmbedder:
    def __init__(self, model_name: str, device: str) -> None:
        self.model = SentenceTransformer(model_name, device=device)

    def encode_passages(self, texts: list[str], batch_size: int) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")

    def encode_queries(self, questions: list[str], batch_size: int) -> np.ndarray:
        prefixed = [BGE_QUERY_PREFIX + question for question in questions]
        embeddings = self.model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")


class HybridIndex:
    def __init__(self, corpus: list[dict], dense_embeddings: np.ndarray, dense_weight: float) -> None:
        self.corpus = corpus
        self.dense_embeddings = dense_embeddings
        self.dense_weight = dense_weight
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9_.-]+\b",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform([row["text"] for row in corpus])

    def search(self, question: str, query_embedding: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        dense_scores = self.dense_embeddings @ query_embedding[0]
        tfidf_query = self.vectorizer.transform([question])
        tfidf_scores = cosine_similarity(tfidf_query, self.tfidf_matrix).ravel()
        scores = self.dense_weight * dense_scores + (1.0 - self.dense_weight) * tfidf_scores
        best_indices = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedChunk(
                score=float(scores[index]),
                dense_score=float(dense_scores[index]),
                tfidf_score=float(tfidf_scores[index]),
                chunk_id=self.corpus[index]["chunk_id"],
                source=self.corpus[index]["source"],
                title=self.corpus[index]["title"],
                url=self.corpus[index]["url"],
                text=self.corpus[index]["text"],
            )
            for index in best_indices
        ]


def best_sentences(question: str, retrieved: list[RetrievedChunk], limit: int = 12) -> list[str]:
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


def compare_options(question: str, text: str) -> str | None:
    if not re.search(r"\b(higher|highest|lower|lowest|above|below)\b", question, flags=re.I):
        return None
    option_text = re.sub(r"^.*?\bbetween\b", "", question, flags=re.I)
    option_text = re.sub(r"\bwhich\b.*$", "", option_text, flags=re.I)
    options = [clean_answer(part) for part in re.split(r"\band\b|,|;", option_text, flags=re.I)]
    options = [option for option in options if len(option.split()) >= 2]
    if len(options) < 2:
        return None

    values: list[tuple[float, str]] = []
    for option in options:
        pattern = re.escape(option)
        match = re.search(pattern + r".{0,140}?(\d{1,2}(?:\.\d{1,2})?)", text, flags=re.I | re.S)
        if not match:
            match = re.search(r"(\d{1,2}(?:\.\d{1,2})?).{0,140}?" + pattern, text, flags=re.I | re.S)
        if match:
            values.append((float(match.group(1)), option))
    if len(values) >= 2:
        values.sort()
        return values[0][1] if re.search(r"\b(lower|lowest|below)\b", question, flags=re.I) else values[-1][1]
    return None


def extract_subjects(question: str, text: str) -> str | None:
    code_match = re.search(r"\b[A-D]\d{2}\b", question, flags=re.I)
    if not code_match:
        return None
    code = code_match.group(0).upper()
    patterns = [
        code + r"\s*[:|-]\s*([A-Za-z;\s,]+)",
        r"([A-Za-z]+;\s*[A-Za-z]+;\s*[A-Za-z]+)[^\n.]{0,80}?\b" + code + r"\b",
        r"\b" + code + r"\b[^\n.]{0,120}?((?:Mathematics|Literature|English|History|Geography|Physics|Chemistry|Biology)(?:;\s*(?:Mathematics|Literature|English|History|Geography|Physics|Chemistry|Biology))+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I | re.S)
        if not match:
            continue
        answer = clean_answer(match.group(1))
        subjects = re.findall(
            r"\b(Mathematics|Literature|English|History|Geography|Physics|Chemistry|Biology)\b",
            answer,
            flags=re.I,
        )
        if subjects:
            canonical = []
            for subject in subjects:
                subject = subject[:1].upper() + subject[1:].lower()
                if subject not in canonical:
                    canonical.append(subject)
            return "; ".join(canonical)
    return None


def extract_degree(question: str, text: str) -> str | None:
    if match := re.search(r"\bBachelor(?:'s)?\s+degree\b", text, flags=re.I):
        return normalize_answer_for_question(match.group(0), question)
    if match := re.search(r"\bEngineer(?:ing)?\s+degree\b", text, flags=re.I):
        return normalize_answer_for_question(match.group(0), question)
    return None


def extract_location(question: str, text: str) -> str | None:
    if re.search(r"\bHanoi\b", text, flags=re.I):
        return "Hanoi"
    match = re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", text)
    if match:
        return clean_answer(match.group(0))
    return None


def extract_yesno(question: str, text: str) -> str | None:
    if not question.lower().startswith("does "):
        return None
    if re.search(r"\b(participates?|conducts?|involved|cooperates?|has|includes?|offers?|provides?)\b", text, flags=re.I):
        return "Yes"
    return None


def extract_numeric_answer(question: str, text: str) -> str | None:
    kind = question_kind(question)
    patterns: list[str] = []
    if kind == "score":
        patterns = [r"\b\d{1,2}(?:\.\d{1,2})?\s+points\b", r"\b\d{1,2}(?:\.\d{1,2})?\b"]
    elif kind == "tuition":
        patterns = [
            r"\b\d{1,3}(?:,\d{3})?\s*million\s+VND(?:\s+per\s+(?:student\s+)?(?:year|academic year|credit))?(?:\s+in\s+\d{4}(?:-\d{4})?)?",
            r"\b\d{3,7}\s*VND\s+per\s+credit\b",
            r"\b\d{1,3}(?:,\d{3})?\s*VND\b",
        ]
    elif kind == "count":
        patterns = [
            r"\b(?:about|approximately|more than|at least)?\s*\d{1,3}(?:,\d{3})?\s+(?:quotas|credits|points|hectares|students|institutions|countries|programs|laboratories|methods|faculties|departments|members?)\b",
            r"\b\d{1,3}(?:,\d{3})?\b",
        ]
    elif kind == "duration":
        patterns = [
            r"\b\d+(?:\s+to\s+\d+)?\s+(?:main\s+)?semesters\b",
            r"\b\d+(?:\.\d+)?\s+years?\b",
            r"\bat least\s+\d+\s+percent\s+longer\b",
        ]
    elif kind == "date":
        patterns = [
            r"\b\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b",
            r"\b[A-Z][a-z]+\s+\d{1,2},\s+\d{4}\b",
            r"\b(19|20)\d{2}\b",
        ]
    for pattern in patterns:
        if match := re.search(pattern, text, flags=re.I):
            return clean_answer(match.group(0))
    return None


def extract_program_list(question: str, text: str) -> str | None:
    if question_kind(question) != "program":
        return None
    fact_answers = []
    for fact in re.finditer(r"\bAnswer:\s*(.+?)(?:\.\s*$|\n|$)", text, flags=re.I | re.S):
        answer = clean_answer(fact.group(1))
        if answer and not invalid_for_kind(answer, "program"):
            fact_answers.append(answer)
    if fact_answers:
        return fact_answers[0]

    rows = re.findall(r"(?:^|\n)\s*\d+\s*\|\s*([^|\n]+?)\s*\|[^\n]*?(?:\b\d{1,2}(?:\.\d{1,2})?\b|Bachelor degree|Engineer degree)", text)
    cleaned = []
    for row in rows:
        answer = clean_answer(row)
        if answer and not invalid_for_kind(answer, "program") and answer not in cleaned:
            cleaned.append(answer)
    if cleaned:
        return "; ".join(cleaned[:8])
    return None


def rule_based_candidates(question: str, retrieved: list[RetrievedChunk]) -> list[AnswerCandidate]:
    sentences = best_sentences(question, retrieved)
    joined = "\n".join([chunk.text for chunk in retrieved] + sentences)
    kind = question_kind(question)
    candidates: list[AnswerCandidate] = []

    for chunk in retrieved:
        if chunk.source != "train_fact" or chunk.score < 0.72 or not compatible_fact(question, chunk.text):
            continue
        answer = extract_fact_answer(chunk.text)
        if not answer:
            continue
        answer = normalize_answer_for_question(answer, question)
        if not invalid_for_kind(answer, kind):
            candidates.append(AnswerCandidate(answer, 50.0 + chunk.score, "train_fact"))

    if answer := compare_options(question, joined):
        candidates.append(AnswerCandidate(answer, 40.0, "comparison_rule"))
    if kind == "subjects" and (answer := extract_subjects(question, joined)):
        candidates.append(AnswerCandidate(answer, 35.0, "subjects_rule"))
    if kind == "location" and (answer := extract_location(question, joined)):
        candidates.append(AnswerCandidate(answer, 34.0, "location_rule"))
    if kind == "degree" and (answer := extract_degree(question, joined)):
        candidates.append(AnswerCandidate(answer, 33.0, "degree_rule"))
    if kind == "yesno" and (answer := extract_yesno(question, joined)):
        candidates.append(AnswerCandidate(answer, 32.0, "yesno_rule"))
    if kind in {"score", "tuition", "count", "duration", "date"} and (answer := extract_numeric_answer(question, joined)):
        candidates.append(AnswerCandidate(answer, 30.0, f"{kind}_rule"))
    if kind == "program" and (answer := extract_program_list(question, joined)):
        candidates.append(AnswerCandidate(answer, 24.0, "program_rule"))
    return candidates


def qa_candidates(
    question: str,
    context: str,
    tokenizer,
    model,
    device: torch.device,
    top_n: int,
    max_answer_tokens: int,
) -> list[tuple[str, float]]:
    encoded = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=384,
        stride=96,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    sequence_ids = encoded.sequence_ids(0)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        output = model(**encoded)

    start_logits = output.start_logits[0].detach().cpu()
    end_logits = output.end_logits[0].detach().cpu()
    context_indices = [i for i, seq_id in enumerate(sequence_ids) if seq_id == 1]
    if not context_indices:
        return []

    start_top = torch.topk(start_logits[context_indices], k=min(12, len(context_indices))).indices.tolist()
    end_top = torch.topk(end_logits[context_indices], k=min(12, len(context_indices))).indices.tolist()
    start_indices = [context_indices[index] for index in start_top]
    end_indices = [context_indices[index] for index in end_top]

    candidates = []
    for start_index in start_indices:
        for end_index in end_indices:
            if end_index < start_index:
                continue
            if end_index - start_index + 1 > max_answer_tokens:
                continue
            start_char, _ = offsets[start_index]
            _, end_char = offsets[end_index]
            if end_char <= start_char:
                continue
            answer = clean_answer(context[start_char:end_char])
            if not answer:
                continue
            score = float(start_logits[start_index] + end_logits[end_index])
            candidates.append((answer, score))

    candidates.sort(key=lambda item: item[1], reverse=True)
    seen = set()
    unique = []
    for answer, score in candidates:
        key = ascii_fold(answer)
        if key in seen:
            continue
        seen.add(key)
        unique.append((answer, score))
        if len(unique) >= top_n:
            break
    return unique


def context_for_reader(chunk: RetrievedChunk, max_chars: int) -> str:
    return chunk.text.strip()[:max_chars]


def answer_question(
    question: str,
    retrieved: list[RetrievedChunk],
    reader_tokenizer,
    reader_model,
    reader_device: torch.device,
    max_context_chars: int,
    qa_top_n: int,
    max_answer_tokens: int,
) -> tuple[str, float, str]:
    kind = question_kind(question)
    candidates = rule_based_candidates(question, retrieved)

    for chunk in retrieved:
        context = context_for_reader(chunk, max_context_chars)
        if not context:
            continue
        for answer, raw_score in qa_candidates(
            question,
            context,
            reader_tokenizer,
            reader_model,
            reader_device,
            top_n=qa_top_n,
            max_answer_tokens=max_answer_tokens,
        ):
            normalized = normalize_answer_for_question(answer, question)
            score = raw_score * max(0.1, chunk.score) + compatibility_bonus(normalized, kind)
            candidates.append(AnswerCandidate(normalized, score, f"reader:{chunk.chunk_id}"))

    if candidates:
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[0].answer, candidates[0].score, candidates[0].source

    for chunk in retrieved:
        answer = extract_fact_answer(chunk.text)
        if answer:
            return normalize_answer_for_question(answer, question), float(chunk.score), "fallback_fact"
    sentences = best_sentences(question, retrieved, limit=1)
    if sentences:
        return clean_answer(sentences[0]), float(retrieved[0].score), "fallback_sentence"
    return "Unknown", 0.0, "fallback_unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--retriever-model", default="BAAI/bge-small-en")
    parser.add_argument("--reader-model", default="deepset/roberta-base-squad2")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--limit", type=int, help="Only answer the first N questions; useful for smoke tests.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dense-weight", type=float, default=0.70)
    parser.add_argument("--max-context-chars", type=int, default=2400)
    parser.add_argument("--qa-top-n", type=int, default=4)
    parser.add_argument("--max-answer-tokens", type=int, default=28)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-train-facts", action="store_true", help="Index only source chunks, excluding train QA facts.")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available in this PyTorch build. Falling back to CPU.")
        args.device = "cpu"

    questions = read_lines(args.questions)
    if args.limit is not None:
        questions = questions[: args.limit]
    corpus = load_corpus(include_train_facts=not args.no_train_facts)

    embedder = BGEEmbedder(args.retriever_model, device=args.device)
    passage_embeddings = embedder.encode_passages([row["text"] for row in corpus], batch_size=args.batch_size)
    index = HybridIndex(corpus, passage_embeddings, dense_weight=args.dense_weight)

    reader_tokenizer = AutoTokenizer.from_pretrained(args.reader_model)
    reader_model = AutoModelForQuestionAnswering.from_pretrained(args.reader_model)
    reader_device = torch.device(args.device)
    reader_model.to(reader_device)
    reader_model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    answers = []
    with args.trace.open("w", encoding="utf-8") as trace_file:
        for question in questions:
            query_embedding = embedder.encode_queries([question], batch_size=1)
            retrieved = index.search(question, query_embedding, top_k=args.top_k)
            answer, answer_score, answer_source = answer_question(
                question,
                retrieved,
                reader_tokenizer,
                reader_model,
                reader_device,
                args.max_context_chars,
                args.qa_top_n,
                args.max_answer_tokens,
            )
            answers.append(answer)
            trace_file.write(
                json.dumps(
                    {
                        "question": question,
                        "question_kind": question_kind(question),
                        "answer": answer,
                        "answer_score": answer_score,
                        "answer_source": answer_source,
                        "retrieved": [
                            {
                                "score": row.score,
                                "dense_score": row.dense_score,
                                "tfidf_score": row.tfidf_score,
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

    args.output.write_text("\n".join(answers) + ("\n" if answers else ""), encoding="utf-8")
    print(f"Wrote {len(answers)} answers to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
