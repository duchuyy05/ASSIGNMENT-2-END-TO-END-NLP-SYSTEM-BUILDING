"""Run an E5 retriever + RoBERTa extractive QA RAG system.

Default model choices:

- Retriever: intfloat/e5-small-v2
- Reader: deepset/roberta-base-squad2

The script uses FAISS if it is installed; otherwise it falls back to NumPy
cosine search over normalized E5 embeddings.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoModelForQuestionAnswering, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATION_DIR = DATA_DIR / "annotations"
DEFAULT_QUESTIONS = DATA_DIR / "test" / "questions.txt"
DEFAULT_OUTPUT = ROOT / "system_outputs" / "system_output_3.txt"
DEFAULT_TRACE = ROOT / "system_outputs" / "system_output_3_trace.jsonl"


@dataclass
class RetrievedChunk:
    score: float
    chunk_id: str
    source: str
    title: str
    url: str
    text: str


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def clean_answer(answer: str) -> str:
    answer = re.sub(r"\s+", " ", answer).strip(" \t\n\r\"'`.,;:")
    return answer[:220].strip(" \t\n\r\"'`.,;:")


def extract_fact_answer(text: str) -> str | None:
    match = re.search(r"\bAnswer:\s*(.+?)(?:\.\s*$|\n|$)", text, flags=re.I | re.S)
    if not match:
        return None
    answer = clean_answer(match.group(1))
    return answer or None


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


class E5Embedder:
    def __init__(self, model_name: str, device: str, max_length: int) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device(device)
        self.max_length = max_length
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: list[str], prefix: str, batch_size: int) -> np.ndarray:
        vectors = []
        prefixed = [f"{prefix}: {text}" for text in texts]
        with torch.no_grad():
            for start in range(0, len(prefixed), batch_size):
                batch = prefixed[start : start + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                output = self.model(**encoded)
                token_embeddings = output.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                summed = torch.sum(token_embeddings * mask, dim=1)
                counts = torch.clamp(mask.sum(dim=1), min=1e-9)
                pooled = summed / counts
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.append(pooled.cpu().numpy())
        return np.vstack(vectors).astype("float32")


class VectorIndex:
    def __init__(self, embeddings: np.ndarray) -> None:
        self.embeddings = embeddings
        self.faiss_index = None
        try:
            import faiss  # type: ignore

            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            self.faiss_index = index
        except Exception:
            self.faiss_index = None

    def search(self, query_embedding: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.faiss_index is not None:
            scores, indices = self.faiss_index.search(query_embedding, top_k)
            return scores[0], indices[0]
        scores = self.embeddings @ query_embedding[0]
        indices = np.argsort(scores)[::-1][:top_k]
        return scores[indices], indices


def context_for_reader(chunks: list[RetrievedChunk], max_chars: int) -> str:
    parts = []
    total = 0
    for chunk in chunks:
        text = chunk.text.strip()
        if not text:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        piece = text[:remaining]
        parts.append(piece)
        total += len(piece)
    return "\n\n".join(parts)


def qa_predict(question: str, context: str, tokenizer, model, device: torch.device) -> tuple[str, float]:
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
    best_score = float("-inf")
    best_answer = ""

    context_token_indices = [i for i, seq_id in enumerate(sequence_ids) if seq_id == 1]
    for start_index in context_token_indices:
        for end_index in context_token_indices:
            if end_index < start_index:
                continue
            if end_index - start_index + 1 > 24:
                continue
            start_char, _ = offsets[start_index]
            _, end_char = offsets[end_index]
            if end_char <= start_char:
                continue
            score = float(start_logits[start_index] + end_logits[end_index])
            if score > best_score:
                best_score = score
                best_answer = context[start_char:end_char]

    return clean_answer(best_answer), best_score


def answer_question(
    question: str,
    retrieved: list[RetrievedChunk],
    reader_tokenizer,
    reader_model,
    reader_device: torch.device,
    max_context_chars: int,
) -> tuple[str, float]:
    for chunk in retrieved:
        if chunk.source == "train_fact" and chunk.score >= 0.84:
            fact_answer = extract_fact_answer(chunk.text)
            if fact_answer:
                return fact_answer, float(chunk.score)

    candidates = []
    for chunk in retrieved:
        context = context_for_reader([chunk], max_context_chars)
        if not context:
            continue
        answer, raw_score = qa_predict(question, context, reader_tokenizer, reader_model, reader_device)
        score = raw_score * max(0.1, chunk.score)
        if answer:
            candidates.append((score, answer))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1], candidates[0][0]

    for chunk in retrieved:
        fact_answer = extract_fact_answer(chunk.text)
        if fact_answer:
            return fact_answer, float(chunk.score)
    return "Unknown", 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--retriever-model", default="intfloat/e5-small-v2")
    parser.add_argument("--reader-model", default="deepset/roberta-base-squad2")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--limit", type=int, help="Only answer the first N questions; useful for smoke tests.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-context-chars", type=int, default=2200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-train-facts", action="store_true", help="Index only source chunks, excluding train QA facts.")
    args = parser.parse_args()

    questions = read_lines(args.questions)
    if args.limit is not None:
        questions = questions[: args.limit]
    corpus = load_corpus(include_train_facts=not args.no_train_facts)

    embedder = E5Embedder(args.retriever_model, device=args.device, max_length=args.max_length)
    passage_embeddings = embedder.encode([row["text"] for row in corpus], prefix="passage", batch_size=args.batch_size)
    index = VectorIndex(passage_embeddings)

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
            query_embedding = embedder.encode([question], prefix="query", batch_size=1)
            scores, indices = index.search(query_embedding, top_k=args.top_k)
            retrieved = [
                RetrievedChunk(
                    score=float(score),
                    chunk_id=corpus[int(index_)]["chunk_id"],
                    source=corpus[int(index_)]["source"],
                    title=corpus[int(index_)]["title"],
                    url=corpus[int(index_)]["url"],
                    text=corpus[int(index_)]["text"],
                )
                for score, index_ in zip(scores, indices)
                if int(index_) >= 0
            ]
            answer, answer_score = answer_question(
                question,
                retrieved,
                reader_tokenizer,
                reader_model,
                reader_device,
                args.max_context_chars,
            )
            answers.append(answer)
            trace_file.write(
                json.dumps(
                    {
                        "question": question,
                        "answer": answer,
                        "answer_score": answer_score,
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

    args.output.write_text("\n".join(answers) + ("\n" if answers else ""), encoding="utf-8")
    print(f"Wrote {len(answers)} answers to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
