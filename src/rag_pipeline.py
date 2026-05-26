from typing import Any

from src.config import DEFAULT_TOP_K
from src.reader import DistilBertReader
from src.retriever import FaissRetriever


class RAGPipeline:
    def __init__(self):
        print("Loading retriever...")
        self.retriever = FaissRetriever()

        print("Loading reader...")
        self.reader = DistilBertReader()

        print("RAG pipeline loaded.")

    def build_context(self, chunks: list[dict[str, Any]]) -> str:
        parts = []

        for chunk in chunks:
            title = chunk.get("title") or ""
            text = chunk.get("text") or ""
            parts.append(f"Title: {title}\nContent: {text}")

        return "\n\n".join(parts)

    def answer_question(self, question: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
        retrieved_chunks = self.retriever.retrieve(question, top_k=top_k)
        context = self.build_context(retrieved_chunks)
        reader_result = self.reader.answer(question, context)

        return {
            "question": question,
            "answer": reader_result["answer"],
            "confidence": reader_result["score"],
            "sources": retrieved_chunks,
            "context": context,
        }
