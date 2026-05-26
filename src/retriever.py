import pickle
from typing import Any

import faiss
from sentence_transformers import SentenceTransformer

from src.config import CHUNKS_PKL_PATH, DEFAULT_TOP_K, EMBEDDING_MODEL, FAISS_INDEX_PATH


def load_embedding_model(model_name: str) -> SentenceTransformer:
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        return SentenceTransformer(model_name)


class FaissRetriever:
    def __init__(
        self,
        index_path: str = FAISS_INDEX_PATH,
        chunks_path: str = CHUNKS_PKL_PATH,
        embedding_model_name: str = EMBEDDING_MODEL,
    ):
        self.index = faiss.read_index(index_path)

        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        self.embedding_model = load_embedding_model(embedding_model_name)

    def retrieve(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        top_k = max(1, min(top_k, len(self.chunks)))
        question_embedding = self.embedding_model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(question_embedding, top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self.chunks):
                continue

            chunk = self.chunks[int(idx)]
            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "chunk_id": chunk.get("chunk_id"),
                    "document_id": chunk.get("document_id"),
                    "chunk_index": chunk.get("chunk_index"),
                    "title": chunk.get("title"),
                    "url": chunk.get("url"),
                    "category": chunk.get("category"),
                    "source_language": chunk.get("source_language"),
                    "qa_language": chunk.get("qa_language"),
                    "word_count": chunk.get("word_count"),
                    "text": chunk.get("text", ""),
                }
            )

        return results
