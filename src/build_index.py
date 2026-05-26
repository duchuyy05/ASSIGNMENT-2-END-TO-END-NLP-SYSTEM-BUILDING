import json
import os
import pickle
import sys

import faiss
from sentence_transformers import SentenceTransformer

from src.config import CHUNKS_PATH, CHUNKS_PKL_PATH, EMBEDDING_MODEL, FAISS_INDEX_PATH


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


REQUIRED_CHUNK_FIELDS = {"chunk_id", "document_id", "title", "url", "text"}


def load_embedding_model(model_name: str) -> SentenceTransformer:
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        return SentenceTransformer(model_name)


def validate_chunk(chunk: dict) -> dict:
    missing = sorted(field for field in REQUIRED_CHUNK_FIELDS if not chunk.get(field))
    if missing:
        raise ValueError(f"Chunk is missing required fields {missing}: {chunk}")
    return chunk


def load_chunks() -> list[dict]:
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return [validate_chunk(chunk) for chunk in chunks if chunk.get("text", "").strip()]


def main() -> None:
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

    print("Loading chunks...")
    chunks = load_chunks()
    print(f"Total chunks: {len(chunks)}")

    texts = [chunk["text"] for chunk in chunks]

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = load_embedding_model(EMBEDDING_MODEL)

    print("Encoding chunks...")
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")

    dim = embeddings.shape[1]

    print("Building FAISS index...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"Saving index to: {FAISS_INDEX_PATH}")
    faiss.write_index(index, FAISS_INDEX_PATH)

    print(f"Saving chunk metadata to: {CHUNKS_PKL_PATH}")
    with open(CHUNKS_PKL_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print("Done.")


if __name__ == "__main__":
    main()
