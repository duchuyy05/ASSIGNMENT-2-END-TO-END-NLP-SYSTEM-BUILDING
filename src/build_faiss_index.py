import json
import os
import pickle
import sys

import faiss
from sentence_transformers import SentenceTransformer


CHUNKS_JSON_PATH = "data/processed/chunks.json"
INDEX_PATH = "data/processed/faiss_index.bin"
CHUNKS_METADATA_PATH = "data/processed/chunks.pkl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_embedding_model(model_name):
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        return SentenceTransformer(model_name)


REQUIRED_CHUNK_FIELDS = {"chunk_id", "document_id", "title", "url", "text"}


def validate_chunk(item):
    missing = sorted(field for field in REQUIRED_CHUNK_FIELDS if not item.get(field))
    if missing:
        raise ValueError(f"Chunk is missing required fields {missing}: {item}")
    return item


def load_chunks_json(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return [validate_chunk(item) for item in rows]


def load_chunks():
    if os.path.exists(CHUNKS_JSON_PATH):
        print(f"Using chunks from {CHUNKS_JSON_PATH}")
        return load_chunks_json(CHUNKS_JSON_PATH)
    raise FileNotFoundError(f"Cannot find chunks file: {CHUNKS_JSON_PATH}")


def build_faiss_index():
    os.makedirs("data/processed", exist_ok=True)

    print("Loading chunks...")
    chunks = load_chunks()

    if not chunks:
        raise ValueError("No documents found in chunks file")

    texts = [chunk["text"] for chunk in chunks]
    print(f"Loaded {len(texts)} chunks.")

    print(f"Loading embedding model: {MODEL_NAME}")
    model = load_embedding_model(MODEL_NAME)

    print("Encoding chunks...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.astype("float32")

    # Normalize vectors so IndexFlatIP behaves like cosine similarity.
    faiss.normalize_L2(embeddings)

    embedding_dim = embeddings.shape[1]
    print(f"Embedding dimension: {embedding_dim}")

    index = faiss.IndexFlatIP(embedding_dim)

    print("Adding embeddings to FAISS index...")
    index.add(embeddings)
    print(f"Total vectors in index: {index.ntotal}")

    print(f"Saving FAISS index to {INDEX_PATH}")
    faiss.write_index(index, INDEX_PATH)

    print(f"Saving chunk metadata to {CHUNKS_METADATA_PATH}")
    with open(CHUNKS_METADATA_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print("Done.")


if __name__ == "__main__":
    build_faiss_index()
