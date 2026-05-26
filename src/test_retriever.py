import pickle
import sys

import faiss
from sentence_transformers import SentenceTransformer


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


def load_faiss_index():
    return faiss.read_index(INDEX_PATH)


def load_chunks():
    with open(CHUNKS_METADATA_PATH, "rb") as f:
        return pickle.load(f)


def retrieve(question, model, index, chunks, top_k=3):
    question_embedding = model.encode([question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(question_embedding)

    top_k = min(top_k, len(chunks))
    scores, indices = index.search(question_embedding, top_k)

    results = []
    for rank, idx in enumerate(indices[0], start=1):
        if idx < 0:
            continue
        chunk = chunks[int(idx)]
        results.append(
            {
                "rank": rank,
                "score": float(scores[0][rank - 1]),
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "title": chunk["title"],
                "url": chunk["url"],
                "text": chunk["text"],
            }
        )

    return results


def main():
    print("Loading FAISS index...")
    index = load_faiss_index()

    print("Loading chunk metadata...")
    chunks = load_chunks()
    print(f"Total chunks: {len(chunks)}")

    print(f"Loading model: {MODEL_NAME}")
    model = load_embedding_model(MODEL_NAME)

    question = "When was UET established?"
    top_k = 3

    print("\nQuestion:")
    print(question)

    results = retrieve(question, model, index, chunks, top_k=top_k)

    print(f"\nTop {top_k} retrieved chunks:")
    print("=" * 80)

    for item in results:
        print(f"\nTop {item['rank']} chunk")
        print(f"Chunk ID: {item['chunk_id']}")
        print(f"Document ID: {item['document_id']}")
        print(f"URL: {item['url']}")
        print(f"Title: {item['title']}")
        print(f"Score: {item['score']:.4f}")
        print("Text:")
        print(item["text"])
        print("-" * 80)


if __name__ == "__main__":
    main()
