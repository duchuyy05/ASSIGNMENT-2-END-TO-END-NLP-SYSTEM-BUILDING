import pickle
import sys

import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline


INDEX_PATH = "data/processed/faiss_index.bin"
CHUNKS_METADATA_PATH = "data/processed/chunks.pkl"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL = "distilbert-base-cased-distilled-squad"


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


def retrieve(question, embedding_model, index, chunks, top_k=3):
    question_embedding = embedding_model.encode([question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(question_embedding)

    top_k = min(top_k, len(chunks))
    scores, indices = index.search(question_embedding, top_k)

    retrieved_chunks = []
    for rank, idx in enumerate(indices[0], start=1):
        if idx < 0:
            continue
        chunk = chunks[int(idx)]
        retrieved_chunks.append(
            {
                "rank": rank,
                "score": float(scores[0][rank - 1]),
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "title": chunk["title"],
                "url": chunk["url"],
                "text": chunk["text"],
                "metadata": chunk,
            }
        )

    return retrieved_chunks


def build_context(retrieved_chunks):
    return "\n".join(chunk["text"] for chunk in retrieved_chunks)


def clean_answer(answer):
    answer = answer.strip().replace("\n", " ")
    return " ".join(answer.split())


def answer_question(question, embedding_model, index, chunks, qa_pipeline, top_k=3):
    retrieved_chunks = retrieve(
        question=question,
        embedding_model=embedding_model,
        index=index,
        chunks=chunks,
        top_k=top_k,
    )

    retrieved_context = build_context(retrieved_chunks)
    result = qa_pipeline(question=question, context=retrieved_context)
    final_answer = clean_answer(result["answer"])

    return final_answer, result, retrieved_chunks, retrieved_context


def main():
    print("Loading FAISS index...")
    index = load_faiss_index()

    print("Loading chunk metadata...")
    chunks = load_chunks()
    print(f"Total chunks: {len(chunks)}")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedding_model = load_embedding_model(EMBEDDING_MODEL)

    print(f"Loading QA reader: {QA_MODEL}")
    qa_pipeline = pipeline("question-answering", model=QA_MODEL)

    question = "When was UET established?"
    top_k = 3

    final_answer, result, retrieved_chunks, retrieved_context = answer_question(
        question=question,
        embedding_model=embedding_model,
        index=index,
        chunks=chunks,
        qa_pipeline=qa_pipeline,
        top_k=top_k,
    )

    print("\nQuestion:")
    print(question)

    print(f"\nTop {top_k} retrieved chunks:")
    print("=" * 80)

    for chunk in retrieved_chunks:
        print(f"\nTop {chunk['rank']} chunk")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Document ID: {chunk['document_id']}")
        print(f"URL: {chunk['url']}")
        print(f"Title: {chunk['title']}")
        print(f"Score: {chunk['score']:.4f}")
        print("Text:")
        print(chunk["text"])
        print("-" * 80)

    print("\nRetrieved context:")
    print(retrieved_context)

    print("\nQA result:")
    print(result)

    print("\nFinal answer:")
    print(final_answer)


if __name__ == "__main__":
    main()
