import json
import os


DOCUMENTS_PATH = "data/processed/documents.json"
OUTPUT_PATH = "data/processed/chunks.json"

CHUNK_SIZE = 140
CHUNK_OVERLAP = 30


def split_long_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [(0, len(words), " ".join(words))]

    chunks = []
    step = max(1, chunk_size - overlap)
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append((start, end, " ".join(words[start:end])))
        if end >= len(words):
            break
        start += step
    return chunks


def load_documents():
    with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def make_chunks(documents, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    all_chunks = []

    for document in documents:
        document_id = document["id"]
        text = document.get("text", "")

        for chunk_index, (start_word, end_word, chunk_text) in enumerate(
            split_long_text(text, chunk_size=chunk_size, overlap=overlap)
        ):
            all_chunks.append(
                {
                    "chunk_id": f"{document_id}_chunk_{chunk_index:04d}",
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "title": document.get("title", ""),
                    "url": document.get("url", ""),
                    "category": document.get("category", "general"),
                    "source_language": document.get("source_language", ""),
                    "qa_language": document.get("qa_language", "en"),
                    "start_word": start_word,
                    "end_word": end_word,
                    "word_count": len(chunk_text.split()),
                    "chunk_source": "source_text",
                    "text": chunk_text,
                }
            )

    return all_chunks


def process_documents():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    documents = load_documents()
    chunks = make_chunks(documents)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Done. Created {len(chunks)} chunks.")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    process_documents()
