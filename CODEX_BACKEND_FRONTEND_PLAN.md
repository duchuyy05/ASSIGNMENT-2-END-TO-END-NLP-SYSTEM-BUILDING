# Codex Task: Add Backend + Frontend to Assignment 2 RAG Project

## Project Context

This project is for **ASSIGNMENT 2: END-TO-END NLP SYSTEM BUILDING**.

The core system is a RAG QA system for VNU/UET factual question answering.

Current pipeline for Model 1:

```text
Question
→ sentence-transformers/all-MiniLM-L6-v2 embedding
→ FAISS retrieve
→ distilbert-base-cased-distilled-squad QA reader
→ answer
→ system_outputs/system_output_1.txt
```

The project already has real dataset files, especially:

```text
data/processed/chunks.json
```

`chunks.json` is the real knowledge base. Each item is a chunk object with fields like:

```json
{
  "chunk_id": "doc_0000_chunk_0000",
  "document_id": "doc_0000",
  "chunk_index": 0,
  "title": "...",
  "url": "...",
  "category": "admission",
  "source_language": "vi",
  "qa_language": "en",
  "word_count": 133,
  "text": "..."
}
```

The goal is to add:

1. A clean RAG core inside `src/`
2. A FastAPI backend inside `backend/`
3. A simple HTML/CSS/JS frontend inside `frontend/`
4. Keep `src/run_inference.py` working for assignment submission

---

## Existing Project Structure

Current structure:

```text
ASSIGNMENT-2-END-TO-END-NLP-SYSTEM-BUILDING/
├── .env.example
├── .gitignore
├── README.md
├── data/
│   ├── manual_annotations/
│   │   ├── train_qa.csv
│   │   └── test_qa.csv
│   ├── annotations/
│   │   ├── train_qa.json
│   │   └── test_qa.json
│   ├── processed/
│   │   ├── documents.json
│   │   ├── chunks.json
│   │   └── dataset_metadata.json
│   ├── train/
│   │   ├── questions.txt
│   │   └── reference_answers.txt
│   └── test/
│       ├── questions.txt
│       └── reference_answers.txt
├── scripts/
│   ├── build_manual_dataset.py
│   ├── build_vnu_data.py
│   ├── generate_qa_openrouter.py
│   └── refine_qa_openrouter.py
└── src/
```

Meaning:

```text
data/manual_annotations/  original manually-created QA data
data/annotations/         QA annotation in JSON format
data/processed/           corpus, chunks, FAISS index, metadata
/data/train/              assignment train questions/answers format
data/test/                assignment test questions/answers format
scripts/                  dataset build/refine/generate scripts
src/                      retrieval, FAISS, inference RAG, reader code
```

---

## Target Project Structure

Update project to this structure:

```text
ASSIGNMENT-2-END-TO-END-NLP-SYSTEM-BUILDING/
├── .env.example
├── .env
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/
│   ├── manual_annotations/
│   │   ├── train_qa.csv
│   │   └── test_qa.csv
│   ├── annotations/
│   │   ├── train_qa.json
│   │   └── test_qa.json
│   ├── processed/
│   │   ├── documents.json
│   │   ├── chunks.json
│   │   ├── dataset_metadata.json
│   │   ├── faiss_index.bin
│   │   └── documents.pkl
│   ├── train/
│   │   ├── questions.txt
│   │   └── reference_answers.txt
│   └── test/
│       ├── questions.txt
│       └── reference_answers.txt
│
├── scripts/
│   ├── build_manual_dataset.py
│   ├── build_vnu_data.py
│   ├── generate_qa_openrouter.py
│   └── refine_qa_openrouter.py
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── build_index.py
│   ├── retriever.py
│   ├── reader.py
│   ├── rag_pipeline.py
│   └── run_inference.py
│
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── service.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
└── system_outputs/
    └── system_output_1.txt
```

---

## Important Constraint

Do **not** replace the main assigned Model 1 with OpenRouter.

OpenRouter is only for dataset generation/refinement scripts:

```text
scripts/generate_qa_openrouter.py
scripts/refine_qa_openrouter.py
```

The main RAG Model 1 must remain local:

```text
MiniLM + FAISS + DistilBERT QA
```

Optional future extension: add `/ask-llm` using OpenRouter, but do not implement it unless specifically requested.

---

## Step 1: Update `.env.example`

Add or update `.env.example` with:

```env
# OpenRouter - only for QA dataset generation/refinement scripts
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=meta-llama/llama-3.1-70b-instruct
OPENROUTER_SITE_URL=http://localhost
OPENROUTER_APP_NAME=vnu-rag-assignment

# Local RAG Model 1
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
READER_MODEL=distilbert-base-cased-distilled-squad

CHUNKS_PATH=data/processed/chunks.json
FAISS_INDEX_PATH=data/processed/faiss_index.bin
DOCUMENTS_PKL_PATH=data/processed/documents.pkl

DEFAULT_TOP_K=5

# Backend
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000
```

Make sure `.env` is ignored in `.gitignore`.

---

## Step 2: Update `requirements.txt`

Create or update `requirements.txt`:

```txt
fastapi
uvicorn
python-dotenv
sentence-transformers
transformers
torch
faiss-cpu
numpy
pydantic
```

---

## Step 3: Create `src/__init__.py`

Create an empty file:

```python
# src/__init__.py
```

---

## Step 4: Create `src/config.py`

```python
import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

READER_MODEL = os.getenv(
    "READER_MODEL",
    "distilbert-base-cased-distilled-squad",
)

CHUNKS_PATH = os.getenv(
    "CHUNKS_PATH",
    "data/processed/chunks.json",
)

FAISS_INDEX_PATH = os.getenv(
    "FAISS_INDEX_PATH",
    "data/processed/faiss_index.bin",
)

DOCUMENTS_PKL_PATH = os.getenv(
    "DOCUMENTS_PKL_PATH",
    "data/processed/documents.pkl",
)

DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
```

---

## Step 5: Create `src/build_index.py`

This script reads `data/processed/chunks.json`, embeds chunks with MiniLM, builds FAISS index, and saves:

```text
data/processed/faiss_index.bin
data/processed/documents.pkl
```

```python
import json
import os
import pickle

import faiss
from sentence_transformers import SentenceTransformer

from src.config import (
    CHUNKS_PATH,
    FAISS_INDEX_PATH,
    DOCUMENTS_PKL_PATH,
    EMBEDDING_MODEL,
)


def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    valid_chunks = []

    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if text:
            valid_chunks.append(chunk)

    return valid_chunks


def main():
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

    print("Loading chunks...")
    documents = load_chunks()
    print(f"Total chunks: {len(documents)}")

    texts = [doc["text"] for doc in documents]

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

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

    print("Saving index...")
    faiss.write_index(index, FAISS_INDEX_PATH)

    print("Saving documents...")
    with open(DOCUMENTS_PKL_PATH, "wb") as f:
        pickle.dump(documents, f)

    print("Done.")
    print(f"Index saved to: {FAISS_INDEX_PATH}")
    print(f"Documents saved to: {DOCUMENTS_PKL_PATH}")


if __name__ == "__main__":
    main()
```

Run:

```bash
python -m src.build_index
```

---

## Step 6: Create `src/retriever.py`

```python
import pickle
from typing import Any

import faiss
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    DOCUMENTS_PKL_PATH,
    DEFAULT_TOP_K,
)


class FaissRetriever:
    def __init__(
        self,
        index_path: str = FAISS_INDEX_PATH,
        documents_path: str = DOCUMENTS_PKL_PATH,
        embedding_model_name: str = EMBEDDING_MODEL,
    ):
        self.index = faiss.read_index(index_path)

        with open(documents_path, "rb") as f:
            self.documents = pickle.load(f)

        self.embedding_model = SentenceTransformer(embedding_model_name)

    def retrieve(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        question_embedding = self.embedding_model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(question_embedding, top_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue

            doc = self.documents[idx]

            results.append(
                {
                    "score": float(score),
                    "chunk_id": doc.get("chunk_id"),
                    "document_id": doc.get("document_id"),
                    "title": doc.get("title"),
                    "url": doc.get("url"),
                    "category": doc.get("category"),
                    "text": doc.get("text", ""),
                }
            )

        return results
```

---

## Step 7: Create `src/reader.py`

```python
from transformers import pipeline

from src.config import READER_MODEL


class DistilBertReader:
    def __init__(self, model_name: str = READER_MODEL):
        self.qa_pipeline = pipeline(
            "question-answering",
            model=model_name,
            tokenizer=model_name,
        )

    def answer(self, question: str, context: str) -> dict:
        if not context.strip():
            return {
                "answer": "",
                "score": 0.0,
            }

        result = self.qa_pipeline(
            question=question,
            context=context,
        )

        return {
            "answer": result.get("answer", "").strip(),
            "score": float(result.get("score", 0.0)),
        }
```

---

## Step 8: Create `src/rag_pipeline.py`

```python
from typing import Any

from src.config import DEFAULT_TOP_K
from src.retriever import FaissRetriever
from src.reader import DistilBertReader


class RAGPipeline:
    def __init__(self):
        print("Loading retriever...")
        self.retriever = FaissRetriever()

        print("Loading reader...")
        self.reader = DistilBertReader()

        print("RAG pipeline loaded.")

    def build_context(self, docs: list[dict[str, Any]]) -> str:
        parts = []

        for doc in docs:
            title = doc.get("title") or ""
            text = doc.get("text") or ""

            parts.append(
                f"Title: {title}\nContent: {text}"
            )

        return "\n\n".join(parts)

    def answer_question(self, question: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
        retrieved_docs = self.retriever.retrieve(question, top_k=top_k)
        context = self.build_context(retrieved_docs)

        reader_result = self.reader.answer(question, context)

        return {
            "question": question,
            "answer": reader_result["answer"],
            "confidence": reader_result["score"],
            "sources": retrieved_docs,
            "context": context,
        }
```

---

## Step 9: Update `src/run_inference.py`

This file is required for assignment submission.

It must read:

```text
data/test/questions.txt
```

And write:

```text
system_outputs/system_output_1.txt
```

```python
import os

from src.rag_pipeline import RAGPipeline


QUESTIONS_PATH = "data/test/questions.txt"
OUTPUT_PATH = "system_outputs/system_output_1.txt"


def main():
    os.makedirs("system_outputs", exist_ok=True)

    rag = RAGPipeline()

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for question in questions:
            result = rag.answer_question(question)
            answer = result["answer"]

            f.write(answer + "\n")

            print(f"Q: {question}")
            print(f"A: {answer}")
            print("-" * 50)

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
```

Run:

```bash
python -m src.run_inference
```

---

## Step 10: Create Backend

### 10.1 Create `backend/__init__.py`

```python
# backend/__init__.py
```

### 10.2 Create `backend/schemas.py`

```python
from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class SourceDocument(BaseModel):
    score: float
    chunk_id: str | None = None
    document_id: str | None = None
    title: str | None = None
    url: str | None = None
    category: str | None = None
    text: str | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    sources: list[SourceDocument]
```

### 10.3 Create `backend/service.py`

```python
from src.rag_pipeline import RAGPipeline


class RAGService:
    def __init__(self):
        self.pipeline = RAGPipeline()

    def ask(self, question: str, top_k: int = 5):
        result = self.pipeline.answer_question(
            question=question,
            top_k=top_k,
        )

        return {
            "question": result["question"],
            "answer": result["answer"],
            "confidence": result["confidence"],
            "sources": result["sources"],
        }

    def retrieve(self, question: str, top_k: int = 5):
        docs = self.pipeline.retriever.retrieve(
            question=question,
            top_k=top_k,
        )

        return {
            "question": question,
            "sources": docs,
        }
```

### 10.4 Create `backend/main.py`

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import AskRequest, AskResponse
from backend.service import RAGService


app = FastAPI(
    title="VNU/UET RAG Assignment Backend",
    description="MiniLM + FAISS + DistilBERT QA API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_service: RAGService | None = None


@app.on_event("startup")
def startup_event():
    global rag_service
    rag_service = RAGService()


@app.get("/")
def root():
    return {
        "message": "VNU/UET RAG backend is running",
        "pipeline": "Question -> MiniLM -> FAISS -> DistilBERT QA -> Answer",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    return rag_service.ask(
        question=request.question,
        top_k=request.top_k,
    )


@app.post("/retrieve")
def retrieve(request: AskRequest):
    return rag_service.retrieve(
        question=request.question,
        top_k=request.top_k,
    )
```

Run backend:

```bash
uvicorn backend.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Test request body:

```json
{
  "question": "When was UET established?",
  "top_k": 5
}
```

---

## Step 11: Create Frontend

Use simple HTML/CSS/JS. No React required.

### 11.1 Create `frontend/index.html`

```html
<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8" />
  <title>VNU/UET RAG QA System</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>
  <div class="app">
    <header>
      <h1>VNU/UET RAG QA System</h1>
      <p>MiniLM + FAISS + DistilBERT QA</p>
    </header>

    <main>
      <section class="card">
        <label for="question">Nhập câu hỏi</label>
        <textarea
          id="question"
          placeholder="Ví dụ: When was UET established?"
        ></textarea>

        <div class="controls">
          <label>
            Top K:
            <input id="topK" type="number" value="5" min="1" max="10" />
          </label>

          <button id="askBtn">Ask</button>
        </div>
      </section>

      <section class="card">
        <h2>Answer</h2>
        <div id="answer" class="answer-box">Chưa có câu trả lời.</div>
        <div id="confidence"></div>
      </section>

      <section class="card">
        <h2>Retrieved Sources</h2>
        <div id="sources"></div>
      </section>
    </main>
  </div>

  <script src="app.js"></script>
</body>
</html>
```

### 11.2 Create `frontend/style.css`

```css
* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: Arial, sans-serif;
  background: #f4f6f8;
  color: #111827;
}

.app {
  max-width: 900px;
  margin: 0 auto;
  padding: 32px 16px;
}

header {
  text-align: center;
  margin-bottom: 24px;
}

header h1 {
  margin-bottom: 8px;
}

header p {
  color: #6b7280;
}

.card {
  background: white;
  padding: 20px;
  border-radius: 14px;
  margin-bottom: 18px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
}

label {
  font-weight: 600;
  display: block;
  margin-bottom: 8px;
}

textarea {
  width: 100%;
  min-height: 110px;
  padding: 12px;
  border: 1px solid #d1d5db;
  border-radius: 10px;
  resize: vertical;
  font-size: 16px;
}

.controls {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-top: 12px;
}

.controls label {
  margin: 0;
}

input {
  width: 70px;
  padding: 8px;
  margin-left: 6px;
}

button {
  padding: 10px 18px;
  border: none;
  border-radius: 10px;
  background: #2563eb;
  color: white;
  cursor: pointer;
  font-weight: 600;
}

button:hover {
  background: #1d4ed8;
}

button:disabled {
  background: #9ca3af;
  cursor: not-allowed;
}

.answer-box {
  padding: 14px;
  background: #f9fafb;
  border-radius: 10px;
  font-size: 18px;
  font-weight: 600;
}

#confidence {
  margin-top: 8px;
  color: #6b7280;
}

.source-item {
  border-top: 1px solid #e5e7eb;
  padding: 14px 0;
}

.source-title {
  font-weight: 700;
}

.source-url {
  font-size: 14px;
  color: #2563eb;
  word-break: break-all;
}

.source-text {
  margin-top: 8px;
  color: #374151;
  line-height: 1.5;
}

.score {
  font-size: 13px;
  color: #6b7280;
}
```

### 11.3 Create `frontend/app.js`

```javascript
const API_URL = "http://127.0.0.1:8000";

const questionInput = document.getElementById("question");
const topKInput = document.getElementById("topK");
const askBtn = document.getElementById("askBtn");
const answerDiv = document.getElementById("answer");
const confidenceDiv = document.getElementById("confidence");
const sourcesDiv = document.getElementById("sources");

askBtn.addEventListener("click", async () => {
  const question = questionInput.value.trim();
  const topK = Number(topKInput.value || 5);

  if (!question) {
    alert("Vui lòng nhập câu hỏi.");
    return;
  }

  askBtn.disabled = true;
  askBtn.textContent = "Đang trả lời...";
  answerDiv.textContent = "Đang xử lý...";
  confidenceDiv.textContent = "";
  sourcesDiv.innerHTML = "";

  try {
    const response = await fetch(`${API_URL}/ask`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question: question,
        top_k: topK,
      }),
    });

    if (!response.ok) {
      throw new Error("API error");
    }

    const data = await response.json();

    answerDiv.textContent = data.answer || "Không tìm thấy câu trả lời.";
    confidenceDiv.textContent = `Confidence: ${Number(data.confidence).toFixed(4)}`;

    renderSources(data.sources || []);
  } catch (error) {
    answerDiv.textContent = "Có lỗi khi gọi backend.";
    console.error(error);
  } finally {
    askBtn.disabled = false;
    askBtn.textContent = "Ask";
  }
});

function renderSources(sources) {
  if (!sources.length) {
    sourcesDiv.innerHTML = "<p>Không có source.</p>";
    return;
  }

  sourcesDiv.innerHTML = sources
    .map((source, index) => {
      const title = source.title || "Untitled";
      const url = source.url || "";
      const text = source.text || "";
      const score = source.score || 0;
      const category = source.category || "";

      return `
        <div class="source-item">
          <div class="source-title">${index + 1}. ${escapeHtml(title)}</div>
          <div class="score">
            Score: ${Number(score).toFixed(4)} | Category: ${escapeHtml(category)}
          </div>
          ${
            url
              ? `<div class="source-url">
                   <a href="${escapeHtml(url)}" target="_blank">${escapeHtml(url)}</a>
                 </div>`
              : ""
          }
          <div class="source-text">
            ${escapeHtml(text.slice(0, 700))}${text.length > 700 ? "..." : ""}
          </div>
        </div>
      `;
    })
    .join("");
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
```

Run frontend:

```bash
cd frontend
python -m http.server 5500
```

Open:

```text
http://127.0.0.1:5500
```

---

## Step 12: Test Commands

### First-time setup

```bash
pip install -r requirements.txt
python -m src.build_index
```

### Run assignment inference

```bash
python -m src.run_inference
```

Expected output:

```text
system_outputs/system_output_1.txt
```

### Run backend

```bash
uvicorn backend.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

### Run frontend

In another terminal:

```bash
cd frontend
python -m http.server 5500
```

Open:

```text
http://127.0.0.1:5500
```

---

## Step 13: Test Questions

Use these questions to test retrieval and answer quality:

```text
When was UET established?
What is the English name of UET?
What is the address of UET?
How many admission methods does UET use in 2026?
What subject combinations are used for UET admission in 2026?
What is the admission quota for Artificial Intelligence at UET?
```

Expected behavior:

- `/ask` returns a short answer and retrieved sources.
- `/retrieve` returns only retrieved chunks.
- Frontend shows answer, confidence, and source documents.
- `run_inference.py` creates `system_outputs/system_output_1.txt` with one answer per question.

---

## Step 14: README Update

Update `README.md` with these sections:

```markdown
## RAG Model 1

Pipeline:

Question → MiniLM embedding → FAISS retrieve → DistilBERT QA → Answer

## Build Index

```bash
python -m src.build_index
```

## Run Batch Inference

```bash
python -m src.run_inference
```

Output:

```text
system_outputs/system_output_1.txt
```

## Run Backend

```bash
uvicorn backend.main:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## Run Frontend

```bash
cd frontend
python -m http.server 5500
```

Open:

```text
http://127.0.0.1:5500
```
```

---

## Implementation Order

Please implement in this order:

```text
1. requirements.txt
2. .env.example update
3. src/config.py
4. src/build_index.py
5. src/retriever.py
6. src/reader.py
7. src/rag_pipeline.py
8. src/run_inference.py
9. backend/schemas.py
10. backend/service.py
11. backend/main.py
12. frontend/index.html
13. frontend/style.css
14. frontend/app.js
15. README.md update
```

---

## Final Checklist

After implementation, verify:

```text
[ ] python -m src.build_index works
[ ] data/processed/faiss_index.bin is created
[ ] data/processed/documents.pkl is created
[ ] python -m src.run_inference works
[ ] system_outputs/system_output_1.txt is created
[ ] uvicorn backend.main:app --reload works
[ ] http://127.0.0.1:8000/docs opens
[ ] POST /ask works
[ ] POST /retrieve works
[ ] frontend opens at http://127.0.0.1:5500
[ ] frontend can call backend
[ ] frontend displays answer + confidence + sources
```

---

## Notes for Codex

- Do not delete existing dataset scripts.
- Do not move `data/processed/chunks.json`.
- Do not expose `.env` or API keys.
- Keep OpenRouter only for dataset generation/refinement unless explicitly requested.
- Main assignment output must still be `system_outputs/system_output_1.txt`.
- Keep answer output concise because assignment evaluation uses exact match, F1, and recall.
