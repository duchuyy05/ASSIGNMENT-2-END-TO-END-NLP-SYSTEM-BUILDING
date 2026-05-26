# VNU/UET RAG Data Builder

Build the current manual dataset from the curated CSV files:

```powershell
python scripts/build_manual_dataset.py --reuse-documents
```

If `data/processed/documents.json` is missing or you want to re-fetch source
pages, run:

```powershell
python scripts/build_manual_dataset.py --timeout 12
```

Generated files are written under `data/`. The original CSV files in
`data/manual_annotations/` are treated as source files and are not modified by
the build script.

## RAG Model 1

Pipeline:

```text
Question -> MiniLM embedding -> FAISS retrieve -> DistilBERT QA -> Answer
```

OpenRouter is used only by dataset generation/refinement scripts in `scripts/`.
The main assignment model remains local.

## Build Index

```powershell
python -m src.build_index
```

This reads `data/processed/chunks.json` and writes:

```text
data/processed/faiss_index.bin
data/processed/chunks.pkl
```

## Run Batch Inference

```powershell
python -m src.run_inference
```

Output:

```text
system_outputs/system_output_1.txt
```

## Run Backend

```powershell
uvicorn backend.main:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Example request:

```json
{
  "question": "What is the English name of UET?",
  "top_k": 5
}
```

## Run Frontend

In another terminal:

```powershell
cd frontend
python -m http.server 5500
```

Open:

```text
http://127.0.0.1:5500
```
