# VNU/UET RAG Baseline

This repository contains the assignment data pipeline and a lightweight RAG
baseline for factual QA about VNU/UET.

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Build Data

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

## Run RAG

The baseline RAG system is implemented in `scripts/rag_tfidf.py`:

1. Embedder: `TfidfVectorizer` over corpus chunks and optional train QA fact chunks.
2. Retriever: cosine similarity over TF-IDF vectors.
3. Reader: extractive answer heuristics over the retrieved chunks.

Generate answers for the current test questions:

```bash
python scripts/rag_tfidf.py \
  --questions data/test/questions.txt \
  --output system_outputs/system_output_1.txt \
  --trace system_outputs/system_output_1_trace.jsonl
```

Use `--no-train-facts` to retrieve only from `data/processed/chunks.json`.

## Run E5 + RoBERTa RAG

The neural RAG variation is implemented in `scripts/rag_e5_roberta.py`:

1. Embedder/retriever model: `intfloat/e5-small-v2`.
2. Vector search: FAISS if installed, otherwise NumPy inner-product search over
   normalized embeddings.
3. Reader model: `deepset/roberta-base-squad2`.

Generate `system_output_3.txt`:

```bash
python scripts/rag_e5_roberta.py \
  --questions data/test/questions.txt \
  --output system_outputs/system_output_3.txt \
  --trace system_outputs/system_output_3_trace.jsonl
```

The first run downloads HuggingFace model weights unless they are already
cached locally.

## Evaluate

Evaluate a system output against the local reference answers:

```bash
python scripts/evaluate_qa.py \
  --predictions system_outputs/system_output_1.txt \
  --references data/test/reference_answers.txt
```

Current local test result for `system_output_1.txt`:

- Exact Match: 49.12
- Token F1: 58.21
- Answer Recall: 70.18

Current local test result for `system_output_3.txt` using E5 + RoBERTa:

- Exact Match: 50.88
- Token F1: 60.77
- Answer Recall: 71.93
