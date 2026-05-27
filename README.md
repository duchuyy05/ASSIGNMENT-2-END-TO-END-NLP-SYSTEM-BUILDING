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

## Run TF-IDF RAG

The baseline RAG system is implemented in `rag/rag_tfidf.py`:

1. Embedder: `TfidfVectorizer` over corpus chunks and optional train QA fact chunks.
2. Retriever: cosine similarity over TF-IDF vectors.
3. Reader: extractive answer heuristics over the retrieved chunks.

Generate answers for the current test questions:

```bash
python rag/rag_tfidf.py \
  --questions data/test/questions.txt \
  --output system_outputs/system_output_1.txt \
  --trace system_outputs/system_output_1_trace.jsonl
```

Use `--no-train-facts` to retrieve only from `data/processed/chunks.json`.

## Run BGE + RoBERTa Optimized RAG

System 2 is implemented in `rag/rag_bge_roberta.py`:

1. Embedder/retriever model: `BAAI/bge-small-en`.
2. Hybrid retrieval: BGE dense similarity plus TF-IDF lexical similarity,
   combined with a default weight of `0.70` dense and `0.30` lexical.
3. Reader model: `deepset/roberta-base-squad2`.
4. Answer selection: question-type aware reranking for programs, scores,
   tuition, locations, degrees, subjects, and comparison questions.

Generate and evaluate `system_output_2.txt` in one command:

```powershell
python scripts/run_system2_pipeline.py
```

Generate only the answers and trace:

```bash
python rag/rag_bge_roberta.py \
  --questions data/test/questions.txt \
  --output system_outputs/system_output_2.txt \
  --trace system_outputs/system_output_2_trace.jsonl
```

Useful smoke test:

```powershell
python scripts/run_system2_pipeline.py --limit 5
```

## Run E5 + RoBERTa RAG

The neural RAG variation is implemented in `rag/rag_e5_roberta.py`:

1. Embedder/retriever model: `intfloat/e5-small-v2`.
2. Vector search: FAISS if installed, otherwise NumPy inner-product search over
   normalized embeddings.
3. Reader model: `deepset/roberta-base-squad2`.

Generate `system_output_3.txt`:

```bash
python rag/rag_e5_roberta.py \
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

Current local test result for `system_output_1.txt` using MiniLM + DistilBERT :

- Exact Match: 38.60
- Token F1: 55.44
- Answer Recall: 71.05

Current local test result for `system_output_2.txt` using BGE + RoBERTa with
hybrid reranking:

- Exact Match: 57.02
- Token F1: 69.50
- Answer Recall: 85.09

Current local test result for `system_output_3.txt` using E5 + RoBERTa:

- Exact Match: 50.88
- Token F1: 60.77
- Answer Recall: 71.93
