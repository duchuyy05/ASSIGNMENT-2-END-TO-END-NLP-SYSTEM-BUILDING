# VNU/UET RAG QA System

Repo này xây dựng hệ thống RAG cho bài Assignment 2: trả lời câu hỏi factual QA về VNU/UET và một số đơn vị liên quan như USSH, ULIS.

## 1. Cấu trúc chính

```text
data/
  manual_annotations/        CSV QA gốc do nhóm curate
  processed/                 documents, chunks, metadata
  annotations/               QA JSON có metadata
  train/                     train questions/reference answers theo format đề
  test/                      test questions/reference answers theo format đề
  iaaa/                      dữ liệu và kết quả IAA
rag/                         các hệ thống RAG
scripts/                     build data, evaluation, IAA, plot
system_outputs/              output của các hệ thống
results/                     bảng và biểu đồ so sánh model
```

## 2. Cài đặt

```bash
python3 -m pip install -r requirements.txt
```

Lần đầu chạy các neural RAG systems sẽ cần tải model từ HuggingFace.

## 3. Build dữ liệu

Dataset được build từ các file CSV thủ công trong `data/manual_annotations/`.

Chạy lại từ documents đã xử lý:

```bash
python scripts/build_manual_dataset.py --reuse-documents
```

Nếu muốn fetch lại source public:

```bash
python scripts/build_manual_dataset.py --timeout 12
```

Các file chính được tạo:

```text
data/train/questions.txt
data/train/reference_answers.txt
data/test/questions.txt
data/test/reference_answers.txt
data/processed/documents.json
data/processed/chunks.json
data/processed/dataset_metadata.json
```

Hiện tại dataset có:

```text
Train QA: 360
Test QA: 114
Documents: 29
Chunks: 461
```

## 4. Các hệ thống RAG

### System 1: MiniLM + DistilBERT

Theo phân công của Công Cường.

```text
Retriever: sentence-transformers/all-MiniLM-L6-v2
Reader: distilbert-base-cased-distilled-squad
Output: system_outputs/system_output_1.txt
```

Chạy:

```bash
python scripts/run_system1_pipeline.py
```

Hoặc chạy trực tiếp:

```bash
python rag/rag_minilm_distilbert.py \
  --questions data/test/questions.txt \
  --output system_outputs/system_output_1.txt \
  --trace system_outputs/system_output_1_trace.jsonl
```

### System 2: BGE + RoBERTa

Theo phân công của Linh. Đây là hệ thống có kết quả tốt nhất trên local test set hiện tại.

```text
Retriever: BAAI/bge-small-en
Reader: deepset/roberta-base-squad2
Retrieval: hybrid dense + lexical reranking
Output: system_outputs/system_output_2.txt
```

Chạy:

```bash
python scripts/run_system2_pipeline.py
```

Hoặc chạy trực tiếp:

```bash
python rag/rag_bge_roberta.py \
  --questions data/test/questions.txt \
  --output system_outputs/system_output_2.txt \
  --trace system_outputs/system_output_2_trace.jsonl
```

### System 3: E5 + RoBERTa

Theo phân công của Huy.

```text
Retriever: intfloat/e5-small-v2
Reader: deepset/roberta-base-squad2
Output: system_outputs/system_output_3.txt
```

Chạy:

```bash
python rag/rag_e5_roberta.py \
  --questions data/test/questions.txt \
  --output system_outputs/system_output_3.txt \
  --trace system_outputs/system_output_3_trace.jsonl
```

### System 4: TF-IDF RAG baseline

Baseline lexical để so sánh thêm, không nằm trong 3 output chính theo cấu trúc nộp bài.

```text
Retriever: TF-IDF cosine similarity
Reader: rule-based extractive reader
Output: system_outputs/system_output_4.txt
```

Chạy:

```bash
python rag/rag_tfidf.py \
  --questions data/test/questions.txt \
  --output system_outputs/system_output_4.txt \
  --trace system_outputs/system_output_4_trace.jsonl
```

## 5. Đánh giá

Script đánh giá:

```bash
python scripts/evaluate_qa.py \
  --predictions system_outputs/system_output_2.txt \
  --references data/test/reference_answers.txt
```

Metrics:

```text
Exact Match
Token F1
Answer Recall
```

Kết quả local test hiện tại:

| System | Model | EM | F1 | Recall |
|---|---|---:|---:|---:|
| System 1 | MiniLM + DistilBERT | 38.60 | 55.44 | 71.05 |
| System 2 | BGE + RoBERTa | 57.02 | 69.50 | 85.09 |
| System 3 | E5 + RoBERTa | 50.88 | 60.77 | 71.93 |
| System 4 | TF-IDF baseline | 49.12 | 58.21 | 70.18 |

System 2 được chọn làm model chính vì đạt kết quả tốt nhất trên local test set.

## 6. Vẽ biểu đồ so sánh

```bash
python scripts/plot_model_comparison.py
```

Output:

```text
results/model_comparison.csv
results/model_comparison.json
results/model_comparison.svg
```

## 7. IAA

Hai file annotation:

```text
data/iaaa/annotator_a.csv
data/iaaa/annotator_b.csv
```

Tính agreement:

```bash
python scripts/compute_iaa.py
```

Output:

```text
data/iaaa/iaa_results.json
data/iaaa/iaa_disagreements.csv
```

Kết quả hiện tại:

```text
Items: 30
Exact agreement: 16.67%
Soft agreement, token F1 >= 0.80: 36.67%
Mean token F1: 56.85
```

IAA thấp vì subset được chọn gồm nhiều câu khó, multi-answer, answer dài, và hai annotator có khác biệt về phạm vi câu trả lời. Phần này cần được phân tích/adjudicate trong report.

## 8. Lưu ý về FAISS

Các neural RAG scripts hỗ trợ vector retrieval. Nếu môi trường có `faiss`, script có thể dùng FAISS; nếu không, hệ thống fallback sang NumPy inner-product search trên normalized embeddings.

Trong report nên ghi chính xác là:

```text
exact vector search over normalized embeddings, with optional FAISS support
```

## 9. Cấu trúc nộp bài

Theo đề, thư mục nộp chính nên có:

```text
ANDREWID/
  report.pdf
  github_url.txt
  contributions.md
  data/
    train/
      questions.txt
      reference_answers.txt
    test/
      questions.txt
      reference_answers.txt
  system_outputs/
    system_output_1.txt
    system_output_2.txt
    system_output_3.txt
  README.md
```

`system_output_4.txt` là baseline để so sánh nội bộ/report, không bắt buộc đưa vào zip nộp nếu đề yêu cầu tối đa 3 output files.
