# Phân công và đóng góp

Tài liệu này tổng hợp đóng góp của từng thành viên theo kế hoạch trong file phân công Assignment 2.

## Nguyễn Đức Huy - Data + Evaluation

Phụ trách chính phần dữ liệu và đánh giá.

Đóng góp:

- Crawl/collect dữ liệu public về VNU/UET và các đơn vị liên quan.
- Tập trung vào các mảng:
  - lịch sử
  - tuyển sinh
  - khoa/phòng ban/cơ cấu tổ chức
  - chương trình đào tạo
  - quy chế đào tạo
- Xử lý dữ liệu thô:
  - clean text
  - chuẩn hóa nội dung HTML/PDF
  - chia documents thành chunks cho retrieval
- Xây dựng QA dataset:
  - `data/train/questions.txt`
  - `data/train/reference_answers.txt`
  - `data/test/questions.txt`
  - `data/test/reference_answers.txt`

## Nguyễn Công Cường - RAG Model 1

Phụ trách RAG Model 1.

Mô hình:

```text
Retriever: sentence-transformers/all-MiniLM-L6-v2
Reader: distilbert-base-cased-distilled-squad
Output: system_outputs/system_output_1.txt
```

Pipeline:

```text
Question
-> MiniLM embedding
-> vector retrieval
-> DistilBERT extractive QA
-> answer
```

Đóng góp:

- Xây dựng biến thể RAG dùng MiniLM làm retriever.
- Dùng DistilBERT SQuAD làm reader để trích xuất answer từ retrieved context.
- Sinh output cho local test set:
  - `system_outputs/system_output_1.txt`
- Sinh trace retrieval:
  - `system_outputs/system_output_1_trace.jsonl`

## Ngô Đình Linh - RAG Model 2

Phụ trách RAG Model 2.

Mô hình:

```text
Retriever: BAAI/bge-small-en
Reader: deepset/roberta-base-squad2
Output: system_outputs/system_output_2.txt
```

Pipeline:

```text
Question
-> BGE embedding
-> hybrid dense/lexical retrieval
-> RoBERTa extractive QA
-> answer
```

Đóng góp:

- Xây dựng biến thể RAG dùng BGE làm dense retriever.
- Kết hợp dense retrieval với lexical reranking để cải thiện retrieval quality.
- Dùng RoBERTa SQuAD2 làm reader.
- Thêm logic reranking theo loại câu hỏi như tuition, score, subject combination, degree, location, comparison.
- Sinh output cho local test set:
  - `system_outputs/system_output_2.txt`
- Sinh trace retrieval:
  - `system_outputs/system_output_2_trace.jsonl`
- Chuẩn bị dữ liệu IAA:
  - `data/iaaa/annotator_b.csv`

Kết quả local test hiện tại của System 2 là tốt nhất trong nhóm, nên được chọn làm model chính.

## Nguyễn Trần Huy - RAG Model 3

Phụ trách RAG Model 3.

Mô hình:

```text
Retriever: intfloat/e5-small-v2
Reader: deepset/roberta-base-squad2
Output: system_outputs/system_output_3.txt
```

Pipeline:

```text
Question
-> E5 embedding
-> vector retrieval
-> RoBERTa extractive QA
-> answer
```

Đóng góp:

- Xây dựng biến thể RAG dùng E5 làm retriever.
- Dùng RoBERTa SQuAD2 làm reader.
- Sinh output cho local test set:
  - `system_outputs/system_output_3.txt`
- Sinh trace retrieval:
  - `system_outputs/system_output_3_trace.jsonl`
  - Chuẩn bị dữ liệu IAA:
  - `data/iaaa/annotator_a.csv`
- Viết script tính IAA:
  - `scripts/compute_iaa.py`
- Viết script evaluation:
  - `scripts/evaluate_qa.py`
- Viết script tạo bảng và biểu đồ so sánh model:
  - `scripts/plot_model_comparison.py`
- Thêm TF-IDF RAG baseline để so sánh với các neural RAG systems:
  - `rag/rag_tfidf.py`
  - `system_outputs/system_output_4.txt`

## Công việc chung

Cả nhóm cùng tham gia:

- So sánh các biến thể RAG bằng:
  - Exact Match
  - Token F1
  - Answer Recall
- Phân tích kết quả giữa:
  - System 1: MiniLM + DistilBERT
  - System 2: BGE + RoBERTa
  - System 3: E5 + RoBERTa
  - System 4: TF-IDF baseline
- Chọn System 2 làm final/main system dựa trên local test performance.
- Viết report chung, gồm:
  - Introduction
  - Data Collection
  - Annotation + IAA
  - Models
  - Experiments
  - Analysis
  - Conclusion
