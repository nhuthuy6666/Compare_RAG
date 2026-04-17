# Evaluation Scoring Note

File này tóm tắt cách project hiện tính điểm evaluation trong `evaluation/`.

## 1. Công thức `overall_score`

Ở mức từng câu hỏi, `overall_score` được tính theo công thức V1:

```text
Overall = 0.25 * Recall@k
        + 0.35 * Faithfulness
        + 0.25 * Answer_Relevancy
        + 0.15 * Context_Precision
```

Trong code hiện tại:

- `Recall@k = Recall@3`
- `Context_Precision = Precision@3`


## 2. 4 tiêu chí đi trực tiếp vào `overall`

### 2.1. `Recall@k`

```text
Recall@k = min(1.0, sum(relevance_score(top_k)) / total_relevant)
```

Ý nghĩa:

- đo lượng evidence liên quan mà hệ retrieve được trong top-`k`
- ở project này đang dùng `k = 3`

### 2.2. `Faithfulness`

```text
Faithfulness = supported_claims / total_claims
```

Ý nghĩa:

- tách câu trả lời thành các claim nhỏ
- kiểm tra mỗi claim có được source hỗ trợ hay không
- claim có số liệu mà source không chứa số tương ứng sẽ bị xem là không được support

### 2.3. `Answer_Relevancy`

```text
Answer_Relevance
= 0.35 * cosine(question, answer)
+ 0.40 * cosine(reference_answer, answer)
+ 0.25 * keyword_coverage(answer, answer_keywords)
```

Ý nghĩa:

- đo câu trả lời có bám đúng câu hỏi và đáp án mong đợi không
- không phạt quá nặng chỉ vì khác wording

### 2.4. `Context_Precision`

```text
Context_Precision = Precision@k = mean(relevance_score(top_k))
```

Ý nghĩa:

- đo độ liên quan trung bình của các source trong top-`k`
- ở project này đang dùng `Precision@3`

## 3. `relevance_score` của từng source được tính thế nào

Project không chấm retrieval chỉ bằng tên file hay `source_hint`, mà chấm theo nội dung source.

Nếu bài có số liệu kỳ vọng:

```text
relevance_score
= 0.30 * semantic_reference
+ 0.20 * semantic_question
+ 0.25 * context_coverage
+ 0.10 * answer_coverage
+ 0.15 * numeric_coverage
```

Nếu bài không có số liệu kỳ vọng:

```text
relevance_score
= 0.40 * semantic_reference
+ 0.25 * semantic_question
+ 0.20 * context_coverage
+ 0.15 * answer_coverage
```

Ý nghĩa ngắn gọn:

- `semantic_reference`: source gần nghĩa với đáp án chuẩn đến đâu
- `semantic_question`: source gần nghĩa với câu hỏi đến đâu
- `context_coverage`: source phủ được các keyword ngữ cảnh đến đâu
- `answer_coverage`: source phủ được các keyword đáp án đến đâu
- `numeric_coverage`: source có chứa đúng số liệu kỳ vọng không

## 4. Các tiêu chí vẫn được ghi ra báo cáo

Ngoài `overall_score`, hệ vẫn log nhiều metric khác để phân tích:

- Generation:
  - `answer_quality`
  - `exact_match`
  - `token_f1`
  - `char_similarity`
  - `semantic_similarity`
  - `keyword_coverage`
- RAG:
  - `answer_relevance`
  - `context_relevance`
  - `faithfulness`
  - `hallucination_rate`
- Retrieval:
  - `precision@1/3/5`
  - `recall@1/3/5`
  - `f1@1/3/5`
  - `hit@3`
  - `mrr`
  - `map`
  - `ndcg@3/5`
- System:
  - `latency_ms`
  - `error`
  - `source_count`

## 5. Lưu ý quan trọng khi đọc `comparison.md`

Trong bảng tổng hợp, thường thấy các cột:

- `Overall`
- `Answer`
- `Retrieval`
- `Faithfulness`
- `MRR`
- `Semantic`

Nhưng:

- `Overall` không được tính trực tiếp từ toàn bộ các cột đang hiển thị
- `Overall` chỉ dùng 4 thành phần:
  - `Recall@3`
  - `Faithfulness`
  - `Answer_Relevance`
  - `Precision@3`
- các cột như `answer_quality`, `retrieval_quality`, `mrr`, `semantic_similarity` là metric phụ để phân tích và so sánh, không đi thẳng vào công thức `overall`

## 6. `answer_quality` và `retrieval_quality` là gì

Hai metric này vẫn xuất hiện trong báo cáo, nhưng không phải công thức `overall`.

### 6.1. `answer_quality`

Với câu hỏi thường:

```text
answer_quality
= 0.05 * exact_match
+ 0.10 * token_f1
+ 0.10 * char_similarity
+ 0.35 * semantic_similarity
+ 0.25 * keyword_coverage
+ 0.15 * answer_relevance
```

Nếu hệ trả lời từ chối sai lúc không nên từ chối, điểm này còn bị nhân `0.25`.

Với câu hỏi mà ground truth mong đợi từ chối:

```text
answer_quality = 0.6 * refusal_correct + 0.4 * answer_relevance
```

### 6.2. `retrieval_quality`

Với câu hỏi thường:

```text
retrieval_quality
= 0.20 * context_relevance
+ 0.15 * precision@3
+ 0.20 * recall@3
+ 0.10 * f1@3
+ 0.10 * hit@3
+ 0.10 * mrr
+ 0.05 * map
+ 0.10 * ndcg@3
```

Với câu hỏi mong đợi từ chối:

```text
retrieval_quality = hit@3
```

## 7. Điểm tổng của từng hệ được lấy như thế nào

Ở mức hệ thống:

- mỗi câu hỏi được chấm ra một `overall_score`
- sau đó report lấy trung bình cộng của `overall_score` trên toàn bộ mẫu

Nói ngắn gọn:

```text
System Overall = mean(per-example overall_score)
```
