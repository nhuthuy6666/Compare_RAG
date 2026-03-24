# Evaluation

Thư mục này hiện chỉ còn một flow đánh giá duy nhất: `evaluate-v1`.

Ba hệ được benchmark:

- `baseline`: project `rag`
- `hybrid`: project `hybrid_rag`
- `graphrag`: project `graph_rag`

Cả ba hệ cùng dùng:

- dữ liệu từ `extract_md/rag_baseline.json`
- embedding `bge-m3:latest`
- chat model `llama3.1:8b`
- cùng bộ test V1 cân bằng trong `evaluation/dataset/testset.json` + `evaluation/dataset/testset_additions_v1.json`

## Bộ test V1 cân bằng

Bộ test V1 hiện có tổng `72` câu, chia đều thành `3` nhóm:

- `24` câu `dense_shared`: thiên về thế mạnh chung của `rag` và `hybrid_rag`, chủ yếu là bảng số liệu, quota, scores, IELTS, multi-fact trên chunk
- `24` câu `graph`: thiên về thế mạnh của `graph_rag`, chủ yếu là quan hệ thực thể - thuộc tính như tư vấn ngành, cơ sở vật chất, phạm vi, danh tính
- `24` câu `neutral`: nhóm trung lập gồm câu quy trình, chi phí, câu từ chối và một phần fact đơn giản

Bucket được gán tự động khi load dataset và được ghi ra báo cáo sau mỗi lần evaluate.

## Công thức Overall

`Overall = 0.25 * Recall@k + 0.35 * Faithfulness + 0.25 * Answer_Relevancy + 0.15 * Context_Precision`

Với thiết lập hiện tại:

- `k = 3`
- `Recall@k = Recall@3`
- `Context_Precision = Precision@3`

Giải thích chi tiết metric nằm trong `evaluation/METRICS_V1.md`.

## Chạy benchmark

Chạy toàn bộ:

```bash
python evaluation/evaluate-v1.py
```

Chạy từng hệ:

```bash
python evaluation/evaluate-v1.py --system baseline
python evaluation/evaluate-v1.py --system hybrid
python evaluation/evaluate-v1.py --system graphrag
```

Smoke test:

```bash
python evaluation/evaluate-v1.py --limit 3
```

## Tạo báo cáo từ kết quả V1

```bash
python evaluation/compare.py
python evaluation/visualize.py
python evaluation/reliability.py
```

Các script trên mặc định dùng `evaluation/config_v1.yaml`.

## File quan trọng

- `evaluation/evaluate-v1.py`: entrypoint benchmark V1
- `evaluation/config_v1.yaml`: cấu hình dataset, endpoint và thư mục kết quả
- `evaluation/METRICS_V1.md`: mô tả metric và công thức
- `evaluation/outputs_v1/`: output gốc của từng hệ sau khi chạy
- `evaluation/results_v1/`: file metric và báo cáo tổng hợp sau khi chạy
- `evaluation/results_v1/strength_breakdown.csv`: breakdown theo `dense_shared`, `graph`, `neutral`
- `evaluation/dataset/testset_additions_v1.json`: 23 câu bổ sung để nâng bộ test lên 72 câu
- `evaluation/dataset/strength_buckets_v1.json`: các override bucket cho những câu không thể suy ra chỉ từ `topic`

## Ghi chú

- Thư mục đã được dọn sạch phần legacy để chỉ còn V1.
- Trước mỗi lần benchmark mới, bạn có thể xóa nội dung trong `outputs_v1/` và `results_v1/` hoặc để `evaluate-v1` ghi đè kết quả cũ.
