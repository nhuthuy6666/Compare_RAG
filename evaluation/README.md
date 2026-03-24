# Evaluation

Repo hiện giữ song song hai bộ benchmark:

- Legacy: `evaluation/evaluate.py`, kết quả ở `evaluation/results/` và `evaluation/outputs/`
- V1: `evaluation/evaluate-v1.py`, kết quả ở `evaluation/results_v1/` và `evaluation/outputs_v1/`

V1 hiện chỉ đánh giá 3 hệ local:

- `baseline`: project `rag`
- `hybrid`: project `hybrid_rag`
- `graphrag`: project `graph_rag`

Project AnythingLLM vẫn được giữ nguyên ở nhánh `extract_md`, nhưng không còn nằm trong flow benchmark V1.

## Kiến trúc V1

- `rag`: dense retrieval với LlamaIndex + Qdrant
- `hybrid_rag`: hybrid retrieval với LlamaIndex + Qdrant hybrid search
- `graph_rag`: graph facts extracted từ chunk và lưu trực tiếp trong Qdrant, query bằng LlamaIndex

Cả 3 hệ dùng chung:

- chunk đầu vào từ `extract_md/rag_baseline.json`
- embedding model `bge-m3:latest`
- generation model `llama3.1:8b`
- prompt/refusal/retrieval top-k từ baseline config

## Công thức Overall V1

`Overall = 0.25 * Recall@k + 0.35 * Faithfulness + 0.25 * Answer_Relevancy + 0.15 * Context_Precision`

Với benchmark hiện tại:

- `k = 3`
- `Recall@k = Recall@3`
- `Context_Precision = Precision@3` trên tập context/source retrieve

## Ý nghĩa từng thành phần

- `Recall@k`: trong số source relevant mong đợi, hệ retrieve được bao nhiêu source trong top-`k`
- `Faithfulness`: mức độ câu trả lời được source hỗ trợ, càng cao thì càng ít hallucination
- `Answer_Relevancy`: mức độ câu trả lời bám đúng câu hỏi
- `Context_Precision`: tỷ lệ source relevant trong top-`k` context retrieve

Giải thích chi tiết và công thức con nằm trong `evaluation/METRICS_V1.md`.

## Chạy 3 hệ local

Baseline LlamaIndex:

```bash
pip install -r rag/requirements.txt
python rag/app.py
```

Hybrid RAG:

```bash
pip install -r hybrid_rag/requirements.txt
python hybrid_rag/app.py
```

GraphRAG:

```bash
pip install -r graph_rag/requirements.txt
python graph_rag/src/ingest.py --reset-graph
python graph_rag/app.py
```

Ý nghĩa:

- cài dependency cho GraphRAG
- rebuild graph facts vào Qdrant
- mở webapp và endpoint benchmark ở `http://127.0.0.1:8502`

## Chạy benchmark V1

Chạy toàn bộ:

```bash
python evaluation/evaluate-v1.py
```

Ý nghĩa:

- gọi cả 3 hệ `baseline`, `hybrid`, `graphrag`
- ghi output vào `evaluation/outputs_v1/`
- ghi metric vào `evaluation/results_v1/`

Chạy từng hệ:

```bash
python evaluation/evaluate-v1.py --system baseline
python evaluation/evaluate-v1.py --system hybrid
python evaluation/evaluate-v1.py --system graphrag
```

Ý nghĩa:

- chỉ benchmark đúng một hệ để debug nhanh
- hữu ích khi vừa rebuild riêng một pipeline

Smoke test:

```bash
python evaluation/evaluate-v1.py --limit 3
```

Ý nghĩa:

- chỉ chạy 3 câu đầu của testset
- dùng để kiểm tra nhanh endpoint/model/Qdrant trước khi benchmark full

Tạo báo cáo từ kết quả V1:

```bash
python evaluation/compare.py --config evaluation/config_v1.yaml
python evaluation/visualize.py --config evaluation/config_v1.yaml
python evaluation/reliability.py --config evaluation/config_v1.yaml
```

## File quan trọng của V1

- `evaluation/evaluate-v1.py`: entrypoint benchmark V1
- `evaluation/config_v1.yaml`: endpoint và thư mục output/result của V1
- `evaluation/METRICS_V1.md`: mô tả metric và công thức V1
- `evaluation/results_v1/comparison.csv`: bảng so sánh tổng hợp V1
- `evaluation/results_v1/*_metrics.csv`: metric chi tiết từng câu hỏi
- `evaluation/outputs_v1/*_outputs.json`: output gốc của từng hệ

## Kết quả benchmark mới nhất

Theo `evaluation/results_v1/comparison.csv`:

- `baseline`: `overall = 0.7074`
- `hybrid`: `overall = 0.6395`
- `graphrag`: `overall = 0.2308`

## Ghi chú

- `evaluation/evaluate.py` và `evaluation/METRICS.md` được giữ nguyên để bảo toàn benchmark cũ.
- `evaluation/runners/run_baseline.py` vẫn tương thích AnythingLLM cũ để không phá flow legacy, nhưng `config_v1.yaml` hiện chỉ trỏ sang baseline local mới.
