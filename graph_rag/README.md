# GraphRAG

GraphRAG hiện chỉ dùng `Qdrant + LlamaIndex`.

Kiến trúc hiện tại:

- đọc cùng chunk đầu vào từ `extract_md/rag_baseline.json`
- trích `graph facts` từ mỗi chunk bằng extractor cục bộ
- lưu toàn bộ fact nodes trực tiếp vào Qdrant
- query trên tập graph facts bằng LlamaIndex
- dùng `llama3.1:8b` cho generation/extraction và `bge-m3:latest` cho embedding

Điểm khác với `rag` và `hybrid_rag` là hệ này không retrieve nguyên chunk trước, mà ưu tiên fact dạng quan hệ như mã trường, phương thức, chỉ tiêu, điểm chuẩn, tổ hợp xét tuyển, điều kiện tiếng Anh.

Lưu ý về tính công bằng:

- GraphRAG hiện chỉ giữ các bước thuộc đặc trưng kiến trúc.
- Hệ không còn dùng tăng cường truy vấn thủ công kiểu thêm từ khóa đồng nghĩa trước khi query.
- Câu hỏi được truy vấn trực tiếp trên tập graph facts đã ingest.

## Preset khuyến nghị

```env
LLM_TIMEOUT=180
EMBED_TIMEOUT=300
EMBED_BATCH_SIZE=4
EMBED_RETRY_ATTEMPTS=8
EMBED_RETRY_DELAY=8
GRAPH_PROGRESS_EVERY=5
```

## Lệnh chính

Ingest lại toàn bộ graph facts vào Qdrant:

```bash
python graph_rag/src/ingest.py --reset-graph
```

Ý nghĩa:

- đọc lại toàn bộ chunk dùng chung
- trích graph facts mới
- reset collection `ntu_graphrag`
- ghi lại toàn bộ facts vào Qdrant

Wrapper tương thích cũ:

```bash
python graph_rag/src/sync_qdrant.py
```

Ý nghĩa:

- giữ tương thích với lệnh cũ
- thực chất gọi lại flow ingest Qdrant-only ở trên

Chạy query CLI:

```bash
python graph_rag/src/query.py
python graph_rag/src/query.py --question "Điểm khác nhau giữa các phương thức tuyển sinh năm 2025 là gì?"
```

Ý nghĩa:

- lệnh đầu mở chế độ interactive hỏi nhiều câu liên tiếp
- lệnh thứ hai chạy một câu duy nhất rồi thoát

Chạy webapp:

```bash
python graph_rag/app.py
```

Ý nghĩa:

- mở UI tại `http://127.0.0.1:8502`
- đồng thời expose `GET /health` và `POST /api/chat` cho benchmark/stateless query

## Sau khi đổi model hoặc chunk/schema

- cập nhật `graph_rag/.env` sao cho `LLM_MODEL=llama3.1:8b`
- chạy `python graph_rag/src/ingest.py --reset-graph`
- sau đó chạy `python graph_rag/app.py` và mở `http://127.0.0.1:8502`

## Ghi chú

- Collection Qdrant mặc định lấy từ `QDRANT_COLLECTION`, giá trị mặc định là `ntu_graphrag`.
- File audit graph facts được ghi ra `graph_rag/data/processed/graph_facts.jsonl`.
- `graph_rag/app.py` là entrypoint chính và duy nhất để chạy web UI/API của GraphRAG.
