# RAG

Đây là baseline local để so sánh công bằng với `hybrid_rag` và `graph_rag`.

Hệ này dùng:

- `LlamaIndex + Qdrant`
- cùng chunk từ `extract_md/rag_baseline.json`
- cùng prompt, retrieval top-k, refusal response
- cùng model `llama3.1:8b` và `bge-m3:latest`

Khác biệt chính so với `hybrid_rag` là baseline chỉ dùng dense retrieval, không bật hybrid sparse+dense.

## Luồng chạy

- App đọc config dùng chung từ `extract_md/rag_baseline.json`.
- App nạp chunk JSONL, tạo fingerprint corpus và kiểm tra state tại `rag/.qdrant_state.json`.
- Nếu collection `ntu_rag` đã hợp lệ thì mở lại index trong Qdrant.
- Nếu fingerprint thay đổi thì app sẽ reindex lại toàn bộ corpus vào Qdrant.
- Sau đó app expose UI, `GET /health` và `POST /api/chat`.

## Lệnh chính

Cài dependency:

```bash
pip install -r rag/requirements.txt
```

Ý nghĩa:

- cài package cần thiết cho baseline RAG
- cần chạy lại sau khi thay đổi dependency

Chạy web UI và API:

```bash
python rag/app.py
```

Ý nghĩa:

- khởi động baseline RAG local
- tự động mở/tạo collection `ntu_rag` trong Qdrant
- expose UI và API mặc định tại `http://127.0.0.1:8001`

Chạy chế độ CLI:

```bash
python rag/app.py --cli
```

Ý nghĩa:

- không mở web server
- cho phép test retrieval và answer ngay trong terminal

## Biến môi trường hữu ích

- `LLM_BASE_URL`: mặc định `http://localhost:11434/v1`
- `LLM_MODEL`: mặc định `llama3.1:8b`
- `EMBED_MODEL`: mặc định `bge-m3:latest`
- `QDRANT_URL`: mặc định `http://127.0.0.1:6333`
- `QDRANT_COLLECTION`: mặc định `ntu_rag`
- `CHUNK_JSONL_ROOT`: override thư mục chunk nếu cần
- `CORPUS_SCOPE`: giới hạn phạm vi chunk ingest nếu cần
- `RETRIEVAL_TOP_N`: số chunk retrieve mỗi lần query
- `RETRIEVAL_SIMILARITY_THRESHOLD`: ngưỡng từ chối trả lời khi score quá thấp

## Ghi chú

- Collection Qdrant mặc định: `ntu_rag`.
- Trạng thái index được lưu tại `rag/.qdrant_state.json`.
- `rag/app.py` là entrypoint chính; trong code, `main()` đã được note rõ từng bước chạy.
