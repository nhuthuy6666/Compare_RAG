# Hybrid RAG

Hybrid RAG hiện chạy bằng `LlamaIndex + Qdrant`, không còn dùng Haystack. Hệ này dùng hybrid retrieval:

- dense retrieval lưu/persist trong Qdrant
- sparse retrieval dùng BM25 qua hybrid search của Qdrant

Toàn bộ chunk, prompt và tham số retrieval được đọc từ `extract_md/rag_baseline.json` để giữ công bằng với các hệ local còn lại.

## Yêu cầu

- Python 3.12
- Ollama chạy tại `http://localhost:11434`
- Qdrant chạy tại `http://127.0.0.1:6333`
- Model Ollama:
  - `bge-m3:latest` cho embedding
  - `llama3.1:8b` cho generation

## Luồng chạy

- App đọc config dùng chung từ `extract_md/rag_baseline.json`.
- App nạp chunk JSONL, tạo fingerprint corpus và kiểm tra state tại `hybrid_rag/.qdrant_state.json`.
- Nếu collection `ntu_hybrid_llamaindex` đã hợp lệ thì mở lại hybrid index trong Qdrant.
- Nếu fingerprint thay đổi thì app sẽ reset collection và ingest lại.
- Sau đó app expose UI, `GET /health` và `POST /api/chat`.

## Lệnh chính

Cài dependency:

```bash
pip install -r hybrid_rag/requirements.txt
```

Ý nghĩa:

- cài package cần thiết cho Hybrid RAG
- cần chạy lại sau khi thay đổi dependency

Tải model Ollama:

```bash
ollama pull bge-m3:latest
ollama pull llama3.1:8b
```

Ý nghĩa:

- đảm bảo model embedding và generation đã có sẵn trong Ollama
- chỉ cần tải lại khi máy chưa có model

Chạy web UI và API:

```bash
python hybrid_rag/app.py
```

Ý nghĩa:

- khởi động Hybrid RAG local
- tự động mở/tạo collection `ntu_hybrid_llamaindex` trong Qdrant
- expose giao diện mặc định tại `http://127.0.0.1:8000`

Chạy chế độ CLI:

```bash
python hybrid_rag/app.py --cli
```

Ý nghĩa:

- không mở web server
- cho phép test nhanh retrieval hybrid và answer ngay trong terminal

Chạy với biến môi trường tùy chỉnh:

```bash
set QDRANT_URL=http://127.0.0.1:6333
set QDRANT_COLLECTION=ntu_hybrid_llamaindex
python hybrid_rag/app.py
```

Ý nghĩa:

- override Qdrant server/collection cho phiên chạy hiện tại
- hữu ích khi muốn tách collection giữa các lần thử nghiệm

## Biến môi trường hữu ích

- `LLM_BASE_URL`: mặc định `http://localhost:11434/v1`
- `LLM_MODEL`: mặc định `llama3.1:8b`
- `EMBED_MODEL`: mặc định `bge-m3:latest`
- `QDRANT_URL`: mặc định `http://127.0.0.1:6333`
- `QDRANT_COLLECTION`: mặc định `ntu_hybrid_llamaindex`
- `CHUNK_JSONL_ROOT`: override thư mục chunk nếu cần
- `CORPUS_SCOPE`: giới hạn phạm vi chunk ingest nếu cần
- `RETRIEVAL_TOP_N`: số chunk retrieve mỗi lần query
- `RETRIEVAL_SIMILARITY_THRESHOLD`: ngưỡng từ chối trả lời nếu score quá thấp

## Ghi chú

- Ở lần chạy đầu, app sẽ đọc chunk từ `extract_md/rag_baseline.json`, tạo embedding và ghi collection vào Qdrant.
- Các lần chạy sau sẽ tái sử dụng collection nếu corpus và embedding model không đổi.
- Trạng thái index được lưu tại `hybrid_rag/.qdrant_state.json`.
- `hybrid_rag/app.py` là entrypoint chính; trong code, `main()` đã được note rõ từng bước chạy.
