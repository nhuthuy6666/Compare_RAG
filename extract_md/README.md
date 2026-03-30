# Extract MD

Pipeline baseline cho RAG truyền thống với AnythingLLM: làm sạch dữ liệu tuyển sinh NTU, chunk thành JSONL, cập nhật workspace và ingest vào workspace `eval-clean-v2`.

## Lệnh chuẩn

Chạy trong thư mục `extract_md/`:

```bash
python scripts/build_corpus.py --scope all --sync --apply-settings --ingest --workspace-slug eval-clean-v2 --chat-model llama3.1:8b --embed-model bge-m3:latest
```

Lệnh trên sẽ:

1. Đồng bộ và dọn output stale.
2. Build TXT từ web/PDF theo `--scope`.
3. Chunk TXT thành JSONL.
4. Ghi lại `rag_baseline.json`.
5. Áp settings mới cho AnythingLLM.
6. Ingest chunk vào workspace `eval-clean-v2`.

## Corpus hiện tại

Corpus dùng chung cho `rag`, `hybrid_rag` và `graph_rag` hiện lấy từ `extract_md/data_chunks` và đang gồm 12 tài liệu:

- `2024/ntu-de-an-tuyen-sinh-2024-web.txt`
- `2025/thong-tin-tuyen-sinh-2025.txt`
- `2026/quy-che-tuyen-sinh.txt`
- `web/tuyensinh-ntu-edu-vn-de-an-tuyen-sinh-phuong-thuc-doi-tuong-pham-vi-ts.txt`
- `web/tuyensinh-ntu-edu-vn-de-an-tuyen-sinh-tuyen-sinh-lien-thong-bang-2.txt`
- `web/tuyensinh-ntu-edu-vn-diem-trung-tuyen-cac-nam.txt`
- `web/tuyensinh-ntu-edu-vn-e-an-tuyen-sinh-co-so-vat-chat.txt`
- `web/tuyensinh-ntu-edu-vn-e-an-tuyen-sinh-gioi-thieu-truong.txt`
- `web/tuyensinh-ntu-edu-vn-e-an-tuyen-sinh-hop-tac-doanh-nghiep.txt`
- `web/tuyensinh-ntu-edu-vn-e-an-tuyen-sinh-to-chuc-tuyen-sinh.txt`
- `web/tuyensinh-ntu-edu-vn-nhap-hoc.txt`
- `web/tuyensinh-ntu-edu-vn-tu-van.txt`

## Kiểm tra đồng bộ corpus với Qdrant

Hiện không còn script audit riêng để sinh `corpus_manifest.*`.
Nếu cần kiểm tra đồng bộ:

- kiểm tra `extract_md/data_chunks` đúng số file `.jsonl`/số dòng chunk như mong đợi
- kiểm tra Qdrant collections (`ntu_rag`, `ntu_hybrid_llamaindex`, `ntu_graphrag`) có đúng số points sau khi ingest/reindex

## Baseline hiện tại

- Workspace: `eval-clean-v2`
- Chat model: `llama3.1:8b`
- Embedding model: `bge-m3:latest`
- Chunk profile: `balanced`
- `min_chars=120`
- `max_chars=700`
- `max_sections_per_chunk=2`
- `topN=6`
- `similarityThreshold=0.25`
- `temperature=0.1`
- `history=1`
- `vectorSearchMode=default`
- `chatMode=chat`

## Lệnh hữu ích

```bash
python scripts/build_corpus.py --scope web --offline --sync
python scripts/build_corpus.py --scope all
python scripts/build_corpus.py --scope web --chunk-profile max
python scripts/build_corpus.py --scope web --apply-settings --workspace-slug eval-clean-v2 --chat-model llama3.1:8b
```

## Ghi chú

- Nếu model Ollama chưa có, script sẽ tự `pull` trừ khi dùng `--skip-model-pull`.
- Nếu đổi embedding model, nên restart AnythingLLM trước khi ingest lại để desktop settings được áp dụng ổn định.
- `rag_baseline.json` là cấu hình chung đang được `rag`, `hybrid_rag` và `graph_rag` tái sử dụng.
- `data_raw` chỉ là dữ liệu thô; dữ liệu mà các hệ RAG ingest trực tiếp là `data_chunks` và được quy chiếu về tên tài liệu `.txt` trong metadata.
