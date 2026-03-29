from __future__ import annotations

import argparse
from pathlib import Path

from llama_index.core import Settings
from openai import APIConnectionError, APITimeoutError

from config import build_vector_store, configure_models, load_config, reset_vector_store
from fact_graph import build_graph_fact_nodes, write_fact_records
from utils import (
    configure_console_utf8,
    load_chunk_record_groups,
    summarize_records,
    write_chunk_records,
)


## Embed graph facts theo batch và tự chia nhỏ batch khi Ollama/Qdrant gặp lỗi NaN hoặc timeout.
## Mục tiêu là giữ cho ingest không chết toàn bộ chỉ vì một batch embedding lỗi.
def attach_embeddings(nodes, embed_model, batch_size: int) -> list:
    embedded_nodes: list = []

    # Embed một batch node và gắn embedding trực tiếp vào từng node.
    def _embed_batch(batch_nodes: list) -> None:
        if not batch_nodes:
            return

        texts = [node.text[:3000] for node in batch_nodes]
        try:
            embeddings = embed_model.get_text_embedding_batch(texts)
            for node, embedding in zip(batch_nodes, embeddings, strict=True):
                node.embedding = list(embedding)
                embedded_nodes.append(node)
            return
        except Exception:
            if len(batch_nodes) == 1:
                try:
                    batch_nodes[0].embedding = list(embed_model.get_text_embedding(texts[0]))
                    embedded_nodes.append(batch_nodes[0])
                except Exception:
                    pass
                return

        midpoint = len(batch_nodes) // 2
        _embed_batch(batch_nodes[:midpoint])
        _embed_batch(batch_nodes[midpoint:])

    for index in range(0, len(nodes), batch_size):
        _embed_batch(nodes[index : index + batch_size])

    return embedded_nodes


## Khai báo CLI cho flow ingest GraphRAG Qdrant-only.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest GraphRAG tu shared chunk JSONL vao Qdrant.")
    parser.add_argument("--chunk-jsonl-root", type=Path, help="Folder shared chunk JSONL de dung dung chunk baseline.")
    parser.add_argument("--output-dir", type=Path, help="Folder luu JSONL audit cuc bo cho GraphRAG.")
    parser.add_argument("--limit", type=int, help="Chi xu ly N file dau tien.")
    parser.add_argument("--min-chars", type=int, help="Minimum chunk size before merging adjacent sections.")
    parser.add_argument("--max-chars", type=int, help="Maximum chunk size before splitting a section.")
    parser.add_argument(
        "--max-sections-per-chunk",
        type=int,
        help="Maximum adjacent sections to merge into one chunk.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build chunk outputs and graph facts without writing to Qdrant.",
    )
    parser.add_argument(
        "--reset-graph",
        action="store_true",
        help="Reset Qdrant collection before rebuilding graph facts.",
    )
    return parser.parse_args()


## Nạp tập chunk records đầu vào.
## Chỉ dùng shared JSONL từ `extract_md` để giữ pipeline đồng bộ với baseline.
def _load_records(args: argparse.Namespace, config) -> list[dict]:
    chunk_jsonl_root = args.chunk_jsonl_root or config.source_chunk_root
    output_root = args.output_dir or config.chunk_dir

    all_records: list[dict] = []
    record_groups = load_chunk_record_groups(
        chunk_root=chunk_jsonl_root,
        scope=config.corpus_scope,
        limit=args.limit,
    )

    if record_groups:
        for jsonl_path, records in record_groups:
            all_records.extend(records)
            relative_path = jsonl_path.relative_to(chunk_jsonl_root)
            output_path = output_root / relative_path
            write_chunk_records(records, output_path)
            print(f"[shared-chunks] {jsonl_path} -> {output_path} ({len(records)} chunks)")
        return all_records

    raise FileNotFoundError(
        f"Khong tim thay shared chunk JSONL trong {chunk_jsonl_root} (scope={config.corpus_scope}). "
        "Hay tao/cap nhat corpus chunk truoc khi ingest GraphRAG."
    )


## Entry point ingest GraphRAG Qdrant-only.
## 1) Cấu hình console UTF-8 và đọc CLI/config.
## 2) Nạp chunk records từ shared JSONL.
## 3) In thống kê corpus để theo dõi quy mô ingest.
## 4) Khởi tạo model và trích graph facts từ từng chunk.
## 5) Ghi graph facts ra file JSONL audit để dễ debug.
## 6) Nếu `--dry-run` thì dừng ở bước audit, không ghi Qdrant.
## 7) Embed toàn bộ fact nodes với fallback chia nhỏ batch khi cần.
## 8) Reset collection nếu được yêu cầu rồi ghi toàn bộ graph facts vào Qdrant.
def main() -> None:
    configure_console_utf8()
    args = parse_args()
    config = load_config()
    all_records = _load_records(args, config)

    if not all_records:
        raise ValueError("Khong tao duoc chunk nao tu du lieu dau vao.")

    summary = summarize_records(all_records)
    print(
        f"Tao {summary['chunk_count']} chunks, tong {summary['total_chars']} ky tu. "
        f"Nguon uu tien: {config.source_chunk_root}"
    )

    configure_models(config)
    try:
        fact_nodes, fact_records = build_graph_fact_nodes(
            all_records,
            progress_every=config.graph_progress_every,
        )
    except (APIConnectionError, APITimeoutError) as exc:
        raise RuntimeError(
            "Khong ket noi on dinh toi endpoint LLM/embedding. "
            "Hay kiem tra Ollama va tang timeout neu can."
        ) from exc

    if not fact_nodes:
        raise RuntimeError("Khong trich duoc graph fact nao tu tap chunk hien tai.")

    write_fact_records(fact_records, config.facts_path)
    print(f"Da trich {len(fact_nodes)} graph facts vao {config.facts_path}.")

    if args.dry_run:
        print("Dry run hoan tat.")
        print(f"Mau fact: {fact_records[0]['text']}")
        return

    embedded_nodes = attach_embeddings(fact_nodes, Settings.embed_model, batch_size=config.embed_batch_size)
    if not embedded_nodes:
        raise RuntimeError("Khong embed duoc graph fact nao de ghi vao Qdrant.")

    if args.reset_graph:
        print(f"Resetting Qdrant collection `{config.qdrant_collection}`...")
    reset_vector_store(config)
    vector_store = build_vector_store(config)
    vector_store.add(embedded_nodes)
    print(f"Ingest xong {len(embedded_nodes)} graph facts vao Qdrant.")


if __name__ == "__main__":
    main()
