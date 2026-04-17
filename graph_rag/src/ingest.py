from __future__ import annotations

import argparse
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from llama_index.core import Settings
from openai import APIConnectionError, APITimeoutError

from config import build_neo4j_driver, configure_models, load_config, verify_neo4j_connection
from fact_graph import build_graph_artifacts, write_fact_records
from neo4j_store import ensure_schema, reset_graph, upsert_graph_artifacts
from utils import (
    configure_console_utf8,
    load_chunk_record_groups,
    summarize_records,
    write_chunk_records,
)


# Embed fact text theo batch và tự chia nhỏ batch khi gặp lỗi để ingest không gãy giữa chừng.
def attach_fact_embeddings(facts: list[dict], embed_model, batch_size: int) -> list[dict]:
    """Gắn embedding cho từng fact trước khi ghi vào Neo4j."""

    embedded_facts: list[dict] = []

    # Thử embed nguyên một batch; nếu lỗi thì chia đôi batch để cứu phần dữ liệu còn lại.
    def _embed_batch(batch_rows: list[dict]) -> None:
        """Embed một batch fact và tự thu nhỏ batch nếu backend embedding không ổn định."""

        if not batch_rows:
            return

        texts = [str(row["text"])[:3000] for row in batch_rows]
        try:
            embeddings = embed_model.get_text_embedding_batch(texts)
            for row, embedding in zip(batch_rows, embeddings, strict=True):
                row["embedding"] = list(embedding)
                embedded_facts.append(row)
            return
        except Exception:
            if len(batch_rows) == 1:
                try:
                    batch_rows[0]["embedding"] = list(embed_model.get_text_embedding(texts[0]))
                    embedded_facts.append(batch_rows[0])
                except Exception:
                    pass
                return

        midpoint = len(batch_rows) // 2
        _embed_batch(batch_rows[:midpoint])
        _embed_batch(batch_rows[midpoint:])

    for index in range(0, len(facts), batch_size):
        _embed_batch(facts[index : index + batch_size])

    return embedded_facts


# Khai báo tham số CLI cho pipeline ingest GraphRAG dùng chung trong terminal.
def parse_args() -> argparse.Namespace:
    """Parse tham số CLI cho ingest từ shared chunks."""

    parser = argparse.ArgumentParser(description="Ingest GraphRAG từ shared chunk JSONL vào Neo4j.")
    parser.add_argument("--chunk-jsonl-root", type=Path, help="Folder shared chunk JSONL để dùng đúng chunk baseline.")
    parser.add_argument("--output-dir", type=Path, help="Folder lưu JSONL audit cục bộ cho GraphRAG.")
    parser.add_argument("--limit", type=int, help="Chỉ xử lý N file đầu tiên.")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ trích fact và ghi audit, không ghi vào Neo4j.")
    parser.add_argument(
        "--reset-graph",
        action="store_true",
        help="Xóa toàn bộ graph Neo4j trước khi ingest lại.",
    )
    return parser.parse_args()


# Nạp shared chunks từ extract_md để GraphRAG luôn đi cùng corpus với hai hệ còn lại.
def _load_records(args: argparse.Namespace, config) -> list[dict]:
    """Đọc chunk JSONL dùng chung từ pipeline chuẩn của project."""

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
        f"Không tìm thấy shared chunk JSONL trong {chunk_jsonl_root} (scope={config.corpus_scope}). "
        "Hãy tạo/cập nhật corpus chunk trước khi ingest GraphRAG."
    )


# Chuẩn hóa danh sách tài liệu thô do UI gửi lên trước khi chunk và ingest.
def _prepare_documents(documents: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Lọc tài liệu rỗng, chuẩn hóa tên nguồn và tránh trùng tên file đầu ra."""

    prepared: list[dict[str, str]] = []
    seen_names: dict[str, int] = {}

    for index, document in enumerate(documents, start=1):
        raw_name = str(document.get("name") or f"tai-lieu-{index}.md").strip()
        raw_content = str(document.get("content") or "").strip()
        if not raw_content:
            continue

        safe_name = _slugify_source_name(raw_name or f"tai-lieu-{index}.md", index)
        seen_names[safe_name] = seen_names.get(safe_name, 0) + 1
        duplicate_count = seen_names[safe_name]
        if duplicate_count > 1:
            path = Path(safe_name)
            safe_name = f"{path.stem}-{duplicate_count}{path.suffix or '.md'}"

        prepared.append(
            {
                "name": raw_name or safe_name,
                "safe_name": safe_name,
                "content": raw_content,
            }
        )

    return prepared


# Chuẩn hóa tên file để dữ liệu audit ghi ra đĩa ổn định và không chứa ký tự nguy hiểm.
def _slugify_source_name(name: str, fallback_index: int) -> str:
    """Chuyển tên nguồn bất kỳ thành tên file an toàn cho thư mục processed."""

    path = Path(name)
    suffix = path.suffix or ".md"
    stem = path.stem or f"tai-lieu-{fallback_index}"
    stem = re.sub(r"[^\w\-]+", "-", stem, flags=re.UNICODE).strip("-_")
    stem = stem or f"tai-lieu-{fallback_index}"
    return f"{stem}{suffix.lower()}"


# Suy ra năm nguồn để metadata của GraphRAG vẫn giữ được ngữ cảnh theo năm tuyển sinh.
def _extract_source_year(source_name: str, content: str) -> str:
    """Suy ra năm 20xx từ tên nguồn hoặc nội dung đầu vào nếu có."""

    for candidate in (source_name, content[:400]):
        matched = re.search(r"(20\d{2})", candidate)
        if matched:
            return matched.group(1)
    return "unknown"


# Tách văn bản thô theo heading Markdown để giữ lại phần nào cấu trúc tài liệu gốc.
def _split_document_sections(content: str) -> list[tuple[list[str], str]]:
    """Tách tài liệu thành các section dựa trên heading Markdown."""

    normalized = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    sections: list[tuple[list[str], str]] = []
    heading_stack: list[str] = []
    current_heading: list[str] = []
    current_lines: list[str] = []

    # Đẩy phần nội dung đang gom vào danh sách section trước khi chuyển heading.
    def _flush_current_section() -> None:
        """Đưa section hiện tại vào danh sách kết quả nếu có nội dung hữu ích."""

        section_text = "\n".join(current_lines).strip()
        if section_text:
            sections.append((current_heading.copy(), section_text))

    for raw_line in normalized.split("\n"):
        heading_match = re.match(r"^(#{1,6})\s+(.*)$", raw_line.strip())
        if heading_match:
            _flush_current_section()
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(title)
            current_heading = heading_stack.copy()
            current_lines = []
            continue
        current_lines.append(raw_line.rstrip())

    _flush_current_section()
    if sections:
        return sections
    return [([], normalized)]


# Cắt một đoạn quá dài thành nhiều đoạn nhỏ hơn khi một paragraph vượt quá ngưỡng chunk.
def _split_oversized_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Tách paragraph quá dài theo câu, và fallback về cắt cứng nếu cần."""

    cleaned = paragraph.strip()
    if len(cleaned) <= max_chars:
        return [cleaned]

    sentence_like_parts = [part.strip() for part in re.split(r"(?<=[.!?;:])\s+", cleaned) if part.strip()]
    if len(sentence_like_parts) <= 1:
        return [cleaned[start : start + max_chars].strip() for start in range(0, len(cleaned), max_chars)]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_length = 0
    for sentence in sentence_like_parts:
        if len(sentence) > max_chars:
            if current_parts:
                chunks.append(" ".join(current_parts).strip())
                current_parts = []
                current_length = 0
            chunks.extend(_split_oversized_paragraph(sentence, max_chars))
            continue

        projected = current_length + len(sentence) + (1 if current_parts else 0)
        if current_parts and projected > max_chars:
            chunks.append(" ".join(current_parts).strip())
            current_parts = [sentence]
            current_length = len(sentence)
            continue

        current_parts.append(sentence)
        current_length = projected

    if current_parts:
        chunks.append(" ".join(current_parts).strip())
    return chunks


# Gom paragraph thành chunk theo ngưỡng min/max để extractor có đầu vào ổn định hơn.
def _chunk_section_text(section_text: str, min_chars: int, max_chars: int) -> list[str]:
    """Chia section thành các chunk gần với cấu hình chunk chuẩn của project."""

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", section_text) if part.strip()]
    if not paragraphs:
        paragraphs = [section_text.strip()]

    normalized_parts: list[str] = []
    for paragraph in paragraphs:
        normalized_parts.extend(_split_oversized_paragraph(paragraph, max_chars))

    chunks: list[str] = []
    current_parts: list[str] = []
    current_length = 0

    for part in normalized_parts:
        projected = current_length + len(part) + (2 if current_parts else 0)
        if current_parts and projected > max_chars and current_length >= min_chars:
            chunks.append("\n\n".join(current_parts).strip())
            current_parts = [part]
            current_length = len(part)
            continue

        if current_parts and projected > max_chars:
            chunks.append("\n\n".join(current_parts + [part]).strip())
            current_parts = []
            current_length = 0
            continue

        current_parts.append(part)
        current_length = projected

    if current_parts:
        chunks.append("\n\n".join(current_parts).strip())

    return [chunk for chunk in chunks if chunk]


# Chuyển tài liệu thô người dùng vừa nạp thành chunk record cùng schema với shared chunks.
def _build_raw_chunk_records(documents: list[dict[str, str]], config) -> list[dict]:
    """Sinh chunk records mới hoàn toàn từ văn bản thô do UI gửi lên."""

    records: list[dict] = []

    for document in documents:
        safe_name = document["safe_name"]
        relative_path = Path("uploads") / (Path(safe_name).stem + ".txt")
        default_title = Path(safe_name).stem.replace("-", " ").replace("_", " ").strip().title() or "Tài liệu mới"
        source_year = _extract_source_year(document["name"], document["content"])
        chunk_id = 1

        for heading_path, section_text in _split_document_sections(document["content"]):
            chunk_texts = _chunk_section_text(
                section_text,
                min_chars=config.chunk_min_chars,
                max_chars=config.chunk_max_chars,
            )
            for chunk_text in chunk_texts:
                title = heading_path[-1] if heading_path else default_title
                line_count = sum(1 for line in chunk_text.splitlines() if line.strip())
                records.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": f"{relative_path.as_posix()}::chunk-{chunk_id}",
                        "source_path": f"ui://{safe_name}",
                        "relative_path": relative_path.as_posix(),
                        "source_file": relative_path.name,
                        "source_group": "uploads",
                        "source_year": source_year,
                        "title": title,
                        "source_url": "",
                        "heading_path": heading_path,
                        "text": chunk_text,
                        "record_type": "section",
                        "table_heading": "",
                        "char_count": len(chunk_text),
                        "line_count": line_count,
                    }
                )
                chunk_id += 1

    return records


# Xóa dữ liệu audit cũ trong thư mục processed để lần ingest mới chỉ phản ánh dữ liệu vừa nạp.
def _reset_processed_outputs(config) -> None:
    """Dọn riêng vùng audit `uploads` trước khi ingest từ dữ liệu thô mới."""

    uploads_dir = config.chunk_dir / "uploads"
    if uploads_dir.exists():
        shutil.rmtree(uploads_dir)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    if config.facts_path.exists():
        config.facts_path.unlink()
    config.facts_path.parent.mkdir(parents=True, exist_ok=True)


# Ghi lại chunk audit theo từng nguồn để người dùng có thể kiểm tra dữ liệu mới đã được chunk ra sao.
def _write_raw_chunk_audit(records: list[dict], config) -> None:
    """Lưu chunk records từ dữ liệu thô mới ra các file JSONL audit."""

    grouped_records: dict[str, list[dict]] = {}
    for record in records:
        grouped_records.setdefault(record["relative_path"], []).append(record)

    for relative_path, grouped in grouped_records.items():
        output_path = config.chunk_dir / Path(relative_path).with_suffix(".jsonl")
        write_chunk_records(grouped, output_path)


# Tạo fact fallback trực tiếp từ chunk gốc để GraphRAG vẫn ingest được khi extractor/embedding relation bị lỗi.
def _build_runtime_fallback_facts(records: list[dict]) -> list[dict]:
    """Sinh fact fallback bám theo từng chunk gốc, dùng làm lưới an toàn cho ingest runtime."""

    fallback_facts: list[dict] = []
    for record in records:
        chunk_text = str(record.get("text") or "").strip()
        if not chunk_text:
            continue

        relative_path = str(record.get("relative_path") or record.get("source_file") or "unknown")
        chunk_id = str(record.get("chunk_id") or "unknown")
        fact_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{relative_path}::chunk::{chunk_id}::runtime-fallback"))
        fallback_facts.append(
            {
                "id": fact_id,
                "text": chunk_text,
                "relative_path": relative_path,
                "source_file": str(record.get("source_file") or Path(relative_path).name),
                "heading_path": " | ".join(str(item) for item in (record.get("heading_path") or []) if str(item).strip()),
                "chunk_id": chunk_id,
                "source_year": str(record.get("source_year") or ""),
                "relation_label": "RUNTIME_FALLBACK_CHUNK",
                "source_entity": "",
                "target_entity": "",
                "source_type": "",
                "target_type": "",
                "fact_type": "runtime_chunk_fallback",
                "source_fact_id": "",
                "source_uid": "",
                "target_uid": "",
                "chunk_uid": f"{relative_path}::chunk::{chunk_id}",
                "edge_uid": "",
            }
        )
    return fallback_facts


# Chạy trọn pipeline trích fact, embed và ghi vào Neo4j từ một tập chunk records đã có sẵn.
def _run_ingest_pipeline(
    *,
    all_records: list[dict],
    config,
    reset_graph_first: bool,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Thực thi pipeline ingest dùng chung cho cả CLI và UI."""

    if not all_records:
        raise ValueError("Không tạo được chunk nào từ dữ liệu đầu vào.")

    summary = summarize_records(all_records)
    print(
        f"Tạo {summary['chunk_count']} chunks, tổng {summary['total_chars']} ký tự. "
        f"Nguồn ưu tiên: {config.source_chunk_root}"
    )

    configure_models(config)
    try:
        artifacts, fact_records = build_graph_artifacts(
            all_records,
            progress_every=config.graph_progress_every,
        )
    except (APIConnectionError, APITimeoutError) as exc:
        raise RuntimeError(
            "Không kết nối ổn định tới endpoint LLM/embedding. "
            "Hãy kiểm tra Ollama và tăng timeout nếu cần."
        ) from exc

    if not artifacts.facts:
        raise RuntimeError("Không trích được graph fact nào từ tập chunk hiện tại.")

    write_fact_records(fact_records, config.facts_path)
    print(f"Đã trích {len(artifacts.facts)} graph facts vào {config.facts_path}.")

    if dry_run:
        return {
            "chunk_count": summary["chunk_count"],
            "fact_count": len(artifacts.facts),
            "entity_count": len(artifacts.entities),
            "table_chunks": summary["table_chunks"],
            "dry_run": True,
        }

    verify_neo4j_connection(config)
    embedded_facts = attach_fact_embeddings(
        artifacts.facts,
        Settings.embed_model,
        batch_size=config.embed_batch_size,
    )
    if not embedded_facts:
        print("Không embed được fact trích xuất; đang fallback sang chunk fact gốc...")
        embedded_facts = attach_fact_embeddings(
            _build_runtime_fallback_facts(all_records),
            Settings.embed_model,
            batch_size=1,
        )
    if not embedded_facts:
        raise RuntimeError("Không embed được graph fact nào để ghi vào Neo4j.")

    artifacts.facts = embedded_facts
    embedding_dim = len(embedded_facts[0]["embedding"])

    driver = build_neo4j_driver(config)
    try:
        ensure_schema(driver, config, embedding_dim=embedding_dim)
        if reset_graph_first:
            print("Đang xóa graph Neo4j cũ trước khi ingest lại...")
            reset_graph(driver, config)
        upsert_graph_artifacts(driver, config, artifacts)
    finally:
        driver.close()

    return {
        "chunk_count": summary["chunk_count"],
        "fact_count": len(artifacts.facts),
        "entity_count": len(artifacts.entities),
        "table_chunks": summary["table_chunks"],
        "dry_run": False,
    }


# Nhận danh sách tài liệu thô mới từ UI, ghi đè processed audit cũ và reset graph để ingest lại từ đầu.
def ingest_raw_documents(documents: list[dict[str, Any]], runtime_overrides: dict | None = None) -> dict[str, Any]:
    """Xử lý dữ liệu thô mới ngay trên UI và thay thế hoàn toàn graph hiện tại."""

    config = load_config(overrides=runtime_overrides)
    prepared_documents = _prepare_documents(documents)
    if not prepared_documents:
        raise ValueError("Không có dữ liệu thô hợp lệ để xử lý.")

    raw_records = _build_raw_chunk_records(prepared_documents, config)
    if not raw_records:
        raise ValueError("Không thể tạo chunk từ dữ liệu thô mới.")

    _reset_processed_outputs(config)
    _write_raw_chunk_audit(raw_records, config)

    summary = _run_ingest_pipeline(
        all_records=raw_records,
        config=config,
        reset_graph_first=True,
        dry_run=False,
    )
    summary["document_count"] = len(prepared_documents)
    summary["message"] = (
        f"Đã ingest {summary['document_count']} nguồn mới, tạo {summary['chunk_count']} chunk, "
        f"{summary['entity_count']} entity và {summary['fact_count']} fact."
    )
    return summary


# Pipeline ingest mới cho GraphRAG dùng Neo4j làm graph database chuyên dụng.
def main() -> None:
    """Điểm vào CLI cho pipeline ingest GraphRAG."""

    # Bước 1: cấu hình UTF-8 để log tiếng Việt hiển thị đúng trên terminal.
    configure_console_utf8()
    # Bước 2: đọc tham số CLI và nạp cấu hình GraphRAG/Neo4j.
    args = parse_args()
    config = load_config()
    # Bước 3: nạp shared chunks từ extract_md và ghi lại audit cục bộ cho GraphRAG.
    all_records = _load_records(args, config)
    # Bước 4: chạy pipeline trích fact, embed và ghi vào Neo4j.
    summary = _run_ingest_pipeline(
        all_records=all_records,
        config=config,
        reset_graph_first=args.reset_graph,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("Dry run hoàn tất.")
        return

    print(
        f"Ingest xong {summary['entity_count']} entity, {summary['chunk_count']} chunk "
        f"và {summary['fact_count']} fact vào Neo4j."
    )


if __name__ == "__main__":
    main()
