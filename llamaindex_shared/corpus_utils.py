from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from llama_index.core.schema import TextNode


# Cấu hình UTF-8 cho stdin/stdout/stderr (best-effort).
# Mục tiêu: log/CLI hiển thị tiếng Việt ổn định hơn (đặc biệt trên Windows).
def configure_console_utf8() -> None:
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except ValueError:
                # Một số môi trường không cho phép reconfigure (ví dụ stream đã đóng).
                pass


# Liệt kê các file chunk JSONL dưới `chunk_root`.
# Nếu `scope` trùng tên một thư mục con (ví dụ `web`, `2025`) thì chỉ duyệt trong scope đó.
def iter_chunk_jsonl_files(chunk_root: Path, scope: str | None = None, limit: int | None = None) -> list[Path]:
    search_root = chunk_root
    if scope and (chunk_root / scope).exists():
        search_root = chunk_root / scope
    files = sorted(path for path in search_root.rglob("*.jsonl") if path.is_file())
    if limit is None:
        return files
    return files[:limit]


# Nạp chunk records từ nhiều file JSONL và nhóm theo từng file nguồn.
def load_chunk_record_groups(
    *,
    chunk_root: Path,
    scope: str | None = None,
    limit: int | None = None,
) -> list[tuple[Path, list[dict]]]:
    groups: list[tuple[Path, list[dict]]] = []
    for jsonl_path in iter_chunk_jsonl_files(chunk_root=chunk_root, scope=scope, limit=limit):
        records = load_chunk_records_from_jsonl(jsonl_path=jsonl_path, chunk_root=chunk_root)
        if records:
            groups.append((jsonl_path, records))
    return groups


# Parse một file chunk JSONL thành records chuẩn dùng chung cho toàn bộ pipeline.
# Chunk JSONL từ `extract_md` có thể chỉ có schema tối giản. Hàm này chuẩn hóa.
# bổ sung metadata để ổn định cho ingest, UI, và evaluation:
# - `doc_id`, `relative_path`, `source_group`, `source_year`, `record_type`, ...
def load_chunk_records_from_jsonl(*, jsonl_path: Path, chunk_root: Path) -> list[dict]:
    records: list[dict] = []
    relative_jsonl = jsonl_path.relative_to(chunk_root)
    relative_txt = relative_jsonl.with_suffix(".txt")
    source_file = relative_txt.name
    source_group = relative_txt.parts[0] if relative_txt.parts else "unknown"
    source_year = _infer_year(relative_txt)
    fallback_title = source_file.replace("-", " ").replace("_", " ").replace(".txt", "").strip().title()

    with jsonl_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            text = str(raw.get("text") or "").strip()
            if not text:
                continue

            chunk_id = int(raw.get("chunk_id") or index)
            heading_path = raw.get("heading_path") or []
            if not isinstance(heading_path, list):
                heading_path = [str(heading_path)]

            title = str(raw.get("title") or (heading_path[0] if heading_path else fallback_title))
            inferred_record_type = str(raw.get("record_type") or _infer_record_type(text, heading_path))
            inferred_table_heading = str(raw.get("table_heading") or _infer_table_heading(text, heading_path))

            records.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": f"{relative_txt.as_posix()}::chunk-{chunk_id}",
                    "source_path": str(raw.get("source_txt") or jsonl_path),
                    "relative_path": relative_txt.as_posix(),
                    "source_file": source_file,
                    "source_group": source_group,
                    "source_year": source_year,
                    "title": title,
                    "source_url": str(raw.get("source_url") or ""),
                    "heading_path": heading_path,
                    "text": text,
                    "record_type": inferred_record_type,
                    "table_heading": inferred_table_heading,
                    "char_count": int(raw.get("char_count") or len(text)),
                    "line_count": int(raw.get("line_count") or sum(1 for item in text.splitlines() if item.strip())),
                }
            )

    return records


# Chuyển chunk records sang `TextNode` để LlamaIndex/extractor xử lý.
def records_to_nodes(records: list[dict]) -> list[TextNode]:
    nodes: list[TextNode] = []
    for record in records:
        metadata = _build_node_metadata(record)
        nodes.append(
            TextNode(
                id_=record["doc_id"],
                text=record["text"],
                metadata=metadata,
                excluded_embed_metadata_keys=list(metadata.keys()),
                excluded_llm_metadata_keys=list(metadata.keys()),
            )
        )
    return nodes


# Ghi records ra JSONL để audit/debug.
def write_chunk_records(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# Tính thống kê nhanh của tập records để in log ingest.
def summarize_records(records: list[dict]) -> dict[str, int]:
    return {
        "chunk_count": len(records),
        "total_chars": sum(record["char_count"] for record in records),
        "table_chunks": sum(1 for record in records if any("Bang" in item for item in record["heading_path"])),
    }


# Chuẩn hóa metadata của record trước khi gắn vào `TextNode`.
def _build_node_metadata(record: dict) -> dict[str, str | int]:
    return {
        "source_file": record["source_file"],
        "relative_path": record["relative_path"],
        "source_group": record["source_group"],
        "source_year": record["source_year"],
        "title": record["title"],
        "heading_path": " > ".join(record["heading_path"]),
        "source_url": record["source_url"] or "",
        "record_type": record.get("record_type", "section"),
        "table_heading": record.get("table_heading", ""),
        "chunk_id": record["chunk_id"],
        "char_count": record["char_count"],
    }


# Suy ra năm từ relative path (ưu tiên thư mục `20xx`, fallback stem).
def _infer_year(relative_path: Path) -> str:
    for part in relative_path.parts:
        if re.fullmatch(r"20\d{2}", part):
            return part
    stem_match = re.search(r"(20\d{2})", relative_path.stem)
    if stem_match:
        return stem_match.group(1)
    return "unknown"


# Suy ra record là `section` hay `table_row`.
def _infer_record_type(text: str, heading_path: list[str]) -> str:
    if _looks_like_embedded_table_row(text):
        return "table_row"
    if heading_path and heading_path[-1].lower().startswith("bang "):
        return "table_row"
    return "section"


# Suy ra tiêu đề bảng gắn với record nếu record giống dữ liệu bảng.
def _infer_table_heading(text: str, heading_path: list[str]) -> str:
    if not _looks_like_embedded_table_row(text):
        return ""
    return heading_path[-1] if heading_path else ""


# Heuristic nhận diện chunk có nhúng dạng bảng markdown hay không.
def _looks_like_embedded_table_row(text: str) -> bool:
    row_lines = [line.strip() for line in text.splitlines() if line.strip().startswith("- ") and " | " in line]
    if len(row_lines) < 2:
        return False

    header_like = sum(1 for line in row_lines if _is_table_header_line(line))
    data_like = sum(1 for line in row_lines if not _is_table_header_line(line))
    return header_like >= 1 and data_like >= 1


# Nhận diện dòng header trong bảng markdown.
def _is_table_header_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("- TT |") or stripped.startswith("- STT |") or stripped.startswith("- (")
