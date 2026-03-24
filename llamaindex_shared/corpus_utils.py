from __future__ import annotations

import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from llama_index.core.schema import TextNode


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TABLE_HEADING_RE = re.compile(r"^#{2,6}\s+Bang\s+\d+\s*$", re.IGNORECASE)


@dataclass
class Section:
    heading: str | None
    lines: list[str]

    @property
    # Ghép heading và nội dung section thành một khối text hoàn chỉnh.
    def text(self) -> str:
        parts: list[str] = []
        if self.heading:
            parts.append(self.heading)
        parts.extend(self.lines)
        return "\n".join(parts).strip()


# Cấu hình UTF-8 cho stdin/stdout/stderr để log và CLI hiển thị tiếng Việt ổn định hơn.
def configure_console_utf8() -> None:
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except ValueError:
                pass


# Chuẩn hóa Unicode, xuống dòng và khoảng trắng trước khi tách section/chunk.
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(_collapse_blank_lines(lines)).strip()


# Gộp nhiều dòng trống liên tiếp thành một dòng trống duy nhất.
def _collapse_blank_lines(lines: list[str]) -> list[str]:
    output: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and previous_blank:
            continue
        output.append(line)
        previous_blank = is_blank
    while output and not output[-1].strip():
        output.pop()
    return output


# Liệt kê file TXT đầu vào trong thư mục corpus.
def iter_txt_files(txt_root: Path, limit: int | None = None) -> list[Path]:
    files = sorted(path for path in txt_root.rglob("*.txt") if path.is_file())
    if limit is None:
        return files
    return files[:limit]


# Liệt kê file JSONL chunk để ingest lại mà không cần chunk từ TXT.
def iter_chunk_jsonl_files(chunk_root: Path, scope: str | None = None, limit: int | None = None) -> list[Path]:
    search_root = chunk_root
    if scope and (chunk_root / scope).exists():
        search_root = chunk_root / scope
    files = sorted(path for path in search_root.rglob("*.jsonl") if path.is_file())
    if limit is None:
        return files
    return files[:limit]


# Tách tài liệu thành các section theo heading markdown và giữ section bảng riêng biệt.
def split_sections(text: str) -> list[Section]:
    sections: list[Section] = []
    current_heading: str | None = None
    current_lines: list[str] = []
    current_is_table = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if HEADING_RE.match(line):
            if current_heading is not None or current_lines:
                sections.append(Section(heading=current_heading, lines=current_lines))
            current_heading = line
            current_lines = []
            current_is_table = _is_table_heading(line)
            continue

        if current_is_table:
            if not line.strip():
                continue
            if _is_table_row(line):
                current_lines.append(line)
                continue
            sections.append(Section(heading=current_heading, lines=current_lines))
            current_heading = None
            current_lines = [line]
            current_is_table = False
            continue

        current_lines.append(line)

    if current_heading is not None or current_lines:
        sections.append(Section(heading=current_heading, lines=current_lines))

    return [section for section in sections if section.text]


# Gộp các section quá ngắn liền kề để tạo chunk có ngữ cảnh tốt hơn.
def merge_sections(
    sections: list[Section],
    min_chars: int,
    max_sections_per_chunk: int,
) -> list[Section]:
    merged: list[Section] = []
    index = 0
    while index < len(sections):
        section = sections[index]
        heading = section.heading
        lines = list(section.lines)
        used = 1
        index += 1

        current = Section(heading=heading, lines=lines)
        while index < len(sections) and len(current.text) < min_chars and used < max_sections_per_chunk:
            next_section = sections[index]
            if _is_table_section(current) or _is_table_section(next_section):
                break
            lines.append("")
            if next_section.heading:
                lines.append(next_section.heading)
            lines.extend(next_section.lines)
            current = Section(heading=heading, lines=lines)
            used += 1
            index += 1

        merged.append(current)

    return merged


# Chia một section quá dài thành nhiều đoạn nhỏ hơn theo giới hạn ký tự.
def split_large_section(section: Section, max_chars: int) -> list[str]:
    text = section.text
    if len(text) <= max_chars:
        return [text]

    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    paragraph_chunks: list[str] = []
    buffer: list[str] = []
    for block in blocks:
        candidate = "\n\n".join(buffer + [block]).strip()
        if buffer and len(candidate) > max_chars:
            paragraph_chunks.append("\n\n".join(buffer).strip())
            buffer = [block]
        else:
            buffer.append(block)
    if buffer:
        paragraph_chunks.append("\n\n".join(buffer).strip())
    if paragraph_chunks and all(len(chunk) <= max_chars for chunk in paragraph_chunks):
        return paragraph_chunks

    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    context_heading = lines[0] if HEADING_RE.match(lines[0]) else None
    remaining = lines[1:] if context_heading else lines
    chunks: list[str] = []
    current_lines = [context_heading] if context_heading else []
    for line in remaining:
        candidate = "\n".join(current_lines + [line]).strip()
        if current_lines and len(candidate) > max_chars:
            chunks.append("\n".join(current_lines).strip())
            current_lines = [context_heading, line] if context_heading else [line]
        else:
            current_lines.append(line)
    if current_lines:
        chunks.append("\n".join(current_lines).strip())
    return [chunk for chunk in chunks if chunk]


# Chuyển nội dung tài liệu thô thành danh sách chunk records chuẩn hóa.
def build_chunk_records(
    text: str,
    source_path: Path,
    txt_root: Path,
    min_chars: int,
    max_chars: int,
    max_sections_per_chunk: int,
) -> list[dict]:
    normalized = normalize_text(text)
    sections = split_sections(normalized)
    if not sections:
        return []

    merged_sections = merge_sections(
        sections=sections,
        min_chars=min_chars,
        max_sections_per_chunk=max_sections_per_chunk,
    )

    title = _extract_title(normalized, source_path.stem)
    source_url = _extract_source_url(normalized)
    relative_path = source_path.relative_to(txt_root)

    records: list[dict] = []
    chunk_id = 1
    for section in merged_sections:
        for chunk_info in iter_section_chunks(section, max_chars=max_chars):
            chunk_text = chunk_info["text"]
            heading_path = _extract_heading_path(chunk_text)
            records.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": f"{relative_path.as_posix()}::chunk-{chunk_id}",
                    "source_path": str(source_path),
                    "relative_path": relative_path.as_posix(),
                    "source_file": source_path.name,
                    "source_group": relative_path.parts[0] if relative_path.parts else "unknown",
                    "source_year": _infer_year(relative_path),
                    "title": title,
                    "source_url": source_url,
                    "heading_path": heading_path,
                    "text": chunk_text,
                    "record_type": chunk_info["record_type"],
                    "table_heading": chunk_info["table_heading"],
                    "char_count": len(chunk_text),
                    "line_count": sum(1 for line in chunk_text.splitlines() if line.strip()),
                }
            )
            chunk_id += 1
    return records


# Đọc một file TXT rồi sinh chunk records cho file đó.
def chunk_txt_file(
    source_path: Path,
    txt_root: Path,
    min_chars: int,
    max_chars: int,
    max_sections_per_chunk: int,
) -> list[dict]:
    text = source_path.read_text(encoding="utf-8", errors="ignore")
    return build_chunk_records(
        text=text,
        source_path=source_path,
        txt_root=txt_root,
        min_chars=min_chars,
        max_chars=max_chars,
        max_sections_per_chunk=max_sections_per_chunk,
    )


# Nạp nhiều file JSONL chunk và nhóm chúng theo từng file nguồn.
def load_chunk_record_groups(
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


# Parse một file JSONL chunk thành records chuẩn dùng chung trong toàn bộ pipeline.
def load_chunk_records_from_jsonl(jsonl_path: Path, chunk_root: Path) -> list[dict]:
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


# Chuyển chunk records sang `TextNode` để LlamaIndex và extractor xử lý.
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


# Ghi chunk records ra JSONL để audit hoặc debug bước chunking.
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


# Lấy tiêu đề tài liệu từ heading cấp 1 hoặc fallback theo tên file.
def _extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match and len(match.group(1)) == 1:
            return match.group(2).strip()
    return fallback.replace("-", " ").replace("_", " ").strip().title()


# Trích source URL nếu tài liệu có dòng `Source:`.
def _extract_source_url(text: str) -> str | None:
    for line in text.splitlines():
        if line.lower().startswith("source:"):
            return line.split(":", 1)[1].strip() or None
    return None


# Trích đường dẫn heading xuất hiện trong một chunk.
def _extract_heading_path(chunk_text: str) -> list[str]:
    headings: list[str] = []
    for line in chunk_text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            headings.append(match.group(2).strip())
    return headings


# Suy ra năm từ relative path hoặc tên file nguồn.
def _infer_year(relative_path: Path) -> str:
    for part in relative_path.parts:
        if re.fullmatch(r"20\d{2}", part):
            return part
    stem_match = re.search(r"(20\d{2})", relative_path.stem)
    if stem_match:
        return stem_match.group(1)
    return "unknown"


# Suy ra record là section thường hay table row.
def _infer_record_type(text: str, heading_path: list[str]) -> str:
    if _looks_like_embedded_table_row(text):
        return "table_row"
    if heading_path and heading_path[-1].lower().startswith("bang "):
        return "table_row"
    return "section"


# Suy ra tiêu đề bảng gắn với chunk nếu chunk có dạng bảng.
def _infer_table_heading(text: str, heading_path: list[str]) -> str:
    if not _looks_like_embedded_table_row(text):
        return ""
    return heading_path[-1] if heading_path else ""


# Kiểm tra một chunk text có giống dữ liệu bảng nhúng hay không.
def _looks_like_embedded_table_row(text: str) -> bool:
    row_lines = [line.strip() for line in text.splitlines() if line.strip().startswith("- ") and " | " in line]
    if len(row_lines) < 2:
        return False

    header_like = sum(1 for line in row_lines if _is_table_header_line(line))
    data_like = sum(1 for line in row_lines if not _is_table_header_line(line))
    return header_like >= 1 and data_like >= 1


# Nhận diện heading của một bảng markdown.
def _is_table_heading(line: str) -> bool:
    return bool(TABLE_HEADING_RE.match(line.strip()))


# Nhận diện một dòng dữ liệu bảng.
def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("- ") and " | " in stripped


# Kiểm tra một section có phải section bảng hoàn chỉnh hay không.
def _is_table_section(section: Section) -> bool:
    return bool(section.heading) and _is_table_heading(section.heading) and bool(section.lines) and all(
        _is_table_row(line) for line in section.lines
    )


# Sinh danh sách chunk từ một section, bao gồm cả chế độ split bảng theo từng dòng.
def iter_section_chunks(section: Section, max_chars: int) -> list[dict[str, str]]:
    if _is_table_section(section):
        return _split_table_section(section)

    return [
        {
            "text": chunk_text,
            "record_type": "section",
            "table_heading": "",
        }
        for chunk_text in split_large_section(section, max_chars=max_chars)
    ]


# Tách một section bảng thành nhiều records để extractor bắt được header và từng hàng dữ liệu.
def _split_table_section(section: Section) -> list[dict[str, str]]:
    if not section.heading:
        return []

    header_lines: list[str] = []
    active_group_line = ""
    chunk_records: list[dict[str, str]] = []

    for line in section.lines:
        if _is_table_header_line(line):
            header_lines.append(line)
            continue

        if _is_table_group_line(line):
            active_group_line = line
            continue

        chunk_lines = [section.heading]
        chunk_lines.extend(header_lines[:2])
        if active_group_line:
            chunk_lines.append(active_group_line)
        chunk_lines.append(line)
        chunk_records.append(
            {
                "text": "\n".join(chunk_lines).strip(),
                "record_type": "table_row",
                "table_heading": section.heading.lstrip("# ").strip(),
            }
        )

    if chunk_records:
        return chunk_records

    return [
        {
            "text": section.text,
            "record_type": "section",
            "table_heading": section.heading.lstrip("# ").strip(),
        }
    ]


# Nhận diện dòng header trong section bảng.
def _is_table_header_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("- TT |") or stripped.startswith("- STT |") or stripped.startswith("- (")


# Nhận diện dòng phân nhóm trong bảng để gắn ngữ cảnh cho các hàng bên dưới.
def _is_table_group_line(line: str) -> bool:
    if not _is_table_row(line):
        return False

    first_cell = line[2:].split("|", 1)[0].strip()
    if re.fullmatch(r"[IVX]+", first_cell):
        return True
    if re.fullmatch(r"\d+\.\d+", first_cell):
        non_empty = [part.strip() for part in line[2:].split("|") if part.strip() and part.strip() != "-"]
        return len(non_empty) <= 3
    return False
