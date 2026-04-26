import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import date
from itertools import zip_longest
from pathlib import Path
from typing import TypeAlias


# Tiện ích lõi cho pipeline Extract MD:
# - Chuẩn hóa văn bản (Unicode, bullet, xuống dòng)
# - Phân đoạn theo heading/section và gộp/tách theo độ dài
# - Tạo dữ liệu chunk JSONL (dùng cho ingest/RAG)
SPACE_COMBINING_RE = re.compile(r"([ \t])[\u0300-\u036f]+")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
QUESTION_TOKEN_RE = re.compile(r"^Q\d+$", re.IGNORECASE)
ROMAN_TOKEN_RE = re.compile(r"^[IVXLCDM]+$", re.IGNORECASE)
NUMERIC_TOKEN_RE = re.compile(r"^\d+(?:\.\d+)*$")
HEADING_PREFIX_RE = re.compile(
    r"^(?P<token>(?:Q\d+|[IVXLCDM]+|\d+(?:\.\d+)*))(?:[.)])?\s+(?P<body>.+)$",
    re.IGNORECASE,
)
TABLE_SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")
TABLE_HEADING_ONLY_RE = re.compile(r"^#{2,6}\s+Bang\s+\d+\s*$", re.IGNORECASE)
NUMBERED_ITEM_RE = re.compile(r"\b\d+\.\s+")
INLINE_FACT_LINE_RE = re.compile(r"^(?P<key>[^:\n]{2,80}):\s*(?P<value>.+?)\s*$")
PAGE_NOISE_PATTERNS = [
    re.compile(r"^\s*trang\s*\d+\s*(/\s*\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*page\s*\d+\s*(/\s*\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),
]
SPECIAL_BULLET_CHARS = "\u2022\uf086\uf0a7\uf09f\uf0d8\uf0fc\uf02d\uf02b\uf076\uf0e0\uf071\u27a2"
LEADING_SPECIAL_BULLET_RE = re.compile(rf"^\s*[{re.escape(SPECIAL_BULLET_CHARS)}]+\s*")
SPECIAL_CHAR_RE = re.compile(rf"[{re.escape(SPECIAL_BULLET_CHARS)}]")
LOW_VALUE_TABLE_MARKERS = (
    "họ tên",
    "trình độ, học vị",
    "chuyên môn được đào tạo",
    "ngành tham gia giảng dạy",
)
HeadingNode: TypeAlias = tuple[int, str]


@dataclass
class Section:
    headings: list[HeadingNode]
    lines: list[str]
    is_table: bool = False

    @property
    def heading_path(self) -> list[str]:
        return [text for _, text in self.headings]

    @property
    def document_title(self) -> str:
        return self.heading_path[0] if self.heading_path else ""

    # Render heading context của section, bỏ qua heading root của tài liệu.
    def render_body(self, skip_heading_count: int = 1) -> str:
        parts: list[str] = []
        for level, text in self.headings[skip_heading_count:]:
            parts.append(f"{'#' * level} {text}")
        if self.lines:
            if parts:
                parts.append("")
            parts.extend(self.lines)
        return "\n".join(parts).strip()

    @property
    def text(self) -> str:
        return self.render_body()


# Chuyển chuỗi bất kỳ thành slug lowercase dùng cho tên file.
def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return re.sub(r"-+", "-", slug).strip("-")


# Đoán năm (YYYY) từ path, fallback theo tên file hoặc năm hiện tại.
def guess_year(path: Path) -> str:
    for part in reversed(path.parts):
        if re.fullmatch(r"20\d{2}", part):
            return part
    match = re.search(r"(20\d{2})", path.stem)
    if match:
        return match.group(1)
    return str(date.today().year)


# Chuyển stem của file thành tiêu đề dễ đọc.
def source_title_from_stem(stem: str) -> str:
    return re.sub(r"[-_]+", " ", stem).strip().title()


# Chuẩn hóa một dòng text: Unicode, bullet đặc biệt, khoảng trắng.
def normalize_line(line: str) -> str:
    cleaned = unicodedata.normalize("NFC", line)
    cleaned = cleaned.replace("\ufeff", "").replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    if LEADING_SPECIAL_BULLET_RE.match(cleaned):
        cleaned = LEADING_SPECIAL_BULLET_RE.sub("- ", cleaned)
    else:
        cleaned = SPECIAL_CHAR_RE.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


# Chuẩn hóa toàn bộ văn bản và xử lý xuống dòng.
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = SPACE_COMBINING_RE.sub(r"\1", text)
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()

# Normalize + bỏ dấu chuỗi
def fold_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char)).lower()


# Gộp nhiều dòng trống liên tiếp thành tối đa 1 dòng trống.
def collapse_blank_lines(lines: list[str]) -> list[str]:
    output: list[str] = []
    previous_blank = False
    for line in lines:
        cleaned = line.rstrip()
        is_blank = not cleaned.strip()
        if is_blank and previous_blank:
            continue
        output.append(cleaned)
        previous_blank = is_blank
    while output and not output[-1].strip():
        output.pop()
    return output

# Chuẩn hóa cấp độ heading Markdown (#, ##, …)
def normalize_heading_hierarchy(lines: list[str]) -> list[str]:
    normalized: list[str] = []
    previous_level = 1
    first_heading_seen = False

    for raw_line in lines:
        line = raw_line.rstrip()
        match = HEADING_RE.match(line)
        if not match:
            normalized.append(line)
            continue

        original_level = len(match.group(1))
        heading_text = match.group(2).strip()

        if not first_heading_seen:
            level = 1 if original_level == 1 else 2
            first_heading_seen = True
        else:
            level = min(original_level, previous_level + 1)
            level = max(2, level)

        normalized.append(f"{'#' * level} {heading_text}")
        previous_level = level

    return normalized

# Xóa các dòng trống ở đầu và cuối danh sách
def trim_blank_edges(lines: list[str]) -> list[str]:
    start = 0
    end = len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]

# Dùng để nhận diện một dòng có dạng key: value ngắn
def is_inline_fact_heading(text: str) -> bool:
    return ":" in text and len(text) <= 120


# Đóng gói tiêu đề + nội dung thành tài liệu TXT hoàn chỉnh.
def finalize_document(title: str, body_lines: list[str]) -> str:
    lines = [f"# {title}", *normalize_heading_hierarchy(body_lines)]
    text = normalize_text("\n".join(collapse_blank_lines(lines)))
    return text + "\n"


# Nhận diện heading theo token (Qn, La Mã, số thứ tự) và suy ra level.
def extract_heading_info(text: str) -> dict | None:
    match = HEADING_PREFIX_RE.match(text)
    if not match:
        return None

    token = match.group("token").rstrip(".)")
    body = match.group("body").strip()
    if not body:
        return None
    if QUESTION_TOKEN_RE.fullmatch(token):
        return {"kind": "question", "level": 2, "text": body}
    if ROMAN_TOKEN_RE.fullmatch(token):
        return {"kind": "roman", "level": 2, "text": body}
    if NUMERIC_TOKEN_RE.fullmatch(token):
        return {"kind": "numeric", "level": min(token.count(".") + 3, 6), "text": body}
    return None


# Suy ra cấp độ heading từ nội dung dòng text.
def heading_level_from_text(text: str) -> int | None:
    info = extract_heading_info(text)
    if info:
        return info["level"]

    words = text.split()
    if 1 <= len(words) <= 14 and len(text) <= 100 and text == text.upper() and not text.endswith("."):
        return 2
    return None


# Chuyển một dòng text thành heading markdown nếu hợp lệ.
def classify_heading_line(text: str) -> str | None:
    info = extract_heading_info(text)
    if info:
        return f"{'#' * info['level']} {info['text']}"

    level = heading_level_from_text(text)
    if level is None:
        return None
    return f"{'#' * level} {text}"


# Phân loại dòng text thường/heading/bullet theo quy tắc normalize.
def classify_text_line(text: str) -> str:
    cleaned = normalize_line(text)
    if not cleaned:
        return ""
    if cleaned.startswith(("-", "*")):
        return f"- {cleaned.lstrip('-* ').strip()}"
    heading = classify_heading_line(cleaned)
    if heading:
        return heading
    return cleaned


# Kiểm tra dòng có phải nhiễu header/footer số trang cán bộ.
def is_noise_line(line: str) -> bool:
    if not line or re.fullmatch(r"\d{1,3}", line):
        return True
    return any(pattern.match(line) for pattern in PAGE_NOISE_PATTERNS)


# Lọc dòng nhiễu lặp và noise theo từng trang PDF.
def filter_noise_lines(page_lines: list[list[str]]) -> list[list[str]]:
    page_count = max(len(page_lines), 1)
    line_presence = Counter()
    for lines in page_lines:
        line_presence.update(set(lines))

    repeated_threshold = max(3, int(page_count * 0.6))
    output: list[list[str]] = []
    previous_line = ""
    for lines in page_lines:
        page_output: list[str] = []
        for line in lines:
            if is_noise_line(line):
                continue
            if line_presence[line] >= repeated_threshold and len(line) <= 80:
                continue
            if line == previous_line:
                continue
            page_output.append(line)
            previous_line = line
        output.append(page_output)
    return output


# Render bảng thành các dòng text theo định dạng pipes.
def render_table_lines(rows: list[list[str]], table_index: int, style: str = "pipes") -> list[str]:
    normalized_rows: list[list[str]] = []
    for row in rows:
        cleaned = [normalize_line((cell or "").replace("\n", " ")) for cell in row]
        while cleaned and not cleaned[-1]:
            cleaned.pop()
        if not cleaned:
            continue
        if all(TABLE_SEPARATOR_RE.fullmatch(cell or "") for cell in cleaned):
            continue
        normalized_rows.append(cleaned)

    if not normalized_rows:
        return []

    if style != "pipes":
        raise ValueError("Only 'pipes' table style is supported.")

    target_width = max(len(row) for row in normalized_rows)
    normalized_rows = [row + [""] * (target_width - len(row)) for row in normalized_rows]

    merged_rows: list[list[str]] = []
    for row in normalized_rows:
        first_two_empty = all(index >= len(row) or not row[index] for index in range(2))
        if merged_rows and first_two_empty:
            previous = merged_rows[-1]
            width = max(len(previous), len(row))
            previous.extend([""] * (width - len(previous)))
            row = row + [""] * (width - len(row))
            for index, value in enumerate(row):
                if not value:
                    continue
                previous[index] = f"{previous[index]}; {value}".strip("; ")
            continue
        merged_rows.append(row)

    seen_rows: set[str] = set()
    lines = [f"### Bang {table_index}"]
    for row in merged_rows:
        rendered = "- " + " | ".join(value or "-" for value in row)
        if rendered in seen_rows:
            continue
        seen_rows.add(rendered)
        lines.append(rendered)
    return lines

# Nhận diện xem một dòng có phải heading của bảng hay không
def is_table_heading(text: str) -> bool:
    return bool(TABLE_HEADING_ONLY_RE.match(text.strip()))

# Kiểm tra một dòng có phải dòng dữ liệu của bảng đã render hay không
def is_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("- ") and " | " in stripped

# Kiểm tra một Section có phải là section bảng hay không
def is_table_section(section: Section) -> bool:
    last_heading = section.heading_path[-1] if section.heading_path else ""
    return bool(last_heading) and is_table_heading(f"### {last_heading}") and bool(section.lines) and all(
        is_table_row(line) for line in section.lines
    )

# Tính độ dài phần prefix chung của 2 danh sách heading
def common_heading_prefix_len(left: list[HeadingNode], right: list[HeadingNode]) -> int:
    count = 0
    for lhs, rhs in zip(left, right):
        if lhs != rhs:
            break
        count += 1
    return count

# Quyết định 2 Section có nên merge lại hay không
def same_merge_anchor(left: Section, right: Section) -> bool:
    if left.document_title != right.document_title:
        return False
    # Chỉ merge khi 2 section con cùng nằm dưới ít nhất 1 heading nội dung chung,
    # tránh trộn các fact ngắn ở top-level với các phần khác trong cùng tài liệu.
    return common_heading_prefix_len(left.headings, right.headings) >= 2

# Xác định một Section có quá ít nội dung hay không
def is_sparse_section(section: Section, min_chars: int) -> bool:
    body = section.render_body()
    non_blank_lines = [line for line in section.lines if line.strip()]
    return len(body) < min_chars and len(non_blank_lines) <= 2

# Kiểm tra một Section có chỉ chứa Source hay không
def is_source_only_section(section: Section) -> bool:
    non_blank_lines = [line.strip() for line in section.lines if line.strip()]
    return bool(non_blank_lines) and all(line.startswith("Source: ") for line in non_blank_lines)

# tạo metadata context cho mỗi chunk text
def build_chunk_context(section: Section) -> str:
    # Tạo phần "document_context" để đính kèm vào mỗi chunk (dùng cho truy hồi/ghi dấu vết nguồn).
    parts = ["<document_context>"]
    if section.document_title:
        parts.append(f"document: {section.document_title}")
    if len(section.heading_path) > 1:
        parts.append(f"section_path: {' > '.join(section.heading_path[1:])}")
    parts.append("</document_context>")
    return "\n".join(parts)

# Đóng gói chunk: ghép context + tiêu đề/mục + nội dung + alias tra cứu thành một đoạn text hoàn chỉnh.
def finalize_chunk_text(section: Section, chunk_body: str) -> str:
    # Đóng gói chunk: context + tiêu đề/mục + nội dung; có thể thêm alias "Tra cứu nhanh" từ fact/bảng.
    chunk_body = chunk_body.strip()
    if not chunk_body:
        return ""
    aliases = [*build_basic_fact_aliases(chunk_body), *build_basic_table_aliases(chunk_body)]
    parts = [build_chunk_context(section)]
    if section.document_title:
        parts.append(f"Tài liệu: {section.document_title}")
    if len(section.heading_path) > 1:
        parts.append(f"Mục: {' > '.join(section.heading_path[1:])}")
    parts.append(chunk_body)
    if aliases:
        parts.append("\n".join(aliases))
    return "\n\n".join(part for part in parts if part).strip()

# Đọc từng dòng text dạng key value, rồi biến chúng thành các câu mô tả ngắn
def build_basic_fact_aliases(chunk_body: str) -> list[str]:
    aliases: list[str] = []
    seen: set[str] = set()
    for raw_line in chunk_body.splitlines():
        line = normalize_line(raw_line.lstrip("-+ ").strip())
        if should_skip_fact_alias_line(line):
            continue
        match = INLINE_FACT_LINE_RE.match(line)
        if not match:
            continue
        key = normalize_line(match.group("key"))
        value = normalize_line(match.group("value"))
        if not key or not value or fold_text(key) == "source":
            continue
        aliases.extend(_append_unique_aliases(seen, [f"Tra cứu nhanh: {key} là {value}."]))
    return aliases

# Đọc một đoạn text có bảng, rồi tạo ra tối đa max_aliases câu tóm tắt ngắn từ các dòng dữ liệu trong bảng
def build_basic_table_aliases(chunk_body: str, max_aliases: int = 8) -> list[str]:
    table_lines = [line.strip() for line in chunk_body.splitlines() if is_table_row(line)]
    if len(table_lines) < 2:
        return []

    rows = [parse_table_cells(line) for line in table_lines]
    if not looks_like_table_header(rows[0]):
        return []

    header_rows: list[list[str]] = [rows[0]]
    data_start = 1
    if len(rows) >= 2 and looks_like_table_header(rows[1]):
        header_rows.append(rows[1])
        data_start = 2
    headers = merge_table_headers(header_rows)

    aliases: list[str] = []
    seen: set[str] = set()
    for row in rows[data_start:]:
        summary = summarize_table_row_simple(headers, row)
        if not summary:
            continue
        aliases.extend(_append_unique_aliases(seen, [summary]))
        if len(aliases) >= max_aliases:
            break
    return aliases

# Biến một dòng trong bảng thành một câu mô tả ngắn
def summarize_table_row_simple(headers: list[str], row: list[str]) -> str:
    pairs: list[str] = []
    for header, value in zip_longest(headers, row, fillvalue=""):
        header_clean = normalize_line(header)
        value_clean = normalize_line(value)
        if not header_clean or header_clean == "-" or not value_clean or value_clean == "-":
            continue
        if header_clean == value_clean:
            continue
        pairs.append(f"{header_clean}: {value_clean}")
    if not pairs:
        return ""
    return "Tra cứu bảng: " + "; ".join(pairs) + "."

# Lọc trùng và chỉ trả về những alias chưa xuất hiện
def _append_unique_aliases(seen: set[str], aliases: list[str]) -> list[str]:
    appended: list[str] = []
    for alias in aliases:
        if alias in seen:
            continue
        seen.add(alias)
        appended.append(alias)
    return appended

# Lọc bỏ các dòng không phù hợp, giữ lại text thường
def should_skip_fact_alias_line(line: str) -> bool:
    if not line:
        return True
    if " | " in line:
        return True
    lowered = line.lower()
    return "http://" in lowered or "https://" in lowered

# Biến một dòng bảng thành danh sách các ô
def parse_table_cells(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("- "):
        stripped = stripped[2:]
    return [normalize_line(cell) for cell in stripped.split(" | ")]

# Kiểm tra một dòng có phải là header của bảng hay không
def looks_like_table_header(row: list[str]) -> bool:
    meaningful_cells = [normalize_line(cell) for cell in row if normalize_line(cell) and normalize_line(cell) != "-"]
    if not meaningful_cells:
        return False

    first_cell = meaningful_cells[0]
    if any(char.isdigit() for char in first_cell) or ROMAN_TOKEN_RE.fullmatch(first_cell):
        return False

    informative = 0
    non_empty = 0
    for cell in row:
        cleaned = normalize_line(cell)
        if not cleaned or cleaned == "-":
            continue
        non_empty += 1
        has_alpha = any(char.isalpha() for char in cleaned)
        has_digit = any(char.isdigit() for char in cleaned)
        if has_alpha and not has_digit:
            informative += 1
    return non_empty > 0 and informative >= max(2, non_empty // 2)

# kiểm tra một dòng có phải là dòng nhóm trong bảng hay không
def is_table_group_row(row: list[str]) -> bool:
    meaningful_cells = [normalize_line(cell) for cell in row if normalize_line(cell) and normalize_line(cell) != "-"]
    if not meaningful_cells:
        return False
    return bool(ROMAN_TOKEN_RE.fullmatch(meaningful_cells[0]))

# gộp 1 hoặc 2 dòng header thành 1 danh sách tên cột rõ ràng
def merge_table_headers(header_rows: list[list[str]]) -> list[str]:
    if len(header_rows) == 1:
        return [normalize_line(cell) for cell in header_rows[0]]

    merged: list[str] = []
    for upper, lower in zip_longest(header_rows[0], header_rows[1], fillvalue=""):
        upper_clean = normalize_line(upper)
        lower_clean = normalize_line(lower)
        if not upper_clean or upper_clean == "-":
            merged.append(lower_clean)
            continue
        if not lower_clean or lower_clean == "-" or lower_clean == upper_clean:
            merged.append(upper_clean)
            continue
        merged.append(f"{upper_clean} ({lower_clean})")
    return merged


# Tách tài liệu thành các section dựa trên heading markdown và giữ stack heading cha.
def split_sections(text: str) -> list[Section]:
    sections: list[Section] = []
    heading_stack: list[HeadingNode] = []
    current_lines: list[str] = []
    current_is_table = False

    def flush_current() -> None:
        nonlocal current_lines, current_is_table
        trimmed = trim_blank_edges(current_lines)
        if trimmed:
            sections.append(Section(headings=list(heading_stack), lines=trimmed, is_table=current_is_table))
        current_lines = []
        current_is_table = False

    def materialize_heading_only_fact() -> None:
        if current_lines or current_is_table or len(heading_stack) < 2:
            return
        last_heading = heading_stack[-1][1]
        if not is_inline_fact_heading(last_heading):
            return
        sections.append(Section(headings=list(heading_stack[:-1]), lines=[last_heading], is_table=False))

    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        if current_is_table:
            if not line.strip():
                continue
            if is_table_row(line):
                current_lines.append(line.rstrip())
                continue
            flush_current()
            if heading_stack and is_table_heading(f"### {heading_stack[-1][1]}"):
                heading_stack.pop()

        heading_match = HEADING_RE.match(line)
        if heading_match:
            materialize_heading_only_fact()
            flush_current()
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_text))
            current_is_table = is_table_heading(line)
            continue

        current_lines.append(line.rstrip())

    materialize_heading_only_fact()
    flush_current()
    return [section for section in sections if section.text]


# Gộp các section quá ngắn liền kề, nhưng tránh gộp các phần có nghĩa riêng biệt.
def merge_sections(sections: list[Section], min_chars: int, max_sections_per_chunk: int) -> list[Section]:
    merged: list[Section] = []
    index = 0
    while index < len(sections):
        section = sections[index]
        headings = list(section.headings)
        lines = list(section.lines)
        current = Section(headings=headings, lines=lines, is_table=section.is_table)
        used = 1
        index += 1

        while (
            index < len(sections)
            and used < max_sections_per_chunk
            and is_sparse_section(current, min_chars=min_chars)
        ):
            next_section = sections[index]
            if is_table_section(current) or is_table_section(next_section) or not same_merge_anchor(current, next_section):
                break
            prefix_len = common_heading_prefix_len(current.headings, next_section.headings)
            addition = next_section.render_body(skip_heading_count=prefix_len)
            if addition:
                if lines and lines[-1].strip():
                    lines.append("")
                lines.extend(addition.splitlines())
            current = Section(headings=headings, lines=trim_blank_edges(lines), is_table=False)
            used += 1
            index += 1

        merged.append(current)
    return merged


# Tách section quá lớn thành nhiều chunk nhỏ hơn theo độ dài tối đa.
def split_large_section(section: Section, max_chars: int) -> list[str]:
    text = section.render_body()
    if len(text) <= max_chars:
        return [text]

    if is_table_section(section):
        table_chunks = split_large_table_section(section, max_chars=max_chars)
        if table_chunks:
            return table_chunks

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
    available_chars = max_chars - len(context_heading) - 1 if context_heading else max_chars
    available_chars = max(200, available_chars)
    expanded_remaining: list[str] = []
    for line in remaining:
        expanded_remaining.extend(split_long_line(line, max_chars=available_chars))
    chunks: list[str] = []
    current_lines = [context_heading] if context_heading else []
    for line in expanded_remaining:
        candidate = "\n".join(current_lines + [line]).strip()
        if current_lines and len(candidate) > max_chars:
            chunks.append("\n".join(current_lines).strip())
            current_lines = [context_heading, line] if context_heading else [line]
        else:
            current_lines.append(line)
    if current_lines:
        chunks.append("\n".join(current_lines).strip())
    return [chunk for chunk in chunks if chunk]

# Tách bảng lớn thành nhiều bảng nhỏ, nhưng vẫn giữ cấu trúc (header + group)
def split_large_table_section(section: Section, max_chars: int) -> list[str]:
    lines = [line.rstrip() for line in section.render_body().splitlines() if line.strip()]
    if not lines:
        return []

    heading_lines: list[str] = []
    index = 0
    while index < len(lines) and HEADING_RE.match(lines[index]):
        heading_lines.append(lines[index])
        index += 1

    table_lines = lines[index:]
    if not table_lines:
        return []

    header_rows: list[str] = []
    data_rows: list[str] = []
    for line in table_lines:
        if not is_table_row(line):
            continue
        cells = parse_table_cells(line)
        if not data_rows and looks_like_table_header(cells):
            header_rows.append(line)
            continue
        if not data_rows and header_rows and looks_like_table_header(cells):
            header_rows.append(line)
            continue
        data_rows.append(line)

    if not header_rows or not data_rows:
        return []

    prefix_lines = [*heading_lines, *header_rows]
    chunks: list[str] = []
    current_group_row = ""
    current_lines = list(prefix_lines)
    min_prefix_len = len(prefix_lines)

    for row_line in data_rows:
        row_cells = parse_table_cells(row_line)
        if is_table_group_row(row_cells):
            if len(current_lines) > min_prefix_len:
                chunks.append("\n".join(current_lines).strip())
            current_group_row = row_line
            current_lines = [*prefix_lines, current_group_row]
            continue

        if len(current_lines) == min_prefix_len and current_group_row:
            current_lines.append(current_group_row)

        candidate = "\n".join([*current_lines, row_line]).strip()
        if len(current_lines) > min_prefix_len and len(candidate) > max_chars:
            chunks.append("\n".join(current_lines).strip())
            current_lines = [*prefix_lines]
            if current_group_row:
                current_lines.append(current_group_row)
        current_lines.append(row_line)

    if len(current_lines) > min_prefix_len:
        chunks.append("\n".join(current_lines).strip())
    return [chunk for chunk in chunks if chunk]

# Chia dòng dài thành các đoạn nhỏ
def split_long_line(line: str, max_chars: int) -> list[str]:
    stripped = line.strip()
    if len(stripped) <= max_chars:
        return [stripped]

    if is_table_row(stripped):
        row_chunks = split_long_table_row(stripped, max_chars=max_chars)
        if row_chunks and all(len(chunk) <= max_chars for chunk in row_chunks):
            return row_chunks

    return wrap_text_by_words(stripped, max_chars=max_chars)

# Chia một dòng bảng dài thành nhiều dòng nhỏ bằng cách tách các mục được đánh số ở cột cuối
def split_long_table_row(line: str, max_chars: int) -> list[str]:
    if " | " not in line:
        return []
    prefix, last_cell = line.rsplit(" | ", 1)
    matches = list(NUMBERED_ITEM_RE.finditer(last_cell))
    if len(matches) < 2:
        return []

    items: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(last_cell)
        item = last_cell[match.start() : end].strip(" ;")
        if item:
            items.append(item)
    if len(items) < 2:
        return []

    base = f"{prefix} | "
    available = max(max_chars - len(base), 120)
    chunks: list[str] = []
    current_items: list[str] = []
    for item in items:
        wrapped_parts = wrap_text_by_words(item, max_chars=available)
        for part in wrapped_parts:
            candidate = "; ".join(current_items + [part]).strip()
            rendered = f"{base}{candidate}".strip()
            if current_items and len(rendered) > max_chars:
                chunks.append(f"{base}{'; '.join(current_items)}".strip())
                current_items = [part]
            else:
                current_items.append(part)
    if current_items:
        chunks.append(f"{base}{'; '.join(current_items)}".strip())
    return chunks

# Word wrap thủ công: xuống dòng theo từ, giữ độ dài tối đa.
def wrap_text_by_words(text: str, max_chars: int) -> list[str]:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return [stripped]

    words = stripped.split()
    if not words:
        return [stripped]

    chunks: list[str] = []
    current_words: list[str] = []
    for word in words:
        candidate = " ".join(current_words + [word]).strip()
        if current_words and len(candidate) > max_chars:
            chunks.append(" ".join(current_words))
            current_words = [word]
            continue
        if not current_words and len(word) > max_chars:
            start = 0
            while start < len(word):
                chunks.append(word[start : start + max_chars])
                start += max_chars
            current_words = []
            continue
        current_words.append(word)
    if current_words:
        chunks.append(" ".join(current_words))
    return [chunk for chunk in chunks if chunk.strip()]

# Lọc bỏ bảng rác dựa trên từ khóa trong header
def is_low_value_table_section(section: Section) -> bool:
    if not is_table_section(section):
        return False
    header_sample = re.sub(r"[\W_]+", " ", "\n".join(section.lines[:3]).lower()).strip()
    normalized_markers = [re.sub(r"[\W_]+", " ", marker.lower()).strip() for marker in LOW_VALUE_TABLE_MARKERS]
    return all(marker in header_sample for marker in normalized_markers)


# Tạo các dòng JSONL chunk từ văn bản TXT gốc.
def build_chunk_rows(
    text: str,
    source_txt: Path,
    min_chars: int,
    max_chars: int,
    max_sections_per_chunk: int,
) -> list[dict]:
    # B1) Tách tài liệu thành các section theo heading, rồi lọc bớt section “ít giá trị”.
    sections = [
        section
        for section in split_sections(text)
        if not is_low_value_table_section(section) and not is_source_only_section(section)
    ]
    if not sections:
        return []

    # B2) Gộp section ngắn lại (theo min_chars và số section tối đa) để tạo chunk “vừa đủ”.
    merged_sections = merge_sections(sections, min_chars=min_chars, max_sections_per_chunk=max_sections_per_chunk)
    rows: list[dict] = []
    chunk_id = 1
    for section in merged_sections:
        # B3) Nếu section quá dài thì tách thêm theo max_chars.
        for chunk_body in split_large_section(section, max_chars=max_chars):
            chunk_text = finalize_chunk_text(section, chunk_body)
            if not chunk_text:
                continue
            headings = []
            for line in chunk_text.splitlines():
                match = HEADING_RE.match(line)
                if match:
                    headings.append(match.group(2).strip())
            if not headings:
                headings = section.heading_path
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "source_txt": str(source_txt),
                    "heading_path": section.heading_path or headings,
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                    "line_count": sum(1 for line in chunk_text.splitlines() if line.strip()),
                }
            )
            chunk_id += 1
    return rows


# Ghi danh sách row ra file JSONL UTF-8.
def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
