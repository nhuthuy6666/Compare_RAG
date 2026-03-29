from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from typing import Any

from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core.indices.property_graph.base import KG_NODES_KEY, KG_RELATIONS_KEY
from llama_index.core.schema import BaseNode


PHONE_RE = re.compile(r"(0\d[\d.\s]{7,}\d)")
METHOD_RE = re.compile(
    r"Phương thức\s*(\d)\s*:\s*(.*?)(?=(?:-\s*Phương thức\s*\d\s*:|$))",
    re.IGNORECASE | re.DOTALL,
)
ADMISSION_CODE_RE = re.compile(
    r"(?:Mã cơ sở đào tạo trong tuyển sinh|Mã trường)\s*:\s*([A-Z0-9]+)",
    re.IGNORECASE,
)
ADDRESS_RE = re.compile(r"Địa chỉ trụ sở:\s*(.+)", re.IGNORECASE)
WEBSITE_RE = re.compile(r"https?://[^\s)]+")
PROGRAM_CODE_RE = re.compile(r"^[0-9]{6,}[A-Z]*$")
VALUE_RE = re.compile(r"^[0-9]+(?:[.,][0-9]+)?$")


class AdmissionsGraphExtractor:
    """Extractor rule-based để rút facts tuyển sinh thành node và relation cho GraphRAG."""
    # Extractor “rule-based” cho GraphRAG tuyển sinh:
    # - Đọc text chunk (kể cả table_row) và sinh EntityNode/Relation theo các pattern thường gặp.
    # - Mục tiêu: tạo graph facts ổn định, nhanh và dễ debug (không phụ thuộc LLM để trích quan hệ).
    # Khởi tạo extractor rule-based và cấu hình tần suất in tiến độ.
    def __init__(self, progress_every: int = 5) -> None:
        """Khởi tạo extractor và cấu hình tần suất in tiến độ khi xử lý chunk."""
        self._progress_every = max(progress_every, 1)

    @classmethod
    # Trả về tên lớp để LlamaIndex có thể nhận diện extractor này.
    def class_name(cls) -> str:
        """Trả về tên lớp để LlamaIndex nhận diện extractor này."""
        return "AdmissionsGraphExtractor"

    # Wrapper đồng bộ cho interface extractor của LlamaIndex.
    def __call__(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> Sequence[BaseNode]:
        """Wrapper đồng bộ để tương thích với interface extractor của LlamaIndex."""
        return asyncio.run(self.acall(nodes, show_progress=show_progress, **kwargs))

    # Duyệt lần lượt từng chunk node và gắn thêm KG nodes/relations vào metadata.
    async def acall(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> Sequence[BaseNode]:
        """Duyệt từng chunk, trích facts và gắn KG nodes/relations vào metadata."""
        # Duyệt lần lượt từng chunk node và gắn thêm KG_NODES/KG_RELATIONS vào metadata.
        total = len(nodes)
        results: list[BaseNode] = []
        for index, node in enumerate(nodes, start=1):
            results.append(self._extract_node(node))
            if index % self._progress_every == 0 or index == total:
                print(f"Extracted {index}/{total} chunks")
        return results

    # Trích entity/relation từ một chunk và gắn lại vào metadata của node.
    def _extract_node(self, node: BaseNode) -> BaseNode:
        """Trích entity/relation từ một chunk rồi gắn ngược lại vào metadata của node."""
        # 1) Khởi tạo tập entities/relations cục bộ cho chunk này.
        text = node.get_content()
        metadata = node.metadata.copy()
        entities: dict[str, EntityNode] = {}
        relations: dict[tuple[str, str, str], Relation] = {}

        # Tạo mới hoặc tái sử dụng entity đã có trong chunk hiện tại.
        def ensure_entity(name: str, label: str, **props: Any) -> EntityNode:
            normalized_name = _clean_text(name)
            if not normalized_name:
                raise ValueError("Entity name cannot be empty.")
            entity = entities.get(normalized_name)
            base_props = metadata.copy()
            base_props.update({k: v for k, v in props.items() if v not in (None, "", [])})
            if entity is None:
                entity = EntityNode(name=normalized_name, label=label, properties=base_props)
                entities[normalized_name] = entity
            else:
                entity.properties.update(base_props)
            return entity

        # Tạo relation giữa hai entity và gộp metadata nếu relation đã tồn tại.
        def add_relation(
            source_name: str,
            source_label: str,
            relation_label: str,
            target_name: str,
            target_label: str,
            source_props: dict[str, Any] | None = None,
            target_props: dict[str, Any] | None = None,
            relation_props: dict[str, Any] | None = None,
        ) -> None:
            source = ensure_entity(source_name, source_label, **(source_props or {}))
            target = ensure_entity(target_name, target_label, **(target_props or {}))
            rel_key = (source.id, relation_label, target.id)
            base_props = metadata.copy()
            if relation_props:
                base_props.update(relation_props)
            if rel_key not in relations:
                relations[rel_key] = Relation(
                    label=relation_label,
                    source_id=source.id,
                    target_id=target.id,
                    properties=base_props,
                )
            else:
                relations[rel_key].properties.update(base_props)

        try:
            # 2) Trích “facts thường” từ đoạn văn.
            self._extract_general_facts(text, metadata, add_relation)
            # 3) Trích “facts dạng bảng” nếu record_type=table_row.
            self._extract_table_facts(text, metadata, add_relation)
        except Exception:
            # Keep ingestion resilient even if one chunk has malformed table text.
            pass

        # 4) Ghép entities/relations mới vào metadata để pipeline GraphRAG chuyển thành graph facts.
        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
        existing_nodes.extend(entities.values())
        existing_relations.extend(relations.values())
        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node

    # Trích các fact tổng quát như mã trường, địa chỉ, điện thoại, website, phương thức tuyển sinh.
    def _extract_general_facts(
        self,
        text: str,
        metadata: dict[str, Any],
        add_relation: Any,
    ) -> None:
        """Trích các fact tổng quát như mã trường, địa chỉ, điện thoại, website và phương thức tuyển sinh."""
        if metadata.get("record_type") == "table_row":
            return

        institution = "Trường Đại học Nha Trang"
        year = metadata.get("source_year")
        source_file = str(metadata.get("source_file", ""))

        if "Nha Trang" in text or "ntu.edu.vn" in text:
            add_relation(institution, "INSTITUTION", "HAS_SOURCE_FILE", source_file or institution, "SOURCE")

        admission_code_match = ADMISSION_CODE_RE.search(text)
        if admission_code_match:
            add_relation(
                institution,
                "INSTITUTION",
                "HAS_ADMISSION_CODE",
                admission_code_match.group(1),
                "ADMISSION_CODE",
            )

        address_match = ADDRESS_RE.search(text)
        if address_match:
            address = _clean_inline_text(address_match.group(1))
            add_relation(institution, "INSTITUTION", "LOCATED_AT", address, "ADDRESS")

        for phone in {phone.strip().replace(" ", "") for phone in PHONE_RE.findall(text)}:
            add_relation(institution, "INSTITUTION", "HAS_CONTACT_PHONE", phone, "CONTACT_PHONE")

        for website in WEBSITE_RE.findall(text):
            add_relation(institution, "INSTITUTION", "HAS_WEBSITE", website, "WEBSITE")

        methods = _extract_methods(text)
        if methods:
            if year and year != "unknown":
                add_relation(institution, "INSTITUTION", "HAS_ADMISSION_YEAR", year, "YEAR")
            for method_code, description in methods:
                method_name = f"Phương thức {method_code}: {description}"
                add_relation(institution, "INSTITUTION", "USES_METHOD", method_name, "ADMISSION_METHOD")
                if year and year != "unknown":
                    add_relation(method_name, "ADMISSION_METHOD", "APPLIES_TO_YEAR", year, "YEAR")

    # Phân loại và trích fact từ các chunk bảng tuyển sinh.
    def _extract_table_facts(
        self,
        text: str,
        metadata: dict[str, Any],
        add_relation: Any,
    ) -> None:
        """Phân loại và trích fact từ các chunk bảng tuyển sinh."""
        if metadata.get("record_type") != "table_row":
            return

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        row_lines = [line for line in lines if line.startswith("- ") and " | " in line]
        if not row_lines:
            return

        header_lines = [line for line in row_lines if _is_header_row(line)]
        data_lines = [line for line in row_lines if not _is_header_row(line)]
        if not data_lines:
            return

        header_text = " ".join(header_lines)
        for row_line in data_lines:
            cells = _split_row_cells(row_line)
            if not cells:
                continue
            if _is_group_row(cells):
                continue
            if _looks_like_score_row(metadata, header_text, cells):
                self._extract_score_row(cells, header_text, metadata, add_relation)
                continue
            if _looks_like_quota_row(header_text, cells):
                self._extract_quota_row(cells, metadata, add_relation)
                continue
            if _looks_like_subject_row(header_text, cells):
                self._extract_subject_row(cells, metadata, add_relation)

    # Trích fact chỉ tiêu, mã ngành và phương thức từ một dòng bảng chỉ tiêu.
    def _extract_quota_row(
        self,
        cells: list[str],
        metadata: dict[str, Any],
        add_relation: Any,
    ) -> None:
        """Trích fact về chỉ tiêu, mã ngành và phương thức từ một dòng bảng chỉ tiêu."""
        if len(cells) < 7:
            return

        admission_code = _find_first(cells, PROGRAM_CODE_RE)
        if not admission_code:
            return

        program_name = _find_program_name(cells, admission_code)
        if not program_name:
            return

        year = metadata.get("source_year")
        major_code, major_code_index = _find_second_program_code(cells, admission_code)
        quota = ""
        admission_code_index = cells.index(admission_code)
        start_index = major_code_index + 1 if major_code_index >= 0 else admission_code_index + 1
        for cell in cells[start_index:]:
            if VALUE_RE.match(cell):
                quota = cell
                break

        program_props = {"year": year, "admission_code": admission_code}
        add_relation(program_name, "PROGRAM", "HAS_ADMISSION_CODE", admission_code, "PROGRAM_CODE", program_props)
        if year and year != "unknown":
            add_relation(program_name, "PROGRAM", "APPLIES_TO_YEAR", year, "YEAR", program_props)
        if major_code and major_code != admission_code:
            add_relation(program_name, "PROGRAM", "HAS_MAJOR_CODE", major_code, "MAJOR_CODE", program_props)
        if quota:
            quota_name = f"Chỉ tiêu {program_name} năm {year}: {quota}" if year and year != "unknown" else f"Chỉ tiêu {program_name}: {quota}"
            add_relation(program_name, "PROGRAM", "HAS_QUOTA", quota_name, "QUOTA_FACT", program_props)

        for method_code, description in _extract_methods(" ".join(cells)):
            method_name = f"Phương thức {method_code}: {description}"
            add_relation(program_name, "PROGRAM", "USES_METHOD", method_name, "ADMISSION_METHOD", program_props)
            if year and year != "unknown":
                add_relation(method_name, "ADMISSION_METHOD", "APPLIES_TO_YEAR", year, "YEAR", program_props)

    # Trích tổ hợp xét tuyển và điều kiện tiếng Anh từ một dòng bảng tổ hợp môn.
    def _extract_subject_row(
        self,
        cells: list[str],
        metadata: dict[str, Any],
        add_relation: Any,
    ) -> None:
        """Trích tổ hợp xét tuyển và điều kiện tiếng Anh từ một dòng bảng tổ hợp môn."""
        if len(cells) < 5:
            return

        admission_code = _find_first(cells, PROGRAM_CODE_RE)
        program_name = _find_program_name(cells, admission_code or "")
        if not program_name:
            return

        year = metadata.get("source_year")
        if admission_code:
            add_relation(program_name, "PROGRAM", "HAS_ADMISSION_CODE", admission_code, "PROGRAM_CODE", {"year": year})
        if year and year != "unknown":
            add_relation(program_name, "PROGRAM", "APPLIES_TO_YEAR", year, "YEAR")

        subject_combo = next((cell for cell in cells if "Toán" in cell or "Ngữ văn" in cell or "Tiếng Anh" in cell), "")
        if subject_combo:
            combo_name = f"Tổ hợp xét tuyển {program_name} năm {year}: {subject_combo}" if year and year != "unknown" else f"Tổ hợp xét tuyển {program_name}: {subject_combo}"
            add_relation(program_name, "PROGRAM", "HAS_SUBJECT_COMBINATION", combo_name, "SUBJECT_COMBINATION")

        english_requirement = next((cell for cell in reversed(cells) if cell in {"X", "-", "X; X"} or VALUE_RE.match(cell)), "")
        if english_requirement and english_requirement not in {"-", ""}:
            requirement_name = (
                f"Điều kiện tiếng Anh {program_name} năm {year}: {english_requirement}"
                if year and year != "unknown"
                else f"Điều kiện tiếng Anh {program_name}: {english_requirement}"
            )
            add_relation(program_name, "PROGRAM", "HAS_ENGLISH_REQUIREMENT", requirement_name, "ENGLISH_REQUIREMENT")

    # Trích các fact điểm chuẩn/điểm điều kiện từ một dòng bảng điểm.
    def _extract_score_row(
        self,
        cells: list[str],
        header_text: str,
        metadata: dict[str, Any],
        add_relation: Any,
    ) -> None:
        """Trích các fact điểm chuẩn hoặc điều kiện điểm từ một dòng bảng điểm."""
        if len(cells) < 4:
            return

        year = _extract_score_year(metadata, header_text, cells)
        admission_code = cells[1] if len(cells) > 1 and PROGRAM_CODE_RE.match(cells[1]) else ""
        program_name = cells[2] if len(cells) > 2 else ""
        program_name = _clean_text(program_name)
        if not program_name:
            return

        if admission_code:
            add_relation(program_name, "PROGRAM", "HAS_ADMISSION_CODE", admission_code, "PROGRAM_CODE", {"year": year})
        if year:
            add_relation(program_name, "PROGRAM", "APPLIES_TO_YEAR", year, "YEAR")

        metrics: list[tuple[str, str]] = []
        if len(cells) >= 8 and year == "2025":
            metrics = [
                ("Điểm thi THPT 2025 (hệ số 1)", cells[3]),
                ("Điểm thi THPT 2025 (hệ số 2 hoặc tiếng Anh hệ số 2)", cells[4]),
                ("Điểm ĐGNL ĐHQG-HCM 2025", cells[5]),
                ("Điểm ĐGNL ĐHQG-HN 2025", cells[6]),
                ("Điều kiện tiếng Anh 2025", cells[7]),
            ]
        elif len(cells) >= 11 and year == "2024":
            metrics = [
                ("Điểm thi THPT 2024", cells[4]),
                ("Điểm tiếng Anh thi THPT 2024", cells[5]),
                ("Điểm học bạ THPT 2024", cells[7]),
                ("Điểm tiếng Anh học bạ 2024", cells[8]),
                ("Điểm ĐGNL 2024", cells[9]),
                ("Điểm tiếng Anh ĐGNL 2024", cells[10]),
            ]

        for metric_name, value in metrics:
            cleaned_value = _clean_text(value)
            if not cleaned_value or cleaned_value == "-":
                continue
            fact_name = f"{program_name} - {metric_name}: {cleaned_value}"
            add_relation(program_name, "PROGRAM", "HAS_SCORE_FACT", fact_name, "SCORE_FACT", {"year": year})


# Chuẩn hóa khoảng trắng và ký tự thừa trong một ô bảng hoặc đoạn text ngắn.
def _clean_text(value: str) -> str:
    """Chuẩn hóa khoảng trắng và ký tự thừa trong một ô bảng hoặc đoạn text ngắn."""
    return re.sub(r"\s+", " ", value).strip(" -;")


# Làm sạch text nội tuyến sau khi bỏ xuống dòng.
def _clean_inline_text(value: str) -> str:
    """Làm sạch text nội tuyến sau khi bỏ ký tự xuống dòng."""
    return _clean_text(value.replace("\n", " "))


# Tách một dòng bảng dạng markdown thành danh sách các ô.
def _split_row_cells(line: str) -> list[str]:
    """Tách một dòng bảng dạng markdown thành danh sách các ô."""
    raw = line[2:] if line.startswith("- ") else line
    return [_clean_text(cell) for cell in raw.split("|")]


# Nhận diện dòng header của bảng.
def _is_header_row(line: str) -> bool:
    """Nhận diện một dòng có phải header của bảng hay không."""
    first_cell = _split_row_cells(line)[0].upper()
    return first_cell in {"TT", "STT", "(1)", "(2)"} or line.startswith("- (")


# Nhận diện dòng nhóm/chia mục trong bảng để bỏ qua khi extract.
def _is_group_row(cells: list[str]) -> bool:
    """Nhận diện dòng nhóm hoặc dòng chia mục để bỏ qua khi extract."""
    first_cell = cells[0]
    non_empty = [cell for cell in cells if cell and cell != "-"]
    if re.fullmatch(r"[IVX]+", first_cell):
        return True
    if re.fullmatch(r"\d+\.\d+", first_cell) and len(non_empty) <= 3:
        return True
    return False


# Tìm ô đầu tiên khớp với pattern cho trước.
def _find_first(cells: list[str], pattern: re.Pattern[str]) -> str:
    """Tìm ô đầu tiên khớp với pattern cho trước."""
    for cell in cells:
        if pattern.match(cell):
            return cell
    return ""


# Tìm mã chương trình thứ hai trong dòng để phân biệt mã xét tuyển và mã ngành.
def _find_second_program_code(cells: list[str], first_code: str) -> tuple[str, int]:
    """Tìm mã chương trình thứ hai để phân biệt mã xét tuyển và mã ngành."""
    found_first = False
    for index, cell in enumerate(cells):
        if not PROGRAM_CODE_RE.match(cell):
            continue
        if not found_first and cell == first_code:
            found_first = True
            continue
        return cell, index
    return "", -1


# Suy ra tên chương trình/ngành từ các ô dữ liệu trong một dòng bảng.
def _find_program_name(cells: list[str], admission_code: str) -> str:
    """Suy ra tên chương trình hoặc ngành từ các ô dữ liệu trong một dòng bảng."""
    found_code = False
    for cell in cells:
        if cell == admission_code:
            found_code = True
            continue
        if not found_code:
            continue
        if cell in {"-", ""}:
            continue
        if PROGRAM_CODE_RE.match(cell) or VALUE_RE.match(cell):
            continue
        return cell
    for cell in cells:
        if any(char.isalpha() for char in cell) and "Phương thức" not in cell and "Toán" not in cell:
            return cell
    return ""


# Trích danh sách phương thức tuyển sinh xuất hiện trong text.
def _extract_methods(text: str) -> list[tuple[str, str]]:
    """Trích danh sách phương thức tuyển sinh xuất hiện trong text."""
    normalized = text.replace("\n", " ")
    methods: list[tuple[str, str]] = []
    for method_code, description in METHOD_RE.findall(normalized):
        cleaned = _clean_text(description)
        cleaned = re.sub(r"\s+X$", "", cleaned).strip()
        if cleaned:
            methods.append((method_code, cleaned))
    return methods


# Kiểm tra dòng dữ liệu có giống bảng chỉ tiêu hay không.
def _looks_like_quota_row(header_text: str, cells: list[str]) -> bool:
    """Kiểm tra một dòng dữ liệu có giống bảng chỉ tiêu hay không."""
    return "Chỉ tiêu" in header_text and any("Phương thức" in cell for cell in cells)


# Kiểm tra dòng dữ liệu có giống bảng tổ hợp xét tuyển hay không.
def _looks_like_subject_row(header_text: str, cells: list[str]) -> bool:
    """Kiểm tra một dòng dữ liệu có giống bảng tổ hợp xét tuyển hay không."""
    return "Tổ hợp xét tuyển" in header_text and len(cells) >= 5


# Kiểm tra dòng dữ liệu có thuộc bảng điểm chuẩn/điều kiện điểm hay không.
def _looks_like_score_row(metadata: dict[str, Any], header_text: str, cells: list[str]) -> bool:
    """Kiểm tra một dòng dữ liệu có thuộc bảng điểm chuẩn hoặc điều kiện điểm hay không."""
    source_file = str(metadata.get("source_file", ""))
    return "diem-trung-tuyen" in source_file or (
        "ĐGNL" in header_text and ("Điểm thi" in header_text or "Điểm chuẩn" in header_text)
    )


# Suy ra năm áp dụng cho một dòng điểm từ metadata, header hoặc tên file.
def _extract_score_year(metadata: dict[str, Any], header_text: str, cells: list[str]) -> str:
    """Suy ra năm áp dụng của một dòng điểm từ metadata, header hoặc tên file."""
    source_year = str(metadata.get("source_year", "") or "")
    if source_year and source_year != "unknown":
        return source_year
    match = re.search(r"20\d{2}", header_text)
    if match:
        return match.group(0)
    joined = " ".join(cells)
    match = re.search(r"20\d{2}", joined)
    if match:
        return match.group(0)
    source_file = str(metadata.get("source_file", ""))
    if "2025" in source_file:
        return "2025"
    if "2024" in source_file:
        return "2024"
    if "diem-trung-tuyen" in source_file:
        return "2025"
    return ""
