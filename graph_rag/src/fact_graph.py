from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llama_index.core.indices.property_graph.base import KG_NODES_KEY, KG_RELATIONS_KEY

from admissions_graph import AdmissionsGraphExtractor
from utils import records_to_nodes


RELATION_TEXT = {
    "HAS_SOURCE_FILE": "có nguồn tài liệu",
    "HAS_ADMISSION_CODE": "có mã tuyển sinh là",
    "LOCATED_AT": "đặt tại",
    "HAS_CONTACT_PHONE": "có số điện thoại liên hệ",
    "HAS_WEBSITE": "có website",
    "USES_METHOD": "sử dụng phương thức tuyển sinh",
    "HAS_ADMISSION_YEAR": "có thông tin tuyển sinh năm",
    "APPLIES_TO_YEAR": "áp dụng cho năm",
    "HAS_MAJOR_CODE": "có mã ngành",
    "HAS_QUOTA": "có thông tin chỉ tiêu",
    "HAS_SUBJECT_COMBINATION": "có tổ hợp xét tuyển",
    "HAS_ENGLISH_REQUIREMENT": "có điều kiện tiếng Anh",
    "HAS_SCORE_FACT": "có thông tin điểm chuẩn",
}


@dataclass
class GraphArtifacts:
    """Gói dữ liệu trung gian sẽ được ghi vào Neo4j."""

    entities: dict[str, dict[str, Any]] = field(default_factory=dict)
    chunks: dict[str, dict[str, Any]] = field(default_factory=dict)
    facts: list[dict[str, Any]] = field(default_factory=list)


# Trích entity, relation và fact text từ shared chunks để ingest vào Neo4j.
def build_graph_artifacts(records: list[dict], progress_every: int) -> tuple[GraphArtifacts, list[dict]]:
    """Sinh graph artifacts từ chunk records để phục vụ ingest và audit."""

    extracted_nodes = AdmissionsGraphExtractor(progress_every=progress_every)(records_to_nodes(records))
    artifacts = GraphArtifacts()
    fact_records: list[dict] = []

    for node in extracted_nodes:
        metadata = dict(node.metadata or {})
        chunk_uid = _chunk_uid(metadata)
        artifacts.chunks.setdefault(
            chunk_uid,
            {
                "uid": chunk_uid,
                "relative_path": str(metadata.get("relative_path") or ""),
                "source_file": str(metadata.get("source_file") or ""),
                "heading_path": _stringify_heading_path(metadata.get("heading_path")),
                "chunk_id": str(metadata.get("chunk_id") or ""),
                "source_year": str(metadata.get("source_year") or ""),
            },
        )

        entities = {
            entity.id: entity
            for entity in (metadata.get(KG_NODES_KEY) or [])
            if getattr(entity, "id", None) and getattr(entity, "name", None)
        }
        for entity_id, entity in entities.items():
            artifacts.entities.setdefault(entity_id, _entity_record(entity))

        relations = list(metadata.get(KG_RELATIONS_KEY) or [])
        created = 0
        for relation in relations:
            source = entities.get(relation.source_id)
            target = entities.get(relation.target_id)
            if source is None or target is None:
                continue

            text = _build_relation_text(
                source_name=source.name,
                relation_label=relation.label,
                target_name=target.name,
                heading_context=metadata.get("heading_path") or metadata.get("title") or metadata.get("source_file") or "",
            )
            fact_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{node.node_id}:{relation.source_id}:{relation.label}:{relation.target_id}",
                )
            )
            source_uid = str(source.id)
            target_uid = str(target.id)
            source_record = artifacts.entities.setdefault(source_uid, _entity_record(source))
            target_record = artifacts.entities.setdefault(target_uid, _entity_record(target))
            fact_payload = {
                "id": fact_id,
                "text": text,
                "relative_path": str(metadata.get("relative_path") or ""),
                "source_file": str(metadata.get("source_file") or ""),
                "heading_path": _stringify_heading_path(metadata.get("heading_path")),
                "chunk_id": str(metadata.get("chunk_id") or ""),
                "source_year": str(metadata.get("source_year") or ""),
                "relation_label": str(relation.label),
                "source_entity": str(source.name),
                "target_entity": str(target.name),
                "source_type": str(source.label or ""),
                "target_type": str(target.label or ""),
                "fact_type": "relation",
                "source_fact_id": "",
                "source_uid": source_uid,
                "target_uid": target_uid,
                "chunk_uid": chunk_uid,
                "edge_uid": str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"edge:{source_uid}:{relation.label}:{target_uid}",
                    )
                ),
            }
            artifacts.facts.append(fact_payload)
            fact_records.append(
                {
                    "id": fact_id,
                    "text": text,
                    "relative_path": fact_payload["relative_path"],
                    "source_file": fact_payload["source_file"],
                    "heading_path": fact_payload["heading_path"],
                    "chunk_id": fact_payload["chunk_id"],
                    "source_year": fact_payload["source_year"],
                    "relation_label": relation.label,
                    "source_entity": source.name,
                    "target_entity": target.name,
                    "source_type": source_record["label"],
                    "target_type": target_record["label"],
                    "fact_type": "relation",
                }
            )
            created += 1

        if created > 0:
            continue

        fallback_text = node.get_content().strip()
        if not fallback_text:
            continue

        fact_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{node.node_id}:fallback"))
        fallback_payload = {
            "id": fact_id,
            "text": fallback_text,
            "relative_path": str(metadata.get("relative_path") or ""),
            "source_file": str(metadata.get("source_file") or ""),
            "heading_path": _stringify_heading_path(metadata.get("heading_path")),
            "chunk_id": str(metadata.get("chunk_id") or ""),
            "source_year": str(metadata.get("source_year") or ""),
            "relation_label": "FALLBACK_CHUNK",
            "source_entity": "",
            "target_entity": "",
            "source_type": "",
            "target_type": "",
            "fact_type": "chunk_fallback",
            "source_fact_id": "",
            "source_uid": "",
            "target_uid": "",
            "chunk_uid": chunk_uid,
            "edge_uid": "",
        }
        artifacts.facts.append(fallback_payload)
        fact_records.append(
            {
                "id": fact_id,
                "text": fallback_text,
                "relative_path": fallback_payload["relative_path"],
                "source_file": fallback_payload["source_file"],
                "heading_path": fallback_payload["heading_path"],
                "chunk_id": fallback_payload["chunk_id"],
                "source_year": fallback_payload["source_year"],
                "relation_label": "FALLBACK_CHUNK",
                "source_entity": "",
                "target_entity": "",
                "source_type": "",
                "target_type": "",
                "fact_type": "chunk_fallback",
            }
        )

    return artifacts, fact_records


# Ghi file audit JSONL để dễ xem fact nào đã được trích trước khi import vào Neo4j.
def write_fact_records(records: list[dict], output_path: Path) -> None:
    """Lưu fact audit ra JSONL để tiện kiểm tra dữ liệu trích xuất."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# Ghép một relation thành câu fact tự nhiên để vector search xử lý tốt hơn.
def _build_relation_text(*, source_name: str, relation_label: str, target_name: str, heading_context: Any) -> str:
    """Chuyển relation graph thành câu fact ngắn, dễ truy vấn bằng embedding."""

    relation_text = RELATION_TEXT.get(relation_label, relation_label.replace("_", " ").lower())
    context = _stringify_heading_path(heading_context)
    base = f"{source_name} {relation_text} {target_name}."
    if context:
        base += f" Bối cảnh: {context}"
    return base.strip()


# Chuẩn hóa entity extractor trả về thành payload sẵn sàng ghi vào Neo4j.
def _entity_record(entity) -> dict[str, Any]:
    """Đổi entity từ extractor sang dict metadata ổn định."""

    properties = dict(getattr(entity, "properties", {}) or {})
    return {
        "uid": str(entity.id),
        "name": str(entity.name),
        "name_normalized": _normalize_text(str(entity.name)),
        "label": str(getattr(entity, "label", "") or ""),
        "source_year": str(properties.get("source_year") or ""),
        "relative_path": str(properties.get("relative_path") or ""),
        "source_file": str(properties.get("source_file") or ""),
        "heading_path": _stringify_heading_path(properties.get("heading_path")),
    }


# Tạo khóa chunk ổn định để cùng một chunk luôn ánh xạ về đúng node Chunk.
def _chunk_uid(metadata: dict[str, Any]) -> str:
    """Sinh UID ổn định cho node Chunk trong graph."""

    relative_path = str(metadata.get("relative_path") or metadata.get("source_file") or "unknown")
    chunk_id = str(metadata.get("chunk_id") or "unknown")
    return f"{relative_path}::chunk::{chunk_id}"


# Chuẩn hóa heading path về chuỗi dễ lưu và dễ hiển thị.
def _stringify_heading_path(value: Any) -> str:
    """Chuyển heading path từ list hoặc giá trị bất kỳ thành chuỗi thống nhất."""

    if isinstance(value, list):
        return " | ".join(str(item) for item in value if str(item).strip())
    return str(value or "")


# Chuẩn hóa text về lowercase một dòng để phục vụ tìm kiếm fulltext.
def _normalize_text(value: str) -> str:
    """Chuẩn hóa text để tăng độ ổn định cho so khớp tên entity."""

    return re.sub(r"\s+", " ", value).strip().lower()
