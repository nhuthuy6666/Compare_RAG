from __future__ import annotations

import json
import uuid
from pathlib import Path

from llama_index.core.indices.property_graph.base import KG_NODES_KEY, KG_RELATIONS_KEY
from llama_index.core.schema import TextNode

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


## Map tên quan hệ nội bộ sang câu tiếng Việt tự nhiên hơn để đưa vào fact text.
def _relation_text(label: str) -> str:
    if label in RELATION_TEXT:
        return RELATION_TEXT[label]
    return label.replace("_", " ").lower()


## Chuyển chunk records thành các graph fact nodes và payload audit tương ứng.
## Nếu một chunk không tạo ra relation rõ ràng, hàm sẽ fallback sang fact dạng nguyên chunk.
def build_graph_fact_nodes(records: list[dict], progress_every: int) -> tuple[list[TextNode], list[dict]]:
    extracted_nodes = AdmissionsGraphExtractor(progress_every=progress_every)(records_to_nodes(records))
    fact_nodes: list[TextNode] = []
    fact_records: list[dict] = []

    for node in extracted_nodes:
        metadata = dict(node.metadata or {})
        entities = {
            entity.id: entity
            for entity in (metadata.get(KG_NODES_KEY) or [])
            if getattr(entity, "id", None) and getattr(entity, "name", None)
        }
        relations = list(metadata.get(KG_RELATIONS_KEY) or [])
        created = 0

        for relation in relations:
            source = entities.get(relation.source_id)
            target = entities.get(relation.target_id)
            if source is None or target is None:
                continue

            heading_context = metadata.get("heading_path") or metadata.get("title") or metadata.get("source_file") or ""
            text = f"{source.name} {_relation_text(relation.label)} {target.name}. Bối cảnh: {heading_context}".strip()
            fact_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{node.node_id}:{relation.source_id}:{relation.label}:{relation.target_id}",
                )
            )
            fact_metadata = {
                "relative_path": metadata.get("relative_path", ""),
                "source_file": metadata.get("source_file", ""),
                "heading_path": metadata.get("heading_path", ""),
                "chunk_id": metadata.get("chunk_id", ""),
                "source_year": metadata.get("source_year", ""),
                "relation_label": relation.label,
                "source_entity": source.name,
                "target_entity": target.name,
                "fact_type": "relation",
            }
            fact_nodes.append(
                TextNode(
                    id_=fact_id,
                    text=text,
                    metadata=fact_metadata,
                    excluded_embed_metadata_keys=list(fact_metadata.keys()),
                    excluded_llm_metadata_keys=list(fact_metadata.keys()),
                )
            )
            fact_records.append({"id": fact_id, "text": text, **fact_metadata})
            created += 1

        if created > 0:
            continue

        fallback_text = node.get_content().strip()
        if not fallback_text:
            continue

        fact_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{node.node_id}:fallback"))
        fact_metadata = {
            "relative_path": metadata.get("relative_path", ""),
            "source_file": metadata.get("source_file", ""),
            "heading_path": metadata.get("heading_path", ""),
            "chunk_id": metadata.get("chunk_id", ""),
            "source_year": metadata.get("source_year", ""),
            "relation_label": "FALLBACK_CHUNK",
            "source_entity": "",
            "target_entity": "",
            "fact_type": "chunk_fallback",
        }
        fact_nodes.append(
            TextNode(
                id_=fact_id,
                text=fallback_text,
                metadata=fact_metadata,
                excluded_embed_metadata_keys=list(fact_metadata.keys()),
                excluded_llm_metadata_keys=list(fact_metadata.keys()),
            )
        )
        fact_records.append({"id": fact_id, "text": fallback_text, **fact_metadata})

    return fact_nodes, fact_records


## Ghi các graph fact đã trích ra JSONL để audit/debug và so sánh giữa các lần ingest.
def write_fact_records(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
