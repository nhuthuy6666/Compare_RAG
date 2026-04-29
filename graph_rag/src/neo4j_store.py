from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# Biểu diễn một fact đã retrieve từ Neo4j cùng metadata cần cho answer synthesis.
@dataclass(frozen=True)
class RetrievedFact:

    fact_id: str
    text: str
    relative_path: str
    heading_path: str
    chunk_id: str
    relation_label: str
    source_entity: str
    target_entity: str
    source_type: str
    target_type: str
    source_fact_id: str
    score: float
    retrieval_stage: str


# Tạo constraint và index cho Neo4j một lần trước khi ingest dữ liệu.
def ensure_schema(driver, config, *, embedding_dim: int) -> None:

    with driver.session(database=config.neo4j_database) as session:
        session.run(
            "CREATE CONSTRAINT graph_entity_uid IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.uid IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT graph_fact_id IF NOT EXISTS "
            "FOR (f:Fact) REQUIRE f.id IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT graph_chunk_uid IF NOT EXISTS "
            "FOR (c:Chunk) REQUIRE c.uid IS UNIQUE"
        )
        session.run(
            "CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS "
            "FOR (e:Entity) ON EACH [e.name, e.label, e.name_normalized]".format(
                index_name=config.neo4j_entity_index
            )
        )
        session.run(
            "CREATE RANGE INDEX {index_name} IF NOT EXISTS "
            "FOR (c:Chunk) ON (c.relative_path)".format(index_name=config.neo4j_chunk_index)
        )
        session.run(
            "CREATE VECTOR INDEX {index_name} IF NOT EXISTS "
            "FOR (f:Fact) ON (f.embedding) "
            "OPTIONS {{indexConfig: {{`vector.dimensions`: $embedding_dim, `vector.similarity_function`: 'cosine'}}}}".format(
                index_name=config.neo4j_fact_vector_index
            ),
            embedding_dim=embedding_dim,
        )
        session.run("CALL db.awaitIndexes(300)")


# Xóa toàn bộ graph hiện tại trước khi ingest lại từ shared chunks hoặc dữ liệu UI mới.
def reset_graph(driver, config) -> None:

    with driver.session(database=config.neo4j_database) as session:
        session.run("MATCH (n) DETACH DELETE n")


# Xóa dữ liệu graph theo danh sách tài liệu đã đổi/xóa trước khi upsert delta.
def delete_graph_for_relative_paths(driver, config, *, relative_paths: list[str]) -> None:

    normalized = sorted({str(path).strip() for path in relative_paths if str(path).strip()})
    if not normalized:
        return

    with driver.session(database=config.neo4j_database) as session:
        session.execute_write(_delete_paths, normalized)
        session.execute_write(_delete_orphan_entities)


# Ghi entity, chunk và fact vào Neo4j theo batch để ingest ổn định hơn.
def upsert_graph_artifacts(driver, config, artifacts, *, batch_size: int = 200) -> None:

    entity_rows = list(artifacts.entities.values())
    chunk_rows = list(artifacts.chunks.values())
    fact_rows = artifacts.facts

    with driver.session(database=config.neo4j_database) as session:
        for batch in _iter_batches(entity_rows, batch_size):
            session.execute_write(_merge_entities, batch)
        for batch in _iter_batches(chunk_rows, batch_size):
            session.execute_write(_merge_chunks, batch)
        for batch in _iter_batches(fact_rows, batch_size):
            session.execute_write(_merge_facts, batch)


# Lấy danh sách fact gần nghĩa nhất bằng vector index nằm trực tiếp trong Neo4j.
def vector_search(driver, config, *, embedding: list[float], limit: int) -> list[RetrievedFact]:

    query =             (
            '\n'
            '    CALL db.index.vector.queryNodes($index_name, $limit, $embedding)\n'
            '    YIELD node, score\n'
            '    RETURN\n'
            '      node.id AS fact_id,\n'
            '      node.text AS text,\n'
            "      coalesce(node.relative_path, '') AS relative_path,\n"
            "      coalesce(node.heading_path, '') AS heading_path,\n"
            "      coalesce(toString(node.chunk_id), '') AS chunk_id,\n"
            "      coalesce(node.relation_label, '') AS relation_label,\n"
            "      coalesce(node.source_entity, '') AS source_entity,\n"
            "      coalesce(node.target_entity, '') AS target_entity,\n"
            "      coalesce(node.source_type, '') AS source_type,\n"
            "      coalesce(node.target_type, '') AS target_type,\n"
            "      coalesce(node.source_fact_id, '') AS source_fact_id,\n"
            '      score AS score\n'
            '    ORDER BY score DESC\n'
            '    '
            )
    with driver.session(database=config.neo4j_database) as session:
        rows = session.run(
            query,
            index_name=config.neo4j_fact_vector_index,
            limit=limit,
            embedding=embedding,
        ).data()
    return [_row_to_fact(row, retrieval_stage="vector") for row in rows]


# Tìm entity liên quan đến câu hỏi rồi mở rộng sang các fact lân cận trong graph.
def entity_guided_search(driver, config, *, query_text: str, limit: int) -> list[RetrievedFact]:

    if not query_text.strip():
        return []
    cypher =              (
             '\n'
             '    CALL db.index.fulltext.queryNodes($index_name, $query_text, {limit: $entity_limit})\n'
             '    YIELD node, score\n'
             '    MATCH (node)-[:SOURCE_OF|TARGET_OF]->(fact:Fact)\n'
             '    RETURN\n'
             '      fact.id AS fact_id,\n'
             '      fact.text AS text,\n'
             "      coalesce(fact.relative_path, '') AS relative_path,\n"
             "      coalesce(fact.heading_path, '') AS heading_path,\n"
             "      coalesce(toString(fact.chunk_id), '') AS chunk_id,\n"
             "      coalesce(fact.relation_label, '') AS relation_label,\n"
             "      coalesce(fact.source_entity, '') AS source_entity,\n"
             "      coalesce(fact.target_entity, '') AS target_entity,\n"
             "      coalesce(fact.source_type, '') AS source_type,\n"
             "      coalesce(fact.target_type, '') AS target_type,\n"
             "      coalesce(fact.source_fact_id, '') AS source_fact_id,\n"
             '      max(score) AS score\n'
             '    ORDER BY score DESC\n'
             '    LIMIT $limit\n'
             '    '
             )
    with driver.session(database=config.neo4j_database) as session:
        rows = session.run(
            cypher,
            index_name=config.neo4j_entity_index,
            query_text=query_text,
            entity_limit=max(limit * 2, 8),
            limit=limit,
        ).data()
    return [_row_to_fact(row, retrieval_stage="entity") for row in rows]


# Mở rộng từ các fact seed sang fact anh em cùng entity để tận dụng cấu trúc graph.
def neighbor_search(driver, config, *, seed_fact_ids: list[str], limit: int) -> list[RetrievedFact]:

    if not seed_fact_ids:
        return []
    cypher =              (
             '\n'
             '    UNWIND $seed_fact_ids AS seed_id\n'
             '    MATCH (seed:Fact {id: seed_id})<-[:SOURCE_OF|TARGET_OF]-(entity:Entity)-[:SOURCE_OF|TARGET_OF]->(fact:Fact)\n'
             '    WHERE fact.id <> seed_id\n'
             '    RETURN\n'
             '      fact.id AS fact_id,\n'
             '      fact.text AS text,\n'
             "      coalesce(fact.relative_path, '') AS relative_path,\n"
             "      coalesce(fact.heading_path, '') AS heading_path,\n"
             "      coalesce(toString(fact.chunk_id), '') AS chunk_id,\n"
             "      coalesce(fact.relation_label, '') AS relation_label,\n"
             "      coalesce(fact.source_entity, '') AS source_entity,\n"
             "      coalesce(fact.target_entity, '') AS target_entity,\n"
             "      coalesce(fact.source_type, '') AS source_type,\n"
             "      coalesce(fact.target_type, '') AS target_type,\n"
             "      coalesce(fact.source_fact_id, '') AS source_fact_id,\n"
             '      count(*) AS score\n'
             '    ORDER BY score DESC\n'
             '    LIMIT $limit\n'
             '    '
             )
    with driver.session(database=config.neo4j_database) as session:
        rows = session.run(
            cypher,
            seed_fact_ids=seed_fact_ids,
            limit=limit,
        ).data()
    return [_row_to_fact(row, retrieval_stage="neighbor") for row in rows]


# Kiểm tra graph đã có dữ liệu hay chưa trước khi nhận request chat.
def graph_ready(driver, config) -> bool:

    with driver.session(database=config.neo4j_database) as session:
        row = session.run("MATCH (n) WHERE 'Fact' IN labels(n) RETURN count(n) AS total").single()
    return bool(row and int(row["total"]) > 0)


# MERGE toàn bộ Entity node của batch hiện tại.
def _merge_entities(tx, rows: list[dict[str, Any]]) -> None:

    tx.run(
                (
        '\n'
        '        UNWIND $rows AS row\n'
        '        MERGE (e:Entity {uid: row.uid})\n'
        '        SET e.name = row.name,\n'
        '            e.name_normalized = row.name_normalized,\n'
        '            e.label = row.label,\n'
        '            e.source_year = row.source_year,\n'
        '            e.relative_path = row.relative_path,\n'
        '            e.source_file = row.source_file,\n'
        '            e.heading_path = row.heading_path\n'
        '        '
        ),
        rows=rows,
    )


# MERGE toàn bộ Chunk node của batch hiện tại.
def _merge_chunks(tx, rows: list[dict[str, Any]]) -> None:

    tx.run(
                (
        '\n'
        '        UNWIND $rows AS row\n'
        '        MERGE (c:Chunk {uid: row.uid})\n'
        '        SET c.relative_path = row.relative_path,\n'
        '            c.source_file = row.source_file,\n'
        '            c.heading_path = row.heading_path,\n'
        '            c.chunk_id = row.chunk_id,\n'
        '            c.source_year = row.source_year\n'
        '        '
        ),
        rows=rows,
    )


# MERGE toàn bộ Fact node và các relationship liên quan trong batch hiện tại.
def _merge_facts(tx, rows: list[dict[str, Any]]) -> None:

    tx.run(
                (
        '\n'
        '        UNWIND $rows AS row\n'
        '        MERGE (f:Fact {id: row.id})\n'
        '        SET f.text = row.text,\n'
        '            f.embedding = row.embedding,\n'
        '            f.relative_path = row.relative_path,\n'
        '            f.source_file = row.source_file,\n'
        '            f.heading_path = row.heading_path,\n'
        '            f.chunk_id = row.chunk_id,\n'
        '            f.source_year = row.source_year,\n'
        '            f.relation_label = row.relation_label,\n'
        '            f.source_entity = row.source_entity,\n'
        '            f.target_entity = row.target_entity,\n'
        '            f.source_type = row.source_type,\n'
        '            f.target_type = row.target_type,\n'
        '            f.fact_type = row.fact_type,\n'
        '            f.source_fact_id = row.source_fact_id\n'
        '        WITH f, row\n'
        '        MATCH (c:Chunk {uid: row.chunk_uid})\n'
        '        MERGE (f)-[:FROM_CHUNK]->(c)\n'
        "        FOREACH (_ IN CASE WHEN row.source_uid = '' THEN [] ELSE [1] END |\n"
        '            MERGE (source:Entity {uid: row.source_uid})\n'
        '            MERGE (source)-[:SOURCE_OF]->(f)\n'
        '        )\n'
        "        FOREACH (_ IN CASE WHEN row.target_uid = '' THEN [] ELSE [1] END |\n"
        '            MERGE (target:Entity {uid: row.target_uid})\n'
        '            MERGE (target)-[:TARGET_OF]->(f)\n'
        '        )\n'
        "        FOREACH (_ IN CASE WHEN row.source_uid = '' OR row.target_uid = '' THEN [] ELSE [1] END |\n"
        '            MERGE (source:Entity {uid: row.source_uid})\n'
        '            MERGE (target:Entity {uid: row.target_uid})\n'
        '            MERGE (source)-[rel:RELATES_TO {uid: row.edge_uid}]->(target)\n'
        '            SET rel.label = row.relation_label,\n'
        '                rel.fact_id = row.id\n'
        '        )\n'
        '        '
        ),
        rows=rows,
    )


# Chia list đầu vào thành các batch nhỏ để hạn chế transaction quá lớn.
def _iter_batches(items: list[dict[str, Any]], batch_size: int):

    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _delete_paths(tx, relative_paths: list[str]) -> None:
    tx.run(
                (
        '\n'
        '        UNWIND $relative_paths AS relative_path\n'
        '        OPTIONAL MATCH (f:Fact {relative_path: relative_path})\n'
        '        DETACH DELETE f\n'
        '        '
        ),
        relative_paths=relative_paths,
    )
    tx.run(
                (
        '\n'
        '        UNWIND $relative_paths AS relative_path\n'
        '        OPTIONAL MATCH (c:Chunk {relative_path: relative_path})\n'
        '        DETACH DELETE c\n'
        '        '
        ),
        relative_paths=relative_paths,
    )


def _delete_orphan_entities(tx) -> None:
    tx.run(
                (
        '\n'
        '        MATCH (e:Entity)\n'
        '        WHERE NOT (e)-[:SOURCE_OF|TARGET_OF]->(:Fact)\n'
        '        DETACH DELETE e\n'
        '        '
        )
    )


# Chuẩn hóa một dòng Cypher trả về thành `RetrievedFact`.
def _row_to_fact(row: dict[str, Any], *, retrieval_stage: str) -> RetrievedFact:

    return RetrievedFact(
        fact_id=str(row.get("fact_id") or ""),
        text=str(row.get("text") or ""),
        relative_path=str(row.get("relative_path") or ""),
        heading_path=str(row.get("heading_path") or ""),
        chunk_id=str(row.get("chunk_id") or ""),
        relation_label=str(row.get("relation_label") or ""),
        source_entity=str(row.get("source_entity") or ""),
        target_entity=str(row.get("target_entity") or ""),
        source_type=str(row.get("source_type") or ""),
        target_type=str(row.get("target_type") or ""),
        source_fact_id=str(row.get("source_fact_id") or ""),
        score=float(row.get("score") or 0.0),
        retrieval_stage=retrieval_stage,
    )
