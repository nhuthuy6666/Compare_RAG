from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievedFact:
    """Biểu diễn một fact đã retrieve từ Neo4j cùng metadata cần cho answer synthesis."""

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
    """Tạo các constraint và index Neo4j mà GraphRAG cần để truy vấn nhanh."""

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
    """Xóa toàn bộ node và relationship hiện có trong database Neo4j."""

    with driver.session(database=config.neo4j_database) as session:
        session.run("MATCH (n) DETACH DELETE n")


# Ghi entity, chunk và fact vào Neo4j theo batch để ingest ổn định hơn.
def upsert_graph_artifacts(driver, config, artifacts, *, batch_size: int = 200) -> None:
    """Ghi toàn bộ graph artifacts vào Neo4j theo lô nhỏ."""

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
    """Thực hiện vector search trên Neo4j để lấy các fact gần nghĩa nhất."""

    query = """
    CALL db.index.vector.queryNodes($index_name, $limit, $embedding)
    YIELD node, score
    RETURN
      node.id AS fact_id,
      node.text AS text,
      coalesce(node.relative_path, '') AS relative_path,
      coalesce(node.heading_path, '') AS heading_path,
      coalesce(toString(node.chunk_id), '') AS chunk_id,
      coalesce(node.relation_label, '') AS relation_label,
      coalesce(node.source_entity, '') AS source_entity,
      coalesce(node.target_entity, '') AS target_entity,
      coalesce(node.source_type, '') AS source_type,
      coalesce(node.target_type, '') AS target_type,
      coalesce(node.source_fact_id, '') AS source_fact_id,
      score AS score
    ORDER BY score DESC
    """
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
    """Dùng fulltext entity search để kéo thêm fact bám sát thực thể trong câu hỏi."""

    if not query_text.strip():
        return []
    cypher = """
    CALL db.index.fulltext.queryNodes($index_name, $query_text, {limit: $entity_limit})
    YIELD node, score
    MATCH (node)-[:SOURCE_OF|TARGET_OF]->(fact:Fact)
    RETURN
      fact.id AS fact_id,
      fact.text AS text,
      coalesce(fact.relative_path, '') AS relative_path,
      coalesce(fact.heading_path, '') AS heading_path,
      coalesce(toString(fact.chunk_id), '') AS chunk_id,
      coalesce(fact.relation_label, '') AS relation_label,
      coalesce(fact.source_entity, '') AS source_entity,
      coalesce(fact.target_entity, '') AS target_entity,
      coalesce(fact.source_type, '') AS source_type,
      coalesce(fact.target_type, '') AS target_type,
      coalesce(fact.source_fact_id, '') AS source_fact_id,
      max(score) AS score
    ORDER BY score DESC
    LIMIT $limit
    """
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
    """Lấy thêm fact lân cận của các fact seed dựa trên entity chung trong graph."""

    if not seed_fact_ids:
        return []
    cypher = """
    UNWIND $seed_fact_ids AS seed_id
    MATCH (seed:Fact {id: seed_id})<-[:SOURCE_OF|TARGET_OF]-(entity:Entity)-[:SOURCE_OF|TARGET_OF]->(fact:Fact)
    WHERE fact.id <> seed_id
    RETURN
      fact.id AS fact_id,
      fact.text AS text,
      coalesce(fact.relative_path, '') AS relative_path,
      coalesce(fact.heading_path, '') AS heading_path,
      coalesce(toString(fact.chunk_id), '') AS chunk_id,
      coalesce(fact.relation_label, '') AS relation_label,
      coalesce(fact.source_entity, '') AS source_entity,
      coalesce(fact.target_entity, '') AS target_entity,
      coalesce(fact.source_type, '') AS source_type,
      coalesce(fact.target_type, '') AS target_type,
      coalesce(fact.source_fact_id, '') AS source_fact_id,
      count(*) AS score
    ORDER BY score DESC
    LIMIT $limit
    """
    with driver.session(database=config.neo4j_database) as session:
        rows = session.run(
            cypher,
            seed_fact_ids=seed_fact_ids,
            limit=limit,
        ).data()
    return [_row_to_fact(row, retrieval_stage="neighbor") for row in rows]


# Kiểm tra graph đã có dữ liệu hay chưa trước khi nhận request chat.
def graph_ready(driver, config) -> bool:
    """Kiểm tra database Neo4j đã có ít nhất một fact hay chưa."""

    with driver.session(database=config.neo4j_database) as session:
        row = session.run("MATCH (f:Fact) RETURN count(f) AS total").single()
    return bool(row and int(row["total"]) > 0)


# MERGE toàn bộ Entity node của batch hiện tại.
def _merge_entities(tx, rows: list[dict[str, Any]]) -> None:
    """Ghi batch entity vào Neo4j bằng MERGE theo uid."""

    tx.run(
        """
        UNWIND $rows AS row
        MERGE (e:Entity {uid: row.uid})
        SET e.name = row.name,
            e.name_normalized = row.name_normalized,
            e.label = row.label,
            e.source_year = row.source_year,
            e.relative_path = row.relative_path,
            e.source_file = row.source_file,
            e.heading_path = row.heading_path
        """,
        rows=rows,
    )


# MERGE toàn bộ Chunk node của batch hiện tại.
def _merge_chunks(tx, rows: list[dict[str, Any]]) -> None:
    """Ghi batch chunk vào Neo4j bằng MERGE theo uid."""

    tx.run(
        """
        UNWIND $rows AS row
        MERGE (c:Chunk {uid: row.uid})
        SET c.relative_path = row.relative_path,
            c.source_file = row.source_file,
            c.heading_path = row.heading_path,
            c.chunk_id = row.chunk_id,
            c.source_year = row.source_year
        """,
        rows=rows,
    )


# MERGE toàn bộ Fact node và các relationship liên quan trong batch hiện tại.
def _merge_facts(tx, rows: list[dict[str, Any]]) -> None:
    """Ghi batch fact cùng các cạnh Entity-Chunk-Fact vào Neo4j."""

    tx.run(
        """
        UNWIND $rows AS row
        MERGE (f:Fact {id: row.id})
        SET f.text = row.text,
            f.embedding = row.embedding,
            f.relative_path = row.relative_path,
            f.source_file = row.source_file,
            f.heading_path = row.heading_path,
            f.chunk_id = row.chunk_id,
            f.source_year = row.source_year,
            f.relation_label = row.relation_label,
            f.source_entity = row.source_entity,
            f.target_entity = row.target_entity,
            f.source_type = row.source_type,
            f.target_type = row.target_type,
            f.fact_type = row.fact_type,
            f.source_fact_id = row.source_fact_id
        WITH f, row
        MATCH (c:Chunk {uid: row.chunk_uid})
        MERGE (f)-[:FROM_CHUNK]->(c)
        FOREACH (_ IN CASE WHEN row.source_uid = '' THEN [] ELSE [1] END |
            MERGE (source:Entity {uid: row.source_uid})
            MERGE (source)-[:SOURCE_OF]->(f)
        )
        FOREACH (_ IN CASE WHEN row.target_uid = '' THEN [] ELSE [1] END |
            MERGE (target:Entity {uid: row.target_uid})
            MERGE (target)-[:TARGET_OF]->(f)
        )
        FOREACH (_ IN CASE WHEN row.source_uid = '' OR row.target_uid = '' THEN [] ELSE [1] END |
            MERGE (source:Entity {uid: row.source_uid})
            MERGE (target:Entity {uid: row.target_uid})
            MERGE (source)-[rel:RELATES_TO {uid: row.edge_uid}]->(target)
            SET rel.label = row.relation_label,
                rel.fact_id = row.id
        )
        """,
        rows=rows,
    )


# Chia list đầu vào thành các batch nhỏ để hạn chế transaction quá lớn.
def _iter_batches(items: list[dict[str, Any]], batch_size: int):
    """Yield từng batch nhỏ từ danh sách đầu vào."""

    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


# Chuẩn hóa một dòng Cypher trả về thành `RetrievedFact`.
def _row_to_fact(row: dict[str, Any], *, retrieval_stage: str) -> RetrievedFact:
    """Đổi một row Neo4j thành đối tượng RetrievedFact dùng chung ở tầng retrieval."""

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
