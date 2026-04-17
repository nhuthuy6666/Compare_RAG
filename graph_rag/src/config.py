from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings
from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llamaindex_shared.benchmark_runtime import normalize_runtime_overrides
from openai_compatible import OpenAICompatibleEmbedding, OpenAICompatibleLLM


DEFAULT_BASELINE_CONFIG = PROJECT_ROOT.parent / "extract_md" / "rag_baseline.json"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_CHUNK_DIR = DEFAULT_PROCESSED_DIR / "chunks"
DEFAULT_FACTS_PATH = DEFAULT_PROCESSED_DIR / "graph_facts.jsonl"


@dataclass(frozen=True)
class AppConfig:
    """Gói toàn bộ cấu hình runtime cho GraphRAG Neo4j."""

    project_root: Path
    baseline_config_path: Path
    source_chunk_root: Path
    corpus_scope: str
    processed_dir: Path
    chunk_dir: Path
    facts_path: Path
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str
    neo4j_entity_index: str
    neo4j_fact_vector_index: str
    neo4j_chunk_index: str
    graph_vector_candidates: int
    graph_neighbor_hops: int
    graph_neighbor_facts_limit: int
    llm_base_url: str | None
    llm_model: str | None
    embed_model: str | None
    llm_api_key: str | None
    llm_timeout: int
    embed_timeout: int
    embed_batch_size: int
    embed_retry_attempts: int
    embed_retry_delay: int
    chunk_min_chars: int
    chunk_max_chars: int
    max_sections_per_chunk: int
    retrieval_top_n: int
    retrieval_similarity_threshold: float
    query_fusion_enabled: bool
    query_fusion_num_queries: int
    query_fusion_mode: str
    generation_temperature: float
    generation_top_p: float
    max_output_tokens: int
    llm_seed: int | None
    shared_prompt: str
    query_refusal_response: str
    graph_progress_every: int


# Đọc biến môi trường kiểu số nguyên với fallback rõ ràng.
def _get_int_env(name: str, default: int) -> int:
    """Đọc biến môi trường số nguyên và báo lỗi nếu giá trị không hợp lệ."""

    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got: {raw!r}") from exc


# Đọc biến môi trường kiểu số thực với fallback rõ ràng.
def _get_float_env(name: str, default: float) -> float:
    """Đọc biến môi trường số thực và báo lỗi nếu giá trị không hợp lệ."""

    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float, got: {raw!r}") from exc


# Quy đổi đường dẫn tương đối thành đường dẫn tuyệt đối theo thư mục gốc được chỉ định.
def _resolve_project_path(path_like: str | Path, *, base_dir: Path) -> Path:
    """Chuẩn hóa đường dẫn config để dùng ổn định ở mọi thư mục chạy lệnh."""

    path = Path(path_like)
    return path if path.is_absolute() else base_dir / path


# Đọc biến môi trường boolean theo các alias thông dụng như true/false, on/off.
def _get_bool_env(name: str, default: bool) -> bool:
    """Đọc biến môi trường boolean và hỗ trợ nhiều cách ghi phổ biến."""

    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Environment variable {name} must be a boolean, got: {raw!r}")


# Đọc file baseline JSON để kế thừa cấu hình chung của toàn project.
def _load_baseline_config(path: Path) -> dict:
    """Đọc cấu hình baseline JSON và trả về dict rỗng nếu file không tồn tại."""

    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Baseline config is not valid JSON: {path}") from exc


# Gộp baseline config, biến môi trường và benchmark override thành một cấu hình thống nhất.
def load_config(overrides: dict | None = None) -> AppConfig:
    """Tạo AppConfig hoàn chỉnh cho GraphRAG Neo4j từ mọi nguồn cấu hình."""

    load_dotenv()
    baseline_config_path = Path(os.getenv("BASELINE_CONFIG_PATH", DEFAULT_BASELINE_CONFIG))
    baseline = _load_baseline_config(baseline_config_path)
    corpus = baseline.get("corpus") or {}
    models = baseline.get("models") or {}
    retrieval = baseline.get("retrieval") or {}
    fusion = retrieval.get("fusion") or {}
    generation = baseline.get("generation") or {}

    seed_env = os.getenv("LLM_SEED")
    config = AppConfig(
        project_root=PROJECT_ROOT,
        baseline_config_path=baseline_config_path,
        source_chunk_root=_resolve_project_path(
            os.getenv("CHUNK_JSONL_ROOT", str(corpus.get("chunk_root") or "extract_md/data_chunks")),
            base_dir=PROJECT_ROOT.parent,
        ),
        corpus_scope=os.getenv("CORPUS_SCOPE", str(corpus.get("scope") or "web")),
        processed_dir=Path(os.getenv("PROCESSED_DIR", DEFAULT_PROCESSED_DIR)),
        chunk_dir=Path(os.getenv("CHUNK_DIR", DEFAULT_CHUNK_DIR)),
        facts_path=Path(os.getenv("GRAPH_FACTS_PATH", DEFAULT_FACTS_PATH)),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
        neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "12345678"),
        neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
        neo4j_entity_index=os.getenv("NEO4J_ENTITY_INDEX", "entity_name_index"),
        neo4j_fact_vector_index=os.getenv("NEO4J_FACT_VECTOR_INDEX", "fact_embedding_index"),
        neo4j_chunk_index=os.getenv("NEO4J_CHUNK_INDEX", "chunk_lookup_index"),
        graph_vector_candidates=_get_int_env("GRAPH_VECTOR_CANDIDATES", 18),
        graph_neighbor_hops=_get_int_env("GRAPH_NEIGHBOR_HOPS", 1),
        graph_neighbor_facts_limit=_get_int_env("GRAPH_NEIGHBOR_FACTS_LIMIT", 12),
        llm_base_url=os.getenv("LLM_BASE_URL"),
        llm_model=os.getenv("LLM_MODEL", str(models.get("chat") or "")),
        embed_model=os.getenv("EMBED_MODEL", str(models.get("embedding") or "")),
        llm_api_key=os.getenv("LLM_API_KEY"),
        llm_timeout=_get_int_env("LLM_TIMEOUT", 900),
        embed_timeout=_get_int_env("EMBED_TIMEOUT", 1800),
        embed_batch_size=_get_int_env("EMBED_BATCH_SIZE", 4),
        embed_retry_attempts=_get_int_env("EMBED_RETRY_ATTEMPTS", 8),
        embed_retry_delay=_get_int_env("EMBED_RETRY_DELAY", 8),
        chunk_min_chars=_get_int_env("CHUNK_MIN_CHARS", int(corpus.get("min_chars") or 900)),
        chunk_max_chars=_get_int_env("CHUNK_MAX_CHARS", int(corpus.get("max_chars") or 2800)),
        max_sections_per_chunk=_get_int_env(
            "MAX_SECTIONS_PER_CHUNK",
            int(corpus.get("max_sections_per_chunk") or 5),
        ),
        retrieval_top_n=_get_int_env("RETRIEVAL_TOP_N", int(retrieval.get("top_n") or 6)),
        retrieval_similarity_threshold=_get_float_env(
            "RETRIEVAL_SIMILARITY_THRESHOLD",
            float(retrieval.get("similarity_threshold") or 0.0),
        ),
        query_fusion_enabled=_get_bool_env("QUERY_FUSION_ENABLED", bool(fusion.get("enabled", True))),
        query_fusion_num_queries=max(
            1,
            _get_int_env("QUERY_FUSION_NUM_QUERIES", int(fusion.get("num_queries") or 4)),
        ),
        query_fusion_mode=os.getenv("QUERY_FUSION_MODE", str(fusion.get("mode") or "reciprocal_rank")).strip(),
        generation_temperature=_get_float_env(
            "GENERATION_TEMPERATURE",
            float(generation.get("temperature") or 0.1),
        ),
        generation_top_p=_get_float_env("GENERATION_TOP_P", float(generation.get("top_p") or 1.0)),
        max_output_tokens=_get_int_env("MAX_OUTPUT_TOKENS", int(generation.get("max_tokens") or 1024)),
        llm_seed=int(seed_env) if seed_env is not None and seed_env.strip() else None,
        shared_prompt=os.getenv("SHARED_PROMPT", str(baseline.get("prompt") or "").strip()),
        query_refusal_response=os.getenv(
            "QUERY_REFUSAL_RESPONSE",
            str(baseline.get("query_refusal_response") or "").strip(),
        ),
        graph_progress_every=_get_int_env("GRAPH_PROGRESS_EVERY", 5),
    )
    runtime_overrides = normalize_runtime_overrides(overrides)
    if not runtime_overrides:
        return config
    supported = {key: value for key, value in runtime_overrides.items() if key in config.__dataclass_fields__}
    return replace(config, **supported)


# Kiểm tra các giá trị bắt buộc trước khi cấu hình model.
def _require(values: dict[str, str | None]) -> None:
    """Đảm bảo các biến cấu hình bắt buộc đã có giá trị."""

    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


# Áp cấu hình LLM và embedding vào Settings để toàn bộ pipeline Neo4j dùng chung.
def configure_models(config: AppConfig) -> None:
    """Khởi tạo LLM và embedding model tương thích OpenAI cho GraphRAG."""

    _require(
        {
            "LLM_MODEL": config.llm_model,
            "EMBED_MODEL": config.embed_model,
        }
    )
    api_key = config.llm_api_key or "ollama"
    Settings.llm = OpenAICompatibleLLM(
        model_name=config.llm_model,
        api_key=api_key,
        api_base=config.llm_base_url,
        timeout=float(config.llm_timeout),
        max_tokens=config.max_output_tokens,
        temperature=config.generation_temperature,
        top_p=config.generation_top_p,
        seed=config.llm_seed,
    )
    Settings.embed_model = OpenAICompatibleEmbedding(
        model_name=config.embed_model,
        api_key=api_key,
        api_base=config.llm_base_url,
        timeout=float(config.embed_timeout),
        embed_batch_size=config.embed_batch_size,
        retry_attempts=config.embed_retry_attempts,
        retry_delay=float(config.embed_retry_delay),
    )


# Tạo Neo4j driver dùng chung cho ingest và query.
def build_neo4j_driver(config: AppConfig):
    """Khởi tạo Neo4j driver theo URI và thông tin xác thực hiện tại."""

    return GraphDatabase.driver(
        config.neo4j_uri,
        auth=(config.neo4j_username, config.neo4j_password),
    )


# Kiểm tra kết nối Neo4j sớm để báo lỗi rõ ràng trước khi ingest hoặc serve request.
def verify_neo4j_connection(config: AppConfig) -> None:
    """Kiểm tra Neo4j đã sẵn sàng trước khi dùng trong pipeline."""

    driver = build_neo4j_driver(config)
    try:
        driver.verify_connectivity()
        with driver.session(database=config.neo4j_database) as session:
            session.run("RETURN 1 AS ok").single()
    finally:
        driver.close()
