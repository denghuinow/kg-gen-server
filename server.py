from __future__ import annotations

import asyncio
import colorsys
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from collections import Counter, defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest, urlopen

import yaml
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from itext2kg.atom import Atom
from itext2kg.atom.models import AtomicFact, KnowledgeGraph
from itext2kg.atom.models.entity import Entity, EntityProperties
from itext2kg.atom.models.relationship import Relationship, RelationshipProperties
from itext2kg.graph_integration.neo4j_storage import Neo4jStorage
from itext2kg.llm_output_parsing.langchain_output_parser import LangchainOutputParser

BASE_DIR = Path(__file__).resolve().parent
VISUALIZE_TEMPLATE_PATH = BASE_DIR / "templates" / "visualize.html"


class BuildGraphRequest(BaseModel):
    graph_name: str = Field(..., min_length=1)
    doc_id: str = Field(..., min_length=1)
    doc_name: str = Field(..., min_length=1)
    doc_content: str = Field(..., min_length=1, description="Markdown 文本")
    new_graph_name: Optional[str] = Field(None, min_length=1)


class BuildGraphResponse(BaseModel):
    success: bool
    graph_name: str
    base_graph_name: str
    entities: int
    relationships: int
    atomic_facts: int
    obs_timestamp: str


class RenameGraphRequest(BaseModel):
    new_name: str = Field(..., min_length=1)


class DeleteGraphResponse(BaseModel):
    success: bool
    message: str
    deleted_nodes: int
    deleted_relationships: int


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_value(section: Dict[str, Any], key: str, required: bool = False) -> Optional[str]:
    value = section.get(key)
    if value is not None and str(value).strip() != "":
        return str(value)
    env_key = section.get(f"{key}_env")
    if env_key:
        env_value = os.getenv(str(env_key))
        if env_value is not None and env_value.strip() != "":
            return env_value
    if required:
        raise ValueError(f"配置字段缺失: {key} / {key}_env")
    return None


def setup_logging(config: Dict[str, Any]) -> None:
    level_name = str(config.get("logging", {}).get("level", "INFO"))
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def chunk_text_with_chonkie_fastapi(text: str, chunker_cfg: Dict[str, Any]) -> List[str]:
    url = str(chunker_cfg.get("url", "")).strip()
    if not url:
        raise ValueError("text.chunker.url 未配置（需要指向 chonkie-fastapi 的 /SentenceChunker）")

    timeout_s = float(chunker_cfg.get("timeout_s", 60))
    payload: Dict[str, Any] = {
        "text": text,
        "chunk_size": int(chunker_cfg.get("chunk_size", 512)),
        "chunk_overlap": int(chunker_cfg.get("chunk_overlap", 20)),
        "min_sentences_per_chunk": int(chunker_cfg.get("min_sentences_per_chunk", 1)),
        "min_characters_per_sentence": chunker_cfg.get("min_characters_per_sentence", 12),
        "approximate": bool(chunker_cfg.get("approximate", True)),
        "delim": chunker_cfg.get("delim", [".", "?", "!", "。", "？", "！", "\n"]),
        "include_delim": chunker_cfg.get("include_delim", "prev"),
        "return_type": "chunks",
    }

    tokenizer = chunker_cfg.get("tokenizer")
    if tokenizer:
        payload["tokenizer"] = tokenizer

    req = UrlRequest(
        url=url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )

    try:
        with urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        raise RuntimeError(f"chonkie-fastapi 请求失败（HTTP {e.code}）：{detail}") from e
    except URLError as e:
        raise RuntimeError(f"无法连接 chonkie-fastapi：{e}. 请确认服务已启动且 url 可访问：{url}") from e

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"chonkie-fastapi 返回非 JSON：{raw[:500]}") from e

    chunks = data.get("chunks")
    if not isinstance(chunks, list):
        raise RuntimeError(f"chonkie-fastapi 响应缺少 chunks 字段：{data}")

    sections: List[str] = []
    for item in chunks:
        if not isinstance(item, dict):
            continue
        text_part = str(item.get("text", "")).strip()
        if text_part:
            sections.append(text_part)
    if not sections:
        raise RuntimeError("chonkie-fastapi 未返回任何有效分块（chunks 为空或 text 为空）")
    return sections


async def extract_atomic_facts(
    parser: LangchainOutputParser,
    sections: List[str],
    obs_timestamp: str,
    output_language: str,
    entity_name_mode: str,
) -> List[str]:
    if not sections:
        return []

    contexts = [
        f"observation_date: {obs_timestamp}\n\nparagraph:\n{section.strip()}"
        for section in sections
        if section.strip()
    ]
    if not contexts:
        return []

    system_query = None
    if output_language.lower().startswith("zh") and entity_name_mode == "source":
        system_query = f"""
你是一个“原子事实（atomic facts）”抽取器。
请基于给定的 paragraph 与 observation_date 抽取事实列表，遵守以下要求：
- 输出语言使用中文。
- 涉及到的人名/机构名/术语等专有名词，必须与原文一致：不要翻译成英文、不要拼音化、不要改写。
- 不要添加原文未明确提及的信息；不要输出解释，只输出结构化结果需要的内容。
- 时间表达如果出现相对时间（如“去年/明年”），请结合 observation_date 转换为绝对日期。

observation_date: {obs_timestamp}
"""

    if system_query:
        outputs = await parser.extract_information_as_json_for_context(
            AtomicFact, contexts, system_query=system_query
        )
    else:
        outputs = await parser.extract_information_as_json_for_context(AtomicFact, contexts)

    facts: List[str] = []
    for block in outputs:
        if not block:
            continue
        for fact in getattr(block, "atomic_fact", []) or []:
            cleaned = str(fact).strip()
            if cleaned:
                facts.append(cleaned)
    return facts


def init_atom_and_parser(config: Dict[str, Any]) -> Tuple[Atom, LangchainOutputParser]:
    llm_cfg = config.get("llm", {}) or {}
    embeddings_cfg = config.get("embeddings", {}) or {}

    api_key = resolve_value(llm_cfg, "api_key", required=True)
    llm_model_name = resolve_value(llm_cfg, "model", required=True)
    embeddings_api_key = resolve_value(embeddings_cfg, "api_key", required=True)
    embeddings_model_name = resolve_value(embeddings_cfg, "model", required=True)

    llm = ChatOpenAI(
        api_key=api_key,
        model=llm_model_name,
        temperature=float(llm_cfg.get("temperature", 0)),
        max_retries=int(llm_cfg.get("max_retries", 2)),
        base_url=resolve_value(llm_cfg, "api_base_url"),
    )

    embeddings = OpenAIEmbeddings(
        api_key=embeddings_api_key,
        model=embeddings_model_name,
        base_url=resolve_value(embeddings_cfg, "api_base_url"),
    )

    batch_cfg = config.get("batch_processing", {}) or {}
    sleep_time = int(batch_cfg.get("sleep_time", 5))
    lg_cfg = batch_cfg.get("langchain_output_parser", {}) or {}

    sleep_between_batches = lg_cfg.get("sleep_between_batches")
    max_concurrency = lg_cfg.get("max_concurrency")
    max_elements_per_batch = lg_cfg.get("max_elements_per_batch")
    max_tokens_per_batch = lg_cfg.get("max_tokens_per_batch")

    parser = LangchainOutputParser(
        llm_model=llm,
        embeddings_model=embeddings,
        sleep_time=sleep_time,
        sleep_between_batches=float(sleep_between_batches) if sleep_between_batches is not None else None,
        max_concurrency=int(max_concurrency) if max_concurrency is not None else None,
        max_elements_per_batch=int(max_elements_per_batch) if max_elements_per_batch is not None else None,
        max_tokens_per_batch=int(max_tokens_per_batch) if max_tokens_per_batch is not None else None,
    )

    atom = Atom(llm_model=llm, embeddings_model=embeddings, llm_output_parser=parser)
    return atom, parser


def init_storage(config: Dict[str, Any]) -> Neo4jStorage:
    neo4j_cfg = config.get("neo4j", {}) or {}
    uri = resolve_value(neo4j_cfg, "uri", required=True)
    username = resolve_value(neo4j_cfg, "username", required=True)
    password = resolve_value(neo4j_cfg, "password", required=True)
    database = resolve_value(neo4j_cfg, "database") or None
    if database == "neo4j":
        database = None
    return Neo4jStorage(uri=uri, username=username, password=password, database=database)


def graph_exists(storage: Neo4jStorage, graph_name: str) -> bool:
    escaped = Neo4jStorage.format_value(graph_name)
    query = f'MATCH (n {{graph_name: "{escaped}"}}) RETURN count(n) AS c'
    records = storage.run_query_with_result(query)
    if not records:
        return False
    return int(records[0].get("c", 0) or 0) > 0


def load_graph_by_name(storage: Neo4jStorage, graph_name: str) -> KnowledgeGraph:
    escaped = Neo4jStorage.format_value(graph_name)
    node_query = f'MATCH (n {{graph_name: "{escaped}"}}) RETURN n'
    nodes = storage.run_query_with_result(node_query)
    if not nodes:
        return KnowledgeGraph()

    entities: List[Entity] = []
    entity_map: Dict[str, Entity] = {}
    for record in nodes:
        node = record["n"]
        props = dict(node.items())
        embeddings = None
        if "embeddings" in props and props["embeddings"]:
            embeddings = storage.transform_str_list_to_embeddings(props.pop("embeddings"))
        props.pop("graph_name", None)
        props.pop("doc_id", None)

        label = list(node.labels)[0] if node.labels else "entity"
        entity = Entity(
            name=props.get("name", ""),
            label=label,
            properties=EntityProperties(embeddings=embeddings),
        )
        entities.append(entity)
        entity_map[node.element_id] = entity

    rel_query = (
        f'MATCH (n {{graph_name: "{escaped}"}})-[r {{graph_name: "{escaped}"}}]->'
        f'(m {{graph_name: "{escaped}"}}) RETURN n, r, m'
    )
    rel_records = storage.run_query_with_result(rel_query)
    relationships: List[Relationship] = []
    for record in rel_records:
        start_node = record["n"]
        rel = record["r"]
        end_node = record["m"]
        start_entity = entity_map.get(start_node.element_id)
        end_entity = entity_map.get(end_node.element_id)
        if not start_entity or not end_entity:
            continue

        rel_props_dict = dict(rel.items())
        embeddings = None
        if "embeddings" in rel_props_dict and rel_props_dict["embeddings"]:
            embeddings = storage.transform_str_list_to_embeddings(rel_props_dict.pop("embeddings"))
        atomic_facts = rel_props_dict.pop("atomic_facts", [])
        t_obs = rel_props_dict.pop("t_obs", [])
        t_start = rel_props_dict.pop("t_start", [])
        t_end = rel_props_dict.pop("t_end", [])
        rel_props_dict.pop("graph_name", None)
        rel_props_dict.pop("doc_id", None)
        predicate = rel_props_dict.pop("predicate", None)

        rel_props = RelationshipProperties(
            embeddings=embeddings,
            atomic_facts=atomic_facts if isinstance(atomic_facts, list) else [],
            t_obs=t_obs if isinstance(t_obs, list) else [],
            t_start=t_start if isinstance(t_start, list) else [],
            t_end=t_end if isinstance(t_end, list) else [],
        )

        relationships.append(
            Relationship(
                name=predicate or rel.type,
                startEntity=start_entity,
                endEntity=end_entity,
                properties=rel_props,
            )
        )

    return KnowledgeGraph(entities=entities, relationships=relationships)


def _compute_touched_keys(
    kg: KnowledgeGraph, current_facts: set[str]
) -> Tuple[set[Tuple[str, str]], set[Tuple[str, str, str, str, str]]]:
    touched_entities: set[Tuple[str, str]] = set()
    touched_relationships: set[Tuple[str, str, str, str, str]] = set()
    for rel in kg.relationships:
        rel_facts = getattr(rel.properties, "atomic_facts", None) or []
        if not any(fact in current_facts for fact in rel_facts):
            continue

        start_label = Neo4jStorage.sanitize_label(rel.startEntity.label)
        end_label = Neo4jStorage.sanitize_label(rel.endEntity.label)
        start_name = rel.startEntity.name
        end_name = rel.endEntity.name
        predicate = rel.name

        touched_entities.add((start_label, start_name))
        touched_entities.add((end_label, end_name))
        touched_relationships.add((start_label, start_name, end_label, end_name, predicate))
    return touched_entities, touched_relationships


def persist_graph_with_doc_id(
    storage: Neo4jStorage,
    kg: KnowledgeGraph,
    graph_name: str,
    doc_id: str,
    current_atomic_facts: set[str],
    relationship_type: str,
) -> None:
    escaped_graph_name = Neo4jStorage.format_value(graph_name)
    escaped_doc_id = Neo4jStorage.format_value(doc_id)

    relationship_type = relationship_type or "REL"
    rel_type = Neo4jStorage.sanitize_relationship_type(relationship_type)
    if any(not (c.isascii() and (c.isalnum() or c == "_")) for c in rel_type):
        logging.warning("neo4j_relationship_type=%r 不是 ASCII 标识符，已回退为 'REL'", relationship_type)
        rel_type = "REL"

    touched_entities, touched_relationships = _compute_touched_keys(kg, current_atomic_facts)

    for entity in kg.entities:
        label = Neo4jStorage.sanitize_label(entity.label)
        name = Neo4jStorage.format_value(entity.name)

        props = entity.properties.model_dump()
        statements: List[str] = []
        for key, value in props.items():
            if value is None:
                continue
            formatted = Neo4jStorage.format_property_value(key, value)
            statements.append(f'n.{key.replace(" ", "_")} = {formatted}')

        if (label, entity.name) in touched_entities:
            statements.append(
                'n.doc_id = CASE '
                f'WHEN "{escaped_doc_id}" IN COALESCE(n.doc_id, []) THEN COALESCE(n.doc_id, []) '
                f'ELSE COALESCE(n.doc_id, []) + ["{escaped_doc_id}"] END'
            )

        set_clause = " SET " + ", ".join(statements) if statements else ""
        query = f'MERGE (n:{label} {{name: "{name}", graph_name: "{escaped_graph_name}"}}){set_clause}'
        storage.run_query(query)

    for rel in kg.relationships:
        start_label = Neo4jStorage.sanitize_label(rel.startEntity.label)
        end_label = Neo4jStorage.sanitize_label(rel.endEntity.label)
        start_name = Neo4jStorage.format_value(rel.startEntity.name)
        end_name = Neo4jStorage.format_value(rel.endEntity.name)
        predicate_value = Neo4jStorage.format_value(rel.name)

        props = rel.properties.model_dump()
        statements = []
        for key, value in props.items():
            if value is None:
                continue
            formatted = Neo4jStorage.format_property_value(key, value)
            statements.append(f'r.{key.replace(" ", "_")} = {formatted}')

        if (start_label, rel.startEntity.name, end_label, rel.endEntity.name, rel.name) in touched_relationships:
            statements.append(
                'r.doc_id = CASE '
                f'WHEN "{escaped_doc_id}" IN COALESCE(r.doc_id, []) THEN COALESCE(r.doc_id, []) '
                f'ELSE COALESCE(r.doc_id, []) + ["{escaped_doc_id}"] END'
            )

        set_clause = " SET " + ", ".join(statements) if statements else ""
        query = (
            f'MATCH (n:{start_label} {{name: "{start_name}", graph_name: "{escaped_graph_name}"}}), '
            f'(m:{end_label} {{name: "{end_name}", graph_name: "{escaped_graph_name}"}}) '
            f'MERGE (n)-[r:{rel_type} {{predicate: "{predicate_value}", graph_name: "{escaped_graph_name}"}}]->(m)'
            f"{set_clause}"
        )
        storage.run_query(query)


def fetch_doc_id_index(
    storage: Neo4jStorage, graph_name: str
) -> Tuple[Dict[Tuple[str, str], List[str]], Dict[Tuple[str, str, str, str, str], List[str]]]:
    escaped = Neo4jStorage.format_value(graph_name)
    node_query = f'MATCH (n {{graph_name: "{escaped}"}}) RETURN labels(n) AS labels, n.name AS name, n.doc_id AS doc_id'
    node_records = storage.run_query_with_result(node_query)
    node_index: Dict[Tuple[str, str], List[str]] = {}
    for record in node_records:
        name = record.get("name")
        labels = record.get("labels") or []
        doc_ids = record.get("doc_id") or []
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(doc_ids, list):
            doc_ids = []
        for label in labels:
            if not isinstance(label, str) or not label:
                continue
            node_index[(Neo4jStorage.sanitize_label(label), name)] = [str(x) for x in doc_ids if str(x).strip()]

    rel_query = (
        f'MATCH (n {{graph_name: "{escaped}"}})-[r {{graph_name: "{escaped}"}}]->(m {{graph_name: "{escaped}"}}) '
        "RETURN labels(n) AS start_labels, n.name AS start_name, labels(m) AS end_labels, m.name AS end_name, "
        "r.predicate AS predicate, r.doc_id AS doc_id"
    )
    rel_records = storage.run_query_with_result(rel_query)
    rel_index: Dict[Tuple[str, str, str, str, str], List[str]] = {}
    for record in rel_records:
        start_name = record.get("start_name")
        end_name = record.get("end_name")
        predicate = record.get("predicate")
        start_labels = record.get("start_labels") or []
        end_labels = record.get("end_labels") or []
        doc_ids = record.get("doc_id") or []
        if not (isinstance(start_name, str) and start_name and isinstance(end_name, str) and end_name):
            continue
        if not isinstance(predicate, str) or not predicate:
            continue
        if not isinstance(doc_ids, list):
            doc_ids = []
        cleaned_doc_ids = [str(x) for x in doc_ids if str(x).strip()]
        for sl in start_labels:
            if not isinstance(sl, str) or not sl:
                continue
            for el in end_labels:
                if not isinstance(el, str) or not el:
                    continue
                key = (Neo4jStorage.sanitize_label(sl), start_name, Neo4jStorage.sanitize_label(el), end_name, predicate)
                rel_index[key] = cleaned_doc_ids

    return node_index, rel_index


def persist_graph_to_new_graph_with_doc_id(
    storage: Neo4jStorage,
    kg: KnowledgeGraph,
    graph_name: str,
    doc_id: str,
    current_atomic_facts: set[str],
    base_node_doc_ids: Dict[Tuple[str, str], List[str]],
    base_rel_doc_ids: Dict[Tuple[str, str, str, str, str], List[str]],
    relationship_type: str,
) -> None:
    escaped_graph_name = Neo4jStorage.format_value(graph_name)
    relationship_type = relationship_type or "REL"
    rel_type = Neo4jStorage.sanitize_relationship_type(relationship_type)
    if any(not (c.isascii() and (c.isalnum() or c == "_")) for c in rel_type):
        logging.warning("neo4j_relationship_type=%r 不是 ASCII 标识符，已回退为 'REL'", relationship_type)
        rel_type = "REL"

    touched_entities, touched_relationships = _compute_touched_keys(kg, current_atomic_facts)

    for entity in kg.entities:
        label = Neo4jStorage.sanitize_label(entity.label)
        name_raw = entity.name
        name = Neo4jStorage.format_value(name_raw)

        doc_ids = list(base_node_doc_ids.get((label, name_raw), []))
        if (label, name_raw) in touched_entities and doc_id not in doc_ids:
            doc_ids.append(doc_id)
        if not doc_ids and (label, name_raw) in touched_entities:
            doc_ids = [doc_id]

        props = entity.properties.model_dump()
        statements: List[str] = []
        for key, value in props.items():
            if value is None:
                continue
            formatted = Neo4jStorage.format_property_value(key, value)
            statements.append(f'n.{key.replace(" ", "_")} = {formatted}')
        statements.append(f"n.doc_id = {Neo4jStorage.format_property_value('doc_id', doc_ids)}")

        set_clause = " SET " + ", ".join(statements) if statements else ""
        query = f'MERGE (n:{label} {{name: "{name}", graph_name: "{escaped_graph_name}"}}){set_clause}'
        storage.run_query(query)

    for rel in kg.relationships:
        start_label = Neo4jStorage.sanitize_label(rel.startEntity.label)
        end_label = Neo4jStorage.sanitize_label(rel.endEntity.label)
        start_name_raw = rel.startEntity.name
        end_name_raw = rel.endEntity.name
        start_name = Neo4jStorage.format_value(start_name_raw)
        end_name = Neo4jStorage.format_value(end_name_raw)
        predicate_raw = rel.name
        predicate_value = Neo4jStorage.format_value(predicate_raw)

        key = (start_label, start_name_raw, end_label, end_name_raw, predicate_raw)
        doc_ids = list(base_rel_doc_ids.get(key, []))
        if key in touched_relationships and doc_id not in doc_ids:
            doc_ids.append(doc_id)
        if not doc_ids and key in touched_relationships:
            doc_ids = [doc_id]

        props = rel.properties.model_dump()
        statements = []
        for k, v in props.items():
            if v is None:
                continue
            formatted = Neo4jStorage.format_property_value(k, v)
            statements.append(f'r.{k.replace(" ", "_")} = {formatted}')
        statements.append(f"r.doc_id = {Neo4jStorage.format_property_value('doc_id', doc_ids)}")

        set_clause = " SET " + ", ".join(statements) if statements else ""
        query = (
            f'MATCH (n:{start_label} {{name: "{start_name}", graph_name: "{escaped_graph_name}"}}), '
            f'(m:{end_label} {{name: "{end_name}", graph_name: "{escaped_graph_name}"}}) '
            f'MERGE (n)-[r:{rel_type} {{predicate: "{predicate_value}", graph_name: "{escaped_graph_name}"}}]->(m)'
            f"{set_clause}"
        )
        storage.run_query(query)


def list_graphs_with_stats(storage: Neo4jStorage) -> List[Dict[str, Any]]:
    query = """
    MATCH (n)
    WHERE n.graph_name IS NOT NULL
    WITH DISTINCT n.graph_name AS graph_name
    CALL {
      WITH graph_name
      MATCH (n {graph_name: graph_name})
      RETURN count(DISTINCT n) AS node_count
    }
    CALL {
      WITH graph_name
      MATCH ()-[r {graph_name: graph_name}]-()
      RETURN count(DISTINCT r) AS relationship_count
    }
    RETURN graph_name, node_count, relationship_count
    ORDER BY graph_name
    """
    records = storage.run_query_with_result(query)
    result: List[Dict[str, Any]] = []
    for record in records:
        result.append(
            {
                "graph_name": record.get("graph_name"),
                "node_count": int(record.get("node_count", 0) or 0),
                "relationship_count": int(record.get("relationship_count", 0) or 0),
            }
        )
    return result


def get_graph_stats(storage: Neo4jStorage, graph_name: str) -> Dict[str, int]:
    escaped = Neo4jStorage.format_value(graph_name)
    node_query = f'MATCH (n {{graph_name: "{escaped}"}}) RETURN count(DISTINCT n) AS c'
    rel_query = f'MATCH ()-[r {{graph_name: "{escaped}"}}]-() RETURN count(DISTINCT r) AS c'
    node_records = storage.run_query_with_result(node_query)
    rel_records = storage.run_query_with_result(rel_query)
    node_count = int(node_records[0].get("c", 0) or 0) if node_records else 0
    relationship_count = int(rel_records[0].get("c", 0) or 0) if rel_records else 0
    return {"node_count": node_count, "relationship_count": relationship_count}


def delete_graph(storage: Neo4jStorage, graph_name: str) -> Tuple[int, int]:
    stats = get_graph_stats(storage, graph_name)
    escaped = Neo4jStorage.format_value(graph_name)
    storage.run_query(f'MATCH (n {{graph_name: "{escaped}"}}) DETACH DELETE n')
    return stats["node_count"], stats["relationship_count"]


def rename_graph(storage: Neo4jStorage, old_name: str, new_name: str) -> Tuple[int, int]:
    old_escaped = Neo4jStorage.format_value(old_name)
    new_escaped = Neo4jStorage.format_value(new_name)

    if graph_exists(storage, new_name):
        raise ValueError(f"新图谱名称已存在: {new_name}")

    stats = get_graph_stats(storage, old_name)

    storage.run_query(
        f'MATCH (n {{graph_name: "{old_escaped}"}}) SET n.graph_name = "{new_escaped}"'
    )
    storage.run_query(
        f'MATCH ()-[r {{graph_name: "{old_escaped}"}}]-() SET r.graph_name = "{new_escaped}"'
    )
    return stats["node_count"], stats["relationship_count"]


@dataclass(frozen=True)
class GraphView:
    entities: set[str]
    relations: set[Tuple[str, str, str]]
    entity_clusters: Dict[str, List[str]] = None  # type: ignore[assignment]
    edge_clusters: Dict[str, List[str]] = None  # type: ignore[assignment]

    def __post_init__(self):
        object.__setattr__(self, "entity_clusters", self.entity_clusters or {})
        object.__setattr__(self, "edge_clusters", self.edge_clusters or {})


def _string_to_color(label: str) -> str:
    digest = hashlib.sha1(label.encode("utf-8")).hexdigest()
    hue = int(digest[:2], 16) / 255.0
    saturation = 0.55 + (int(digest[2:4], 16) / 255.0) * 0.3
    lightness = 0.45 + (int(digest[4:6], 16) / 255.0) * 0.25
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _sorted_ignore_case(items) -> List[str]:
    return sorted(items, key=lambda value: str(value).lower())


def _build_view_model(graph: GraphView) -> Dict[str, Any]:
    all_entities = set(graph.entities)
    for subject, _, obj in graph.relations:
        all_entities.add(subject)
        all_entities.add(obj)
    entities = _sorted_ignore_case(all_entities)

    relations = sorted(
        graph.relations,
        key=lambda triple: (triple[1].lower(), triple[0].lower(), triple[2].lower()),
    )

    entity_clusters = graph.entity_clusters or {}
    edge_clusters = graph.edge_clusters or {}

    entity_member_to_cluster: Dict[str, str] = {}
    cluster_view: List[Dict[str, Any]] = []

    for representative, members in entity_clusters.items():
        full_members = set(members)
        full_members.add(representative)
        ordered_members = _sorted_ignore_case(full_members)
        color = _string_to_color(f"entity::{representative}")
        cluster_view.append(
            {
                "id": representative,
                "label": representative,
                "members": ordered_members,
                "size": len(ordered_members),
                "color": color,
            }
        )
        for member in ordered_members:
            entity_member_to_cluster[member] = representative

    node_color_lookup: Dict[str, str] = {}
    if cluster_view:
        for cluster in cluster_view:
            for member in cluster["members"]:
                node_color_lookup[member] = cluster["color"]
    else:
        for entity in entities:
            node_color_lookup[entity] = _string_to_color(f"entity::{entity}")

    edge_member_to_cluster: Dict[str, str] = {}
    edge_color_lookup: Dict[str, str] = {}
    edge_cluster_view: List[Dict[str, Any]] = []

    for representative, members in edge_clusters.items():
        full_members = set(members)
        full_members.add(representative)
        ordered_members = _sorted_ignore_case(full_members)
        color = _string_to_color(f"edge::{representative}")
        edge_cluster_view.append(
            {
                "id": representative,
                "label": representative,
                "members": ordered_members,
                "size": len(ordered_members),
                "color": color,
            }
        )
        for member in ordered_members:
            edge_member_to_cluster[member] = representative
            edge_color_lookup[member] = color

    degree = Counter()
    indegree = Counter()
    outdegree = Counter()
    predicate_counts = Counter()

    adjacency: Dict[str, set[str]] = defaultdict(set)
    node_neighbors: Dict[str, set[str]] = defaultdict(set)
    node_edges: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: {"incoming": [], "outgoing": []})

    edges_view: List[Dict[str, Any]] = []
    for index, (subject, predicate, obj) in enumerate(relations):
        predicate_counts[predicate] += 1
        degree[subject] += 1
        degree[obj] += 1
        outdegree[subject] += 1
        indegree[obj] += 1
        adjacency[subject].add(obj)
        adjacency[obj].add(subject)
        node_neighbors[subject].add(obj)
        node_neighbors[obj].add(subject)

        edge_id = f"e{index}"
        color = edge_color_lookup.get(predicate)
        if not color:
            color = _string_to_color(f"predicate::{predicate}")
            edge_color_lookup[predicate] = color

        edges_view.append(
            {
                "id": edge_id,
                "source": subject,
                "target": obj,
                "predicate": predicate,
                "cluster": edge_member_to_cluster.get(predicate),
                "color": color,
                "tooltip": f"{subject} —{predicate}→ {obj}",
            }
        )

        node_edges[subject]["outgoing"].append(edge_id)
        node_edges[obj]["incoming"].append(edge_id)

    def connected_components() -> List[Dict[str, Any]]:
        visited: set[str] = set()
        components: List[Dict[str, Any]] = []
        for node in entities:
            if node in visited:
                continue
            queue: deque[str] = deque([node])
            visited.add(node)
            members: List[str] = []
            while queue:
                current = queue.popleft()
                members.append(current)
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append({"size": len(members), "members": _sorted_ignore_case(members)})
        components.sort(key=lambda comp: (-comp["size"], comp["members"][0]))
        return components

    components = connected_components()
    isolated_entities = [entity for entity in entities if degree[entity] == 0]

    nodes_view: List[Dict[str, Any]] = []
    for entity in entities:
        cluster_id = entity_member_to_cluster.get(entity)
        radius = 18 + min(degree[entity], 8) * 2
        nodes_view.append(
            {
                "id": entity,
                "label": entity,
                "cluster": cluster_id,
                "color": node_color_lookup.get(entity, "#64748b"),
                "degree": degree[entity],
                "indegree": indegree[entity],
                "outdegree": outdegree[entity],
                "isRepresentative": cluster_id == entity if cluster_id else False,
                "radius": radius,
                "neighbors": _sorted_ignore_case(node_neighbors.get(entity, set())),
                "edgeIds": node_edges.get(entity, {"incoming": [], "outgoing": []}),
            }
        )

    top_entities = sorted(
        (
            {
                "label": node["label"],
                "degree": node["degree"],
                "indegree": node["indegree"],
                "outdegree": node["outdegree"],
                "cluster": node["cluster"],
            }
            for node in nodes_view
        ),
        key=lambda item: (-item["degree"], item["label"].lower()),
    )[:10]

    top_relations = sorted(
        (
            {
                "predicate": predicate,
                "count": count,
                "cluster": edge_member_to_cluster.get(predicate),
                "color": edge_color_lookup.get(predicate, "#64748b"),
            }
            for predicate, count in predicate_counts.items()
        ),
        key=lambda item: (-item["count"], item["predicate"].lower()),
    )[:10]

    stats = {
        "entities": len(entities),
        "relations": len(edges_view),
        "relationTypes": len(predicate_counts),
        "entityClusters": len(cluster_view),
        "edgeClusters": len(edge_cluster_view),
        "isolatedEntities": len(isolated_entities),
        "components": len(components),
        "averageDegree": round(sum(degree[entity] for entity in entities) / len(entities), 2) if entities else 0,
        "density": round(len(edges_view) / (len(entities) * (len(entities) - 1)), 3) if len(entities) > 1 else 0,
    }

    relation_records = [
        {
            "source": subject,
            "predicate": predicate,
            "target": obj,
            "edgeId": edge["id"],
            "color": edge["color"],
        }
        for edge, (subject, predicate, obj) in zip(edges_view, relations)
    ]

    return {
        "nodes": nodes_view,
        "edges": edges_view,
        "clusters": cluster_view,
        "edgeClusters": edge_cluster_view,
        "topEntities": top_entities,
        "topRelations": top_relations,
        "stats": stats,
        "isolatedEntities": isolated_entities,
        "components": components,
        "relations": relation_records,
    }


def _parse_csv_param(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    value = str(value).strip()
    if not value:
        return []
    parts = [part.strip() for part in value.split(",")]
    return [part for part in parts if part]


def _cypher_str_list(values: List[str]) -> str:
    escaped = [Neo4jStorage.format_value(item) for item in values if str(item).strip()]
    return "[" + ", ".join(f'"{item}"' for item in escaped) + "]"


@dataclass(frozen=True)
class GraphViewFilters:
    doc_ids: List[str]
    entities: List[str]
    predicates: List[str]
    q: Optional[str]
    center: Optional[str]
    depth: int
    limit_nodes: int
    limit_edges: int


def _load_graph_view_filtered(storage: Neo4jStorage, graph_name: str, filters: GraphViewFilters) -> GraphView:
    escaped_graph_name = Neo4jStorage.format_value(graph_name)

    doc_id_clause_node = ""
    doc_id_clause_rel = ""
    doc_id_clause_path_rel = ""
    if filters.doc_ids:
        doc_list = _cypher_str_list(filters.doc_ids)
        doc_id_clause_node = f" AND ANY(d IN COALESCE(n.doc_id, []) WHERE d IN {doc_list})"
        doc_id_clause_rel = f" AND ANY(d IN COALESCE(r.doc_id, []) WHERE d IN {doc_list})"
        doc_id_clause_path_rel = f" AND ANY(d IN COALESCE(rel.doc_id, []) WHERE d IN {doc_list})"

    entities_clause_node = ""
    entities_clause_rel = ""
    if filters.entities:
        entity_list = _cypher_str_list(filters.entities)
        entities_clause_node = f" AND n.name IN {entity_list}"
        entities_clause_rel = f" AND s.name IN {entity_list} AND o.name IN {entity_list}"

    predicates_clause_rel = ""
    predicates_clause_path_rel = ""
    if filters.predicates:
        predicate_list = _cypher_str_list(filters.predicates)
        predicates_clause_rel = f" AND r.predicate IN {predicate_list}"
        predicates_clause_path_rel = f" AND rel.predicate IN {predicate_list}"

    q_clause_node = ""
    q_clause_rel = ""
    if filters.q:
        escaped_q = Neo4jStorage.format_value(filters.q)
        q_clause_node = f' AND toLower(n.name) CONTAINS toLower("{escaped_q}")'
        q_clause_rel = (
            f' AND (toLower(s.name) CONTAINS toLower("{escaped_q}") '
            f'OR toLower(o.name) CONTAINS toLower("{escaped_q}") '
            f'OR toLower(r.predicate) CONTAINS toLower("{escaped_q}"))'
        )

    relations: set[Tuple[str, str, str]] = set()
    entities: set[str] = set()

    def _add_relation(subject: Any, predicate: Any, obj: Any) -> None:
        if not (subject and predicate and obj):
            return
        subject_s = str(subject)
        predicate_s = str(predicate)
        obj_s = str(obj)
        if not (subject_s and predicate_s and obj_s):
            return
        relations.add((subject_s, predicate_s, obj_s))
        entities.add(subject_s)
        entities.add(obj_s)

    if filters.center:
        center_escaped = Neo4jStorage.format_value(filters.center)
        depth = max(0, int(filters.depth))
        if depth > 0:
            rel_query = (
                f'MATCH (c {{graph_name: "{escaped_graph_name}", name: "{center_escaped}"}}) '
                f'MATCH p=(c)-[*1..{depth}]-(n) '
                f'WHERE ALL(node IN nodes(p) WHERE node.graph_name = "{escaped_graph_name}") '
                f'AND ALL(rel IN relationships(p) WHERE rel.graph_name = "{escaped_graph_name}"'
                f"{doc_id_clause_path_rel}{predicates_clause_path_rel}"
                ") "
                "WITH DISTINCT relationships(p) AS rels "
                "UNWIND rels AS r "
                "WITH DISTINCT r "
                "MATCH (s)-[r]->(o) "
                f'WHERE s.graph_name = "{escaped_graph_name}" AND o.graph_name = "{escaped_graph_name}"'
                f"{entities_clause_rel}{q_clause_rel}"
                " RETURN DISTINCT s.name AS subject, r.predicate AS predicate, o.name AS object "
                f"LIMIT {int(filters.limit_edges)}"
            )
            for record in storage.run_query_with_result(rel_query):
                _add_relation(record.get("subject"), record.get("predicate"), record.get("object"))

        node_query = (
            f'MATCH (c {{graph_name: "{escaped_graph_name}", name: "{center_escaped}"}}) '
            f'MATCH p=(c)-[*0..{depth}]-(n) '
            f'WHERE ALL(node IN nodes(p) WHERE node.graph_name = "{escaped_graph_name}") '
            f'AND ALL(rel IN relationships(p) WHERE rel.graph_name = "{escaped_graph_name}"'
            f"{doc_id_clause_path_rel}{predicates_clause_path_rel}"
            ") "
            "WITH DISTINCT n "
            f'WHERE n.graph_name = "{escaped_graph_name}"'
            f"{doc_id_clause_node}{entities_clause_node}{q_clause_node}"
            " RETURN DISTINCT n.name AS name "
            f"LIMIT {int(filters.limit_nodes)}"
        )
        for record in storage.run_query_with_result(node_query):
            name = record.get("name")
            if name:
                entities.add(str(name))
        entities.add(filters.center)
    else:
        rel_query = (
            f'MATCH (s {{graph_name: "{escaped_graph_name}"}})-[r {{graph_name: "{escaped_graph_name}"}}]->'
            f'(o {{graph_name: "{escaped_graph_name}"}}) '
            f'WHERE 1=1{doc_id_clause_rel}{predicates_clause_rel}{entities_clause_rel}{q_clause_rel} '
            "RETURN DISTINCT s.name AS subject, r.predicate AS predicate, o.name AS object "
            f"LIMIT {int(filters.limit_edges)}"
        )
        for record in storage.run_query_with_result(rel_query):
            _add_relation(record.get("subject"), record.get("predicate"), record.get("object"))

        node_query = (
            f'MATCH (n {{graph_name: "{escaped_graph_name}"}}) '
            f'WHERE 1=1{doc_id_clause_node}{entities_clause_node}{q_clause_node} '
            "RETURN DISTINCT n.name AS name "
            f"LIMIT {int(filters.limit_nodes)}"
        )
        for record in storage.run_query_with_result(node_query):
            name = record.get("name")
            if name:
                entities.add(str(name))

    return GraphView(entities=set(str(e) for e in entities if e), relations=relations)


def _load_graph_view(storage: Neo4jStorage, graph_name: str) -> GraphView:
    escaped = Neo4jStorage.format_value(graph_name)
    node_query = f'MATCH (n {{graph_name: "{escaped}"}}) RETURN DISTINCT n.name AS name'
    node_records = storage.run_query_with_result(node_query)
    entities = {record.get("name") for record in node_records if record.get("name")}

    rel_query = (
        f'MATCH (s {{graph_name: "{escaped}"}})-[r {{graph_name: "{escaped}"}}]->(o {{graph_name: "{escaped}"}}) '
        "RETURN DISTINCT s.name AS subject, r.predicate AS predicate, o.name AS object"
    )
    rel_records = storage.run_query_with_result(rel_query)
    relations: set[Tuple[str, str, str]] = set()
    for record in rel_records:
        subject = record.get("subject")
        predicate = record.get("predicate")
        obj = record.get("object")
        if not (subject and predicate and obj):
            continue
        relations.add((str(subject), str(predicate), str(obj)))
        entities.add(str(subject))
        entities.add(str(obj))

    return GraphView(entities=set(str(e) for e in entities if e), relations=relations)


@dataclass
class ServerResources:
    config_path: Path
    config_mtime_ns: int
    config: Dict[str, Any]
    storage: Neo4jStorage
    atom: Atom
    parser: LangchainOutputParser


_resources: Optional[ServerResources] = None
_resources_lock = asyncio.Lock()


def _get_visualize_view_model(
    resources: ServerResources,
    graph_name: str,
    doc_id: Optional[str],
    entities: Optional[str],
    predicates: Optional[str],
    q: Optional[str],
    center: Optional[str],
    depth: int,
    limit_nodes: Optional[int],
    limit_edges: Optional[int],
) -> Tuple[Dict[str, Any], bool]:
    server_cfg = resources.config.get("server", {}) or {}
    viz_cfg = server_cfg.get("visualize", {}) or {}
    max_depth = int(viz_cfg.get("max_depth", 6))
    max_limit_nodes = int(viz_cfg.get("max_limit_nodes", 6000))
    max_limit_edges = int(viz_cfg.get("max_limit_edges", 12000))
    default_limit_nodes = int(viz_cfg.get("default_limit_nodes", min(2000, max_limit_nodes)))
    default_limit_edges = int(viz_cfg.get("default_limit_edges", min(4000, max_limit_edges)))

    if depth > max_depth:
        raise HTTPException(status_code=400, detail=f"depth 超出上限：{depth} > {max_depth}")

    limit_nodes_final = default_limit_nodes if limit_nodes is None else int(limit_nodes)
    limit_edges_final = default_limit_edges if limit_edges is None else int(limit_edges)
    limit_nodes_final = min(limit_nodes_final, max_limit_nodes)
    limit_edges_final = min(limit_edges_final, max_limit_edges)

    doc_ids_list = _parse_csv_param(doc_id)
    entities_list = _parse_csv_param(entities)
    predicates_list = _parse_csv_param(predicates)
    q_value = str(q).strip() if q is not None and str(q).strip() else None
    center_value = str(center).strip() if center is not None and str(center).strip() else None

    has_filters = any([doc_ids_list, entities_list, predicates_list, q_value, center_value]) or (
        limit_nodes is not None or limit_edges is not None
    )
    if has_filters:
        graph = _load_graph_view_filtered(
            resources.storage,
            graph_name,
            GraphViewFilters(
                doc_ids=doc_ids_list,
                entities=entities_list,
                predicates=predicates_list,
                q=q_value,
                center=center_value,
                depth=depth,
                limit_nodes=limit_nodes_final,
                limit_edges=limit_edges_final,
            ),
        )
    else:
        graph = _load_graph_view(resources.storage, graph_name)

    return _build_view_model(graph), has_filters


def _get_config_path() -> Path:
    env_path = os.getenv("KG_GEN_SERVER_CONFIG")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return (BASE_DIR / "config.yaml").resolve()


async def get_resources() -> ServerResources:
    global _resources

    config_path = _get_config_path()
    try:
        mtime_ns = config_path.stat().st_mtime_ns
    except FileNotFoundError as e:
        raise RuntimeError(f"配置文件不存在: {config_path}") from e

    async with _resources_lock:
        if _resources is not None and _resources.config_path == config_path and _resources.config_mtime_ns == mtime_ns:
            return _resources

        config = load_config(config_path)
        setup_logging(config)
        storage = init_storage(config)
        atom, parser = init_atom_and_parser(config)

        _resources = ServerResources(
            config_path=config_path,
            config_mtime_ns=mtime_ns,
            config=config,
            storage=storage,
            atom=atom,
            parser=parser,
        )
        return _resources


server_cfg = load_config(_get_config_path()).get("server", {}) if _get_config_path().exists() else {}
app = FastAPI(
    title=str(server_cfg.get("title", "kg-gen-server")),
    version=str(server_cfg.get("version", "0.1.0")),
)

allow_origins = server_cfg.get("cors_allow_origins", ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.on_event("shutdown")
async def _shutdown() -> None:
    global _resources
    if _resources is None:
        return
    try:
        _resources.storage.driver.close()
    except Exception:
        pass
    _resources = None


@app.get("/", response_class=HTMLResponse)
@app.get("/test", response_class=HTMLResponse)
async def test_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("build_test.html", {"request": request})


@app.post("/api/graph/build", response_model=BuildGraphResponse)
async def api_build_graph(payload: BuildGraphRequest) -> BuildGraphResponse:
    resources = await get_resources()

    if payload.new_graph_name and payload.new_graph_name.strip() == payload.graph_name.strip():
        raise HTTPException(status_code=400, detail="new_graph_name 不能与 graph_name 相同")

    if payload.new_graph_name:
        if graph_exists(resources.storage, payload.new_graph_name):
            raise HTTPException(status_code=400, detail=f"new_graph_name 已存在: {payload.new_graph_name}")
        if not graph_exists(resources.storage, payload.graph_name):
            raise HTTPException(status_code=404, detail=f"graph_name 不存在，无法合并生成新图谱: {payload.graph_name}")

    text_cfg = resources.config.get("text", {}) or {}
    chunker_cfg = (text_cfg.get("chunker") or {}) if isinstance(text_cfg, dict) else {}
    if str(chunker_cfg.get("type", "")).strip() != "chonkie_fastapi":
        raise HTTPException(status_code=400, detail="仅支持 text.chunker.type=chonkie_fastapi")

    obs_timestamp = datetime.now(UTC).isoformat()

    try:
        sections = await asyncio.to_thread(chunk_text_with_chonkie_fastapi, payload.doc_content, chunker_cfg)
        output_cfg = resources.config.get("output", {}) or {}
        output_language = str(output_cfg.get("language", "en"))
        entity_name_mode = str(output_cfg.get("entity_name_mode", "normalized"))

        atomic_facts = await extract_atomic_facts(
            resources.parser,
            sections,
            obs_timestamp,
            output_language=output_language,
            entity_name_mode=entity_name_mode,
        )
        if not atomic_facts:
            raise HTTPException(status_code=400, detail="未能抽取到原子事实，无法继续构图")

        atom_cfg = resources.config.get("atom", {}) or {}
        ent_threshold = float(atom_cfg.get("ent_threshold", 0.8))
        rel_threshold = float(atom_cfg.get("rel_threshold", 0.7))
        entity_name_weight = float(atom_cfg.get("entity_name_weight", 0.8))
        entity_label_weight = float(atom_cfg.get("entity_label_weight", 0.2))
        max_workers = int(atom_cfg.get("max_workers", 8))
        matching_cfg = atom_cfg.get("matching", {}) or {}
        require_same_entity_label = bool(matching_cfg.get("require_same_entity_label", entity_name_mode == "source"))
        relation_name_mode = str(output_cfg.get("relation_name_mode", "en_snake"))
        rename_relationship_by_embedding = bool(
            matching_cfg.get("rename_relationship_by_embedding", relation_name_mode != "source")
        )

        existing_kg = KnowledgeGraph()
        base_node_doc_ids: Dict[Tuple[str, str], List[str]] = {}
        base_rel_doc_ids: Dict[Tuple[str, str, str, str, str], List[str]] = {}
        if graph_exists(resources.storage, payload.graph_name):
            existing_kg = load_graph_by_name(resources.storage, payload.graph_name)
            if payload.new_graph_name:
                base_node_doc_ids, base_rel_doc_ids = fetch_doc_id_index(resources.storage, payload.graph_name)

        knowledge_graph = await resources.atom.build_graph(
            atomic_facts=atomic_facts,
            obs_timestamp=obs_timestamp,
            existing_knowledge_graph=None if existing_kg.is_empty() else existing_kg,
            output_language=output_language,
            entity_name_mode=entity_name_mode,
            relation_name_mode=relation_name_mode,
            require_same_entity_label=require_same_entity_label,
            rename_relationship_by_embedding=rename_relationship_by_embedding,
            ent_threshold=ent_threshold,
            rel_threshold=rel_threshold,
            entity_name_weight=entity_name_weight,
            entity_label_weight=entity_label_weight,
            max_workers=max_workers,
        )

        target_graph_name = payload.new_graph_name or payload.graph_name
        neo4j_relationship_type = str(output_cfg.get("neo4j_relationship_type", "REL"))
        if payload.new_graph_name:
            persist_graph_to_new_graph_with_doc_id(
                resources.storage,
                knowledge_graph,
                target_graph_name,
                payload.doc_id,
                current_atomic_facts=set(atomic_facts),
                base_node_doc_ids=base_node_doc_ids,
                base_rel_doc_ids=base_rel_doc_ids,
                relationship_type=neo4j_relationship_type,
            )
        else:
            persist_graph_with_doc_id(
                resources.storage,
                knowledge_graph,
                target_graph_name,
                payload.doc_id,
                current_atomic_facts=set(atomic_facts),
                relationship_type=neo4j_relationship_type,
            )

        return BuildGraphResponse(
            success=True,
            graph_name=target_graph_name,
            base_graph_name=payload.graph_name,
            entities=len(knowledge_graph.entities),
            relationships=len(knowledge_graph.relationships),
            atomic_facts=len(atomic_facts),
            obs_timestamp=obs_timestamp,
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("构建图谱失败")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/graphs")
async def api_list_graphs() -> JSONResponse:
    resources = await get_resources()
    try:
        graphs = list_graphs_with_stats(resources.storage)
        return JSONResponse(content={"graphs": graphs})
    except Exception as e:
        logging.exception("列出图谱失败")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/graphs/{graph_name}/stats")
async def api_graph_stats(graph_name: str) -> JSONResponse:
    resources = await get_resources()
    if not graph_exists(resources.storage, graph_name):
        raise HTTPException(status_code=404, detail=f"图谱不存在: {graph_name}")
    try:
        stats = get_graph_stats(resources.storage, graph_name)
        return JSONResponse(content={"graph_name": graph_name, **stats})
    except Exception as e:
        logging.exception("获取图谱统计失败")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/graphs/{graph_name}/visualize", response_class=HTMLResponse)
async def api_visualize_graph(
    graph_name: str,
    doc_id: Optional[str] = Query(None, description="按 doc_id 过滤（逗号分隔，任一命中即保留）"),
    entities: Optional[str] = Query(None, description="实体名白名单（逗号分隔）"),
    predicates: Optional[str] = Query(None, description="谓词白名单（逗号分隔）"),
    q: Optional[str] = Query(None, description="关键字模糊匹配（实体名或谓词）"),
    center: Optional[str] = Query(None, description="中心实体名（配合 depth）"),
    depth: int = Query(1, ge=0, description="center 模式下的跳数"),
    limit_nodes: Optional[int] = Query(None, ge=1, description="节点返回上限（兜底保护）"),
    limit_edges: Optional[int] = Query(None, ge=0, description="边返回上限（兜底保护）"),
) -> HTMLResponse:
    resources = await get_resources()
    if not graph_exists(resources.storage, graph_name):
        raise HTTPException(status_code=404, detail=f"图谱不存在: {graph_name}")

    view_model, has_filters = _get_visualize_view_model(
        resources,
        graph_name=graph_name,
        doc_id=doc_id,
        entities=entities,
        predicates=predicates,
        q=q,
        center=center,
        depth=depth,
        limit_nodes=limit_nodes,
        limit_edges=limit_edges,
    )
    if not view_model.get("nodes") and not has_filters:
        raise HTTPException(status_code=404, detail=f"图谱为空或无法加载: {graph_name}")

    template = VISUALIZE_TEMPLATE_PATH.read_text(encoding="utf-8")
    html = template.replace("<!--DATA-->", json.dumps(view_model, ensure_ascii=False, indent=2))
    html = html.replace(
        "display: none; /* Hidden by default - controlled by main app */",
        "display: block; /* Visible in standalone/iframe mode */",
    )
    return HTMLResponse(content=html)


@app.get("/api/graphs/{graph_name}/visualize/data")
async def api_visualize_graph_data(
    graph_name: str,
    doc_id: Optional[str] = Query(None, description="按 doc_id 过滤（逗号分隔，任一命中即保留）"),
    entities: Optional[str] = Query(None, description="实体名白名单（逗号分隔）"),
    predicates: Optional[str] = Query(None, description="谓词白名单（逗号分隔）"),
    q: Optional[str] = Query(None, description="关键字模糊匹配（实体名或谓词）"),
    center: Optional[str] = Query(None, description="中心实体名（配合 depth）"),
    depth: int = Query(1, ge=0, description="center 模式下的跳数"),
    limit_nodes: Optional[int] = Query(None, ge=1, description="节点返回上限（兜底保护）"),
    limit_edges: Optional[int] = Query(None, ge=0, description="边返回上限（兜底保护）"),
) -> JSONResponse:
    resources = await get_resources()
    if not graph_exists(resources.storage, graph_name):
        raise HTTPException(status_code=404, detail=f"图谱不存在: {graph_name}")

    view_model, _ = _get_visualize_view_model(
        resources,
        graph_name=graph_name,
        doc_id=doc_id,
        entities=entities,
        predicates=predicates,
        q=q,
        center=center,
        depth=depth,
        limit_nodes=limit_nodes,
        limit_edges=limit_edges,
    )
    return JSONResponse(content=view_model)


@app.delete("/api/graphs/{graph_name}", response_model=DeleteGraphResponse)
async def api_delete_graph(graph_name: str) -> DeleteGraphResponse:
    resources = await get_resources()
    if not graph_exists(resources.storage, graph_name):
        raise HTTPException(status_code=404, detail=f"图谱不存在: {graph_name}")
    try:
        deleted_nodes, deleted_relationships = delete_graph(resources.storage, graph_name)
        return DeleteGraphResponse(
            success=True,
            message="图谱已删除",
            deleted_nodes=deleted_nodes,
            deleted_relationships=deleted_relationships,
        )
    except Exception as e:
        logging.exception("删除图谱失败")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/graphs/{graph_name}/rename")
async def api_rename_graph(graph_name: str, payload: RenameGraphRequest) -> JSONResponse:
    resources = await get_resources()
    if not graph_exists(resources.storage, graph_name):
        raise HTTPException(status_code=404, detail=f"图谱不存在: {graph_name}")
    new_name = payload.new_name.strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="new_name 不能为空")
    try:
        updated_nodes, updated_relationships = rename_graph(resources.storage, graph_name, new_name)
        return JSONResponse(
            content={
                "success": True,
                "message": "图谱已重命名",
                "old_name": graph_name,
                "new_name": new_name,
                "updated_nodes": updated_nodes,
                "updated_relationships": updated_relationships,
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logging.exception("重命名图谱失败")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/health")
async def api_health() -> JSONResponse:
    resources = await get_resources()
    return JSONResponse(
        content={
            "ok": True,
            "config_path": str(resources.config_path),
            "neo4j_connected": bool(resources.storage.driver is not None),
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8008")), reload=True)
