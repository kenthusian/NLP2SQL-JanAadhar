"""
rag/schema_index.py — Column-level semantic search for dynamic schema pruning.

At startup:
  - Build an embedding for each column's (name + description).
  - Store in a persistent ChromaDB collection.

At query time:
  - Embed the user's prompt.
  - Retrieve the TOP_K most relevant columns.
  - Pass those column definitions (name, type, description) to the LLM.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    CHROMA_DIR,
    COLUMN_SCHEMA,
    RAG_TOP_K,
    SCHEMA_COLLECTION,
    SCHEMA_JSON,
    TABLE_NAME,
)
from logger import get_logger

log = get_logger("nl2sql.schema_index")


class ColumnDef(TypedDict):
    name: str
    dtype: str
    description: str


_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        import chromadb

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        # Use DefaultEmbeddingFunction (onnxruntime) as it already natively uses all-MiniLM-L6-v2
        # and avoids PyTorch's 260-character Windows Long Path installation errors.
        emb_fn = chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
        _collection = _client.get_or_create_collection(
            name=SCHEMA_COLLECTION,
            embedding_function=emb_fn,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def build_index(schema_json_path: Path | None = None) -> None:
    """
    Embed each column definition and upsert into ChromaDB.
    Safe to call multiple times (idempotent via upsert).

    If *schema_json_path* is provided, uses it; otherwise falls back to
    the hard-coded COLUMN_SCHEMA in config.py.
    """
    col = _get_collection()

    # Load schema — prefer the JSON file if present (richer, runtime-aware)
    if schema_json_path and Path(schema_json_path).exists():
        with open(schema_json_path, encoding="utf-8") as f:
            raw: dict = json.load(f)
        columns: dict[str, tuple[str, str]] = {
            k: (v["type"], v["description"]) for k, v in raw.items()
        }
    else:
        columns = {k: (dtype, desc) for k, (dtype, desc) in COLUMN_SCHEMA.items()}

    ids, docs, metas = [], [], []
    for col_name, (dtype, description) in columns.items():
        doc = f"Column: {col_name} ({dtype}). {description}"
        ids.append(col_name)
        docs.append(doc)
        metas.append({"name": col_name, "dtype": dtype, "description": description})

    col.upsert(ids=ids, documents=docs, metadatas=metas)
    log.info(f"Schema index built/updated: {len(ids)} columns")


def retrieve(prompt: str, top_k: int = RAG_TOP_K) -> list[ColumnDef]:
    """
    Return the *top_k* most semantically relevant columns for *prompt*.

    Args:
        prompt: The user's natural language question.
        top_k:  Maximum number of columns to return.

    Returns:
        List of ColumnDef dicts (name, dtype, description), ordered by relevance.
    """
    col = _get_collection()

    if col.count() == 0:
        # Index not built yet — fall back to returning ALL columns
        log.warning("Schema index empty; returning all columns")
        return [
            ColumnDef(name=k, dtype=dtype, description=desc)
            for k, (dtype, desc) in COLUMN_SCHEMA.items()
        ]

    results = col.query(
        query_texts=[prompt],
        n_results=min(top_k, col.count()),
        include=["metadatas", "distances"],
    )

    out: list[ColumnDef] = []
    for meta in results["metadatas"][0]:
        out.append(ColumnDef(name=meta["name"], dtype=meta["dtype"],
                             description=meta["description"]))

    log.debug(f"Schema RAG retrieved {len(out)} cols for prompt={prompt[:60]!r}")
    return out


def all_column_names() -> list[str]:
    """Return all known column names (from config, not ChromaDB)."""
    return list(COLUMN_SCHEMA.keys())
