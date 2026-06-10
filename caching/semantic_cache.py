from __future__ import annotations

import json
import sqlite3
import uuid
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
import pandas as pd

from config.settings import settings
from embeddings.ollama_embeddings import OllamaEmbedder

@dataclass
class CacheEntry:
    id: str
    original_question: str
    normalized_question: str
    sql: str
    similarity: float = 1.0


class SemanticCache:
    def __init__(
        self,
        db_path: Path = settings.query_cache_db_path,
        faiss_path: Path = settings.query_cache_faiss_path,
        embedder: Optional[OllamaEmbedder] = None,
    ):
        self.db_path = db_path
        self.faiss_path = faiss_path
        self.embedder = embedder or OllamaEmbedder()
        self.index: Optional[faiss.Index] = None
        
        # Exact match / Near Exact match threshold
        self.exact_match_threshold = 0.98
        # Template match / Subset threshold
        self.template_match_threshold = 0.85
        
        self._init_db()
        self._load_faiss()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_cache (
                    id TEXT PRIMARY KEY,
                    original_question TEXT,
                    normalized_question TEXT,
                    sql TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def _load_faiss(self) -> None:
        if self.faiss_path.exists():
            try:
                self.index = faiss.read_index(str(self.faiss_path))
                # Ensure it's an IDMap; if not, we force a rebuild on next add
                if not isinstance(self.index, faiss.IndexIDMap) and not isinstance(self.index, faiss.IndexIDMap2):
                    self.index = None
            except Exception:
                self.index = None

    def _save_faiss(self) -> None:
        if self.index is not None:
            faiss.write_index(self.index, str(self.faiss_path))

    def search(self, normalized_question: str) -> Optional[CacheEntry]:
        if self.index is None:
            return None
            
        vector = self.embedder.embed(normalized_question).reshape(1, -1)
        scores, indexes = self.index.search(vector, 1)
        
        if len(scores) > 0 and len(scores[0]) > 0:
            score = float(scores[0][0])
            row_id = int(indexes[0][0])
            
            if row_id >= 0 and score >= self.template_match_threshold:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute("SELECT * FROM query_cache WHERE rowid = ?", (row_id,))
                    row = cursor.fetchone()
                    if row:
                        return CacheEntry(
                            id=row["id"],
                            original_question=row["original_question"],
                            normalized_question=row["normalized_question"],
                            sql=row["sql"],
                            similarity=score
                        )
        return None

    def add(self, original_question: str, normalized_question: str, sql: str) -> str:
        doc_id = str(uuid.uuid4())
        
        # 1. Clean expired caches periodically to prevent bloat
        self.clean_expired_data_caches(max_age_hours=24)
        
        # 2. Insert into SQLite to get the rowid
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO query_cache (id, original_question, normalized_question, sql) VALUES (?, ?, ?, ?)",
                (doc_id, original_question, normalized_question, sql)
            )
            row_id = cursor.lastrowid
            
        # 3. Add to FAISS IndexIDMap using the rowid
        vector = self.embedder.embed(normalized_question).reshape(1, -1)
        if self.index is None:
            base_index = faiss.IndexFlatIP(vector.shape[1])
            self.index = faiss.IndexIDMap(base_index)
            
        # If we failed to load an IDMap earlier but SQLite has records, 
        # we might be appending to a fresh index. 
        # For a completely robust system we'd rebuild, but appending works 
        # as long as we use explicit rowids.
        self.index.add_with_ids(vector, np.array([row_id], dtype=np.int64))
        self._save_faiss()
            
        return doc_id

    def save_data_cache(self, cache_id: str, df: pd.DataFrame) -> None:
        safe_id = re.sub(r'[^a-zA-Z0-9]', '_', cache_id)
        table_name = f"results_{safe_id}"
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            
    def has_data_cache(self, cache_id: str) -> bool:
        safe_id = re.sub(r'[^a-zA-Z0-9]', '_', cache_id)
        table_name = f"results_{safe_id}"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            return cursor.fetchone() is not None

    def clean_expired_data_caches(self, max_age_hours: int = 24) -> None:
        """Removes `results_*` data tables for cache entries older than max_age_hours."""
        with sqlite3.connect(self.db_path) as conn:
            # Find expired entries
            cursor = conn.execute(
                "SELECT id FROM query_cache WHERE timestamp < datetime('now', ?)",
                (f"-{max_age_hours} hours",)
            )
            expired_ids = [row[0] for row in cursor.fetchall()]
            
            # Drop their result tables to free space
            for eid in expired_ids:
                safe_id = re.sub(r'[^a-zA-Z0-9]', '_', eid)
                table_name = f"results_{safe_id}"
                conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                
            # Optional: Delete them from query_cache table? 
            # We will keep them for the Semantic Cache hits to avoid losing the learned SQL, 
            # we just drop the bloated data table.

