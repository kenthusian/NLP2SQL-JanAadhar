"""
rag/cache.py — Semantic query cache backed by ChromaDB.

Flow:
  1. On every new query, embed the prompt and search the collection.
  2. If cosine similarity ≥ CACHE_SIMILARITY_THRESHOLD → return the cached SQL.
  3. After a successful LLM+DuckDB round-trip → upsert the (prompt, sql) pair.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CACHE_COLLECTION, CACHE_SIMILARITY_THRESHOLD, CHROMA_DIR
from logger import get_logger

log = get_logger("nl2sql.cache")

# Lazy import so the module can be loaded even if chromadb isn't installed yet
_client = None
_collection = None


def _get_collection():
    global _client, _collection
    
    try:
        # Check if collection is still valid (might have been deleted externally)
        if _collection is not None:
            _collection.count()
    except Exception:
        _collection = None

    if _collection is None:
        import chromadb
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        # Use DefaultEmbeddingFunction (onnxruntime) as it already natively uses all-MiniLM-L6-v2
        # and avoids PyTorch's 260-character Windows Long Path installation errors.
        emb_fn = chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
        _collection = _client.get_or_create_collection(
            name=CACHE_COLLECTION,
            embedding_function=emb_fn,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(f"ChromaDB query cache ready ({_collection.count()} entries)")
    return _collection


def _extract_all_numbers(prompt: str) -> list[str]:
    """Extract digits and common English/Hindi number words from the prompt."""
    import re
    # Digits
    digits = re.findall(r'\b\d+(?:\.\d+)?\b', prompt)
    
    # Common number words
    words = [
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
        "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
        "hundred", "thousand", "lakh", "crore", "million", "billion",
        "ek", "do", "teen", "char", "chaar", "paanch", "chhah", "saat", "aath", "nau", "das", "sau", "hazaar"
    ]
    pattern = r'\b(?:' + '|'.join(words) + r')\b'
    found_words = re.findall(pattern, prompt.lower())
    
    # Math operators
    ops = re.findall(r'[><=!]+', prompt)
    
    # Text modifiers
    mod_words = [
        "more than", "greater than", "less than", "fewer than", "at least", "at most", 
        "exactly", "over", "under", "above", "below"
    ]
    mod_pattern = r'\b(?:' + '|'.join(mod_words) + r')\b'
    found_mods = re.findall(mod_pattern, prompt.lower())
    
    return digits + found_words + ops + found_mods

def _extract_intent(prompt: str) -> str:
    """Determine if the query is a 'count' or 'list' query to avoid cache collisions."""
    import re
    prompt_lower = prompt.lower()
    
    count_words = r'\b(how many|count|total|number of)\b'
    if re.search(count_words, prompt_lower):
        return "count"
        
    list_words = r'\b(list|show|what|who|which|find|get|details of|data of)\b'
    if re.search(list_words, prompt_lower):
        return "list"
        
    return "unknown"

def strip_noise(text: str) -> str:
    """Remove conversational noise before embedding."""
    import re
    stopwords = ["show me", "what is", "can you", "please", "count of", "give me", "find", "get", "the", "a", "an"]
    text = text.lower()
    for word in stopwords:
        text = re.sub(rf'\b{word}\b', '', text)
    # Remove extra spaces left by removals
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def _prompt_id(prompt: str, domain_hints: list[str] | None = None) -> str:
    """Stable ID for deduplication — SHA-256 hex of the normalised prompt + hints."""
    text = prompt.strip().lower()
    if domain_hints:
        text += " | " + " | ".join(sorted(domain_hints)).lower()
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def exact_lookup(prompt: str, domain_hints: list[str] | None = None) -> str | None:
    """O(1) exact match lookup bypassing the ONNX embedding model."""
    col = _get_collection()
    if col.count() == 0:
        return None

    doc_id = _prompt_id(prompt, domain_hints)
    results = col.get(ids=[doc_id])
    if results["ids"]:
        cached_sql = results["metadatas"][0]["sql"]
        log.info(f"Exact Cache HIT → {cached_sql[:80]!r}")
        return cached_sql
    return None


def semantic_lookup(prompt: str, domain_hints: list[str] | None = None, unmapped_words: list[str] | None = None) -> tuple[str, str] | None:
    """
    Return (cached_sql, cached_data) if a semantically similar prompt exists, else None.
    Appends domain_hints and explicit numbers to the search text to prevent >95% collision.
    """
    col = _get_collection()
    if col.count() == 0:
        return None

    search_text = strip_noise(prompt)
    if domain_hints:
        search_text += " " + " ".join(domain_hints)
        
    if unmapped_words:
        search_text += " " + " ".join(unmapped_words)
    
    numbers = _extract_all_numbers(prompt)
    if numbers:
        search_text += " " + " ".join(numbers)

    results = col.query(
        query_texts=[search_text],
        n_results=1,
        include=["documents", "distances", "metadatas"],
    )

    if not results["ids"][0]:
        return None

    distance = results["distances"][0][0]     # cosine distance (0 = identical)
    similarity = 1.0 - distance               # convert to similarity

    if similarity >= CACHE_SIMILARITY_THRESHOLD:
        cached_meta = results["metadatas"][0][0]
        cached_sql = cached_meta["sql"]
        
        # Prevent numerical collisions (e.g. '8 lakh' vs '4 lakh' scoring high similarity)
        cached_numbers = cached_meta.get("numbers", "")
        incoming_numbers = ",".join(numbers)
        if incoming_numbers != cached_numbers:
            log.debug(f"Cache MISS (similarity={similarity:.4f} but numbers mismatch: {incoming_numbers} != {cached_numbers})")
            return None
            
        # Prevent intent collisions (e.g. 'show list' vs 'how many')
        cached_intent = cached_meta.get("intent", "unknown")
        incoming_intent = _extract_intent(prompt)
        if incoming_intent != "unknown" and cached_intent != "unknown" and incoming_intent != cached_intent:
            log.debug(f"Cache MISS (similarity={similarity:.4f} but intent mismatch: {incoming_intent} != {cached_intent})")
            return None
            
        # Prevent domain hint collisions (e.g., query A had district=Jaipur, query B has no district)
        cached_hints = cached_meta.get("domain_hints", "")
        incoming_hints = "||".join(sorted(domain_hints)) if domain_hints else ""
        if incoming_hints != cached_hints:
            log.debug(f"Cache MISS (similarity={similarity:.4f} but domain hints mismatch)")
            return None
            
        # Prevent unmapped word collisions (e.g., query A had unmapped 'pichkarai', query B didn't)
        cached_unmapped = cached_meta.get("unmapped_words", "")
        incoming_unmapped = "||".join(sorted(unmapped_words)) if unmapped_words else ""
        if incoming_unmapped != cached_unmapped:
            log.debug(f"Cache MISS (similarity={similarity:.4f} but unmapped words mismatch: {incoming_unmapped} != {cached_unmapped})")
            return None
            
            
        cached_data = cached_meta.get("data", "{}")
        log.info(f"Semantic Cache HIT (similarity={similarity:.4f}) → {cached_sql[:80]!r}")
        return cached_sql, cached_data

    log.debug(f"Cache MISS (similarity={similarity:.4f})")
    return None


def store(prompt: str, sql: str, db_result: dict, domain_hints: list[str] | None = None, unmapped_words: list[str] | None = None) -> None:
    """
    Upsert the (prompt → sql, data) pair into the cache.
    """
    import json
    col = _get_collection()
    doc_id = _prompt_id(prompt, domain_hints)

    store_text = prompt
    if domain_hints:
        store_text += " " + " ".join(domain_hints)
        
    if unmapped_words:
        store_text += " " + " ".join(unmapped_words)

    numbers = _extract_all_numbers(prompt)
    if numbers:
        store_text += " " + " ".join(numbers)

    intent = _extract_intent(prompt)

    hints_str = "||".join(sorted(domain_hints)) if domain_hints else ""
    unmapped_str = "||".join(sorted(unmapped_words)) if unmapped_words else ""

    col.upsert(
        ids=[doc_id],
        documents=[store_text],
        metadatas=[{
            "sql": sql, 
            "data": json.dumps(db_result, default=str), 
            "numbers": ",".join(numbers), 
            "intent": intent,
            "domain_hints": hints_str,
            "unmapped_words": unmapped_str
        }],
    )
    log.info(f"Cache STORE id={doc_id[:8]}… prompt={prompt[:60]!r}")


def cache_size() -> int:
    """Return the number of cached Q→SQL pairs."""
    return _get_collection().count()
