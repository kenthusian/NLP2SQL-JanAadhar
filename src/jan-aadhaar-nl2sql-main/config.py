"""
config.py — Centralized configuration for Jan-Aadhaar NL2SQL Pipeline.
All tunables live here; import this module everywhere else.
"""
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR = ROOT_DIR / "data" / "aadhaar"           # Partitioned Parquet files
SCHEMA_JSON = ROOT_DIR / "data" / "schema.json"   # Auto-generated column metadata
CHROMA_DIR = ROOT_DIR / "data" / "chroma_store"   # ChromaDB persistent storage
LOG_FILE = ROOT_DIR / "data" / "pipeline.log"

# ── Real schema (29 columns) ─────────────────────────────────────────────────
# Column name → (DuckDB type, human description)
COLUMN_SCHEMA: dict[str, tuple[str, str]] = {
    "DISTRICT_NAME_ENG":  ("VARCHAR",  "District name in English"),
    "IS_RURAL":           ("VARCHAR",  "Whether the area is Rural or Urban (R/U)"),
    "BLOCK_NAME_ENG":     ("VARCHAR",  "Block/Tehsil name in English (rural areas)"),
    "CITY_NAME_ENG":      ("VARCHAR",  "City/Town name in English (urban areas)"),
    "WARD_NAME_ENG":      ("VARCHAR",  "Ward name in English (urban areas)"),
    "GP_NAME_ENG":        ("VARCHAR",  "Gram Panchayat name in English (rural areas)"),
    "VILL_NAME_ENG":      ("VARCHAR",  "Village name in English (rural areas)"),
    "ENROLLMENT_ID":      ("VARCHAR",  "Unique household enrollment identifier"),
    "MEMBER_ID":          ("VARCHAR",  "Unique member identifier within the household"),
    "MEM_TYPE":           ("VARCHAR",  "Member type (HOF = Head of Family, etc.)"),
    "RELATION_WITH_HOF":  ("VARCHAR",  "Relation of member with Head of Family"),
    "NAME_EN":            ("VARCHAR",  "Member full name in English"),
    "FATHER_NAME_EN":     ("VARCHAR",  "Father's name in English"),
    "MOTHER_NAME_EN":     ("VARCHAR",  "Mother's name in English"),
    "MARITAL_STATUS":     ("VARCHAR",  "Marital status (Married/Unmarried/Widowed/Divorced)"),
    "SPOUCE_NAME_EN":     ("VARCHAR",  "Spouse name in English (if married)"),
    "DOB":                ("DATE",     "Date of birth (YYYY-MM-DD)"),
    "AGE":                ("INTEGER",  "Age of the member in years"),
    "GENDER":             ("VARCHAR",  "Gender (M/F/T)"),
    "CASTE_CATEGORY":     ("VARCHAR",  "Caste category (GEN/OBC/SC/ST)"),
    "CASTE":              ("VARCHAR",  "Specific caste name"),
    "BANK":               ("VARCHAR",  "Bank name for direct benefit transfer"),
    "IFSC_CODE":          ("VARCHAR",  "IFSC code of the bank branch"),
    "ACCOUNT_NO":         ("VARCHAR",  "Bank account number"),
    "MOBILE_NO":          ("VARCHAR",  "Mobile/phone number"),
    "INCOME":             ("DOUBLE",   "Annual household income in INR"),
    "OCCUPATION":         ("VARCHAR",  "Primary occupation of the member"),
    "MINORITY":           ("VARCHAR",  "Whether the member belongs to a minority community (Y/N)"),
    "EDUCATION":          ("VARCHAR",  "Highest education level attained"),
}

# ── Table name used in DuckDB SQL ────────────────────────────────────────────
TABLE_NAME = "aadhaar"

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5-coder:3b"
OLLAMA_TIMEOUT = 300  # seconds

# ── RAG / ChromaDB ────────────────────────────────────────────────────────────
CACHE_COLLECTION = "query_cache"
SCHEMA_COLLECTION = "schema_index"
CACHE_SIMILARITY_THRESHOLD = 0.82   # cosine similarity ≥ this → cache hit
RAG_TOP_K = 5                       # number of columns to inject into prompt

# ── LLM generation ────────────────────────────────────────────────────────────
LLM_TEMPERATURE = 0.0
MAX_CORRECTION_ATTEMPTS = 2

# ── Server ports ─────────────────────────────────────────────────────────────
FASTAPI_PORT = 8000
STREAMLIT_PORT = 8501

# ── Ingestion ─────────────────────────────────────────────────────────────────
INGEST_CHUNK_ROWS = 1_000_000       # rows per chunk when reading source CSV
PARTITION_COL = "DISTRICT_NAME_ENG" # Hive partition key
MOCK_ROW_COUNT = 80_000_000         # synthetic dataset size
