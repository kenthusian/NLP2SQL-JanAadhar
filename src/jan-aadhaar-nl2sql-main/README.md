# Jan-Aadhaar NL2SQL Pipeline

A fully local, zero-API-call **Natural Language → SQL** engine for querying an 80-million-row Jan-Aadhaar welfare dataset. Built on DuckDB + Ollama (qwen2.5-coder:3b) + ChromaDB.

---

## Architecture

```
User Question
    │
    ▼
┌─────────────┐     HIT     ┌──────────────────┐
│ Query Cache │────────────►│  DuckDB Execute  │
│ (ChromaDB)  │             └──────────────────┘
└─────────────┘
    │ MISS
    ▼
┌─────────────┐
│ Schema RAG  │  (retrieves top-7 relevant columns)
│ (ChromaDB)  │
└─────────────┘
    │
    ▼
┌─────────────┐
│ Prompt      │  (system + schema + 8 few-shots + question)
│ Builder     │
└─────────────┘
    │
    ▼
┌─────────────┐
│ Ollama LLM  │  qwen2.5-coder:3b @ localhost:11434
└─────────────┘
    │
    ▼
┌─────────────┐
│ SQL Guard   │  sqlglot AST parse + whitelist + self-correction (2 retries)
│ (sqlglot)   │
└─────────────┘
    │
    ▼
┌─────────────┐
│ DuckDB      │  Partitioned Parquet (Hive-style by DISTRICT_NAME_ENG)
└─────────────┘
    │
    ▼
  Results → FastAPI → Streamlit UI
```

---

## Dataset Schema (29 columns)

| Column | Type | Description |
|--------|------|-------------|
| DISTRICT_NAME_ENG | VARCHAR | District name in English |
| IS_RURAL | VARCHAR | R = Rural, U = Urban |
| BLOCK_NAME_ENG | VARCHAR | Block/Tehsil name (rural) |
| CITY_NAME_ENG | VARCHAR | City name (urban) |
| WARD_NAME_ENG | VARCHAR | Ward name (urban) |
| GP_NAME_ENG | VARCHAR | Gram Panchayat name (rural) |
| VILL_NAME_ENG | VARCHAR | Village name (rural) |
| ENROLLMENT_ID | VARCHAR | Household enrollment ID |
| MEMBER_ID | VARCHAR | Member ID |
| MEM_TYPE | VARCHAR | HOF or MEMBER |
| RELATION_WITH_HOF | VARCHAR | Relation with Head of Family |
| NAME_EN | VARCHAR | Member name |
| FATHER_NAME_EN | VARCHAR | Father's name |
| MOTHER_NAME_EN | VARCHAR | Mother's name |
| MARITAL_STATUS | VARCHAR | Married/Unmarried/Widowed/Divorced |
| SPOUCE_NAME_EN | VARCHAR | Spouse name |
| DOB | DATE | Date of birth |
| AGE | INTEGER | Age in years |
| GENDER | VARCHAR | M/F/T |
| CASTE_CATEGORY | VARCHAR | GEN/OBC/SC/ST |
| CASTE | VARCHAR | Specific caste name |
| BANK | VARCHAR | Bank name |
| IFSC_CODE | VARCHAR | Bank IFSC code |
| ACCOUNT_NO | VARCHAR | Bank account number |
| MOBILE_NO | VARCHAR | Mobile number |
| INCOME | DOUBLE | Annual income (INR) |
| OCCUPATION | VARCHAR | Primary occupation |
| MINORITY | VARCHAR | Y/N |
| EDUCATION | VARCHAR | Highest education level |

---

## Quick Start

### 1. Install dependencies

```powershell
pip install -r requirements.txt
```

### 2. Start Ollama and pull the model

```powershell
ollama serve          # In a separate terminal
ollama pull qwen2.5-coder:3b
```

### 3. Ingest your data

**Option A — Real CSV:**
```powershell
python -m db.ingest csv --source "C:\path\to\your\data.csv"
```

**Option B — Synthetic 80M-row mock (takes ~10 min):**
```powershell
python -m db.ingest mock --rows 80000000
```

**Option C — Small mock for testing (fast):**
```powershell
python -m db.ingest mock --rows 100000
```

### 4. Start the backend

```powershell
python main.py
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

### 5. Start the UI (separate terminal)

```powershell
streamlit run ui/streamlit_app.py --server.port 8501
# UI: http://localhost:8501
```

---

## Running Tests

```powershell
pip install pytest
python -m pytest tests/ -v
```

---

## Project Structure

```
jan-aadhaar-nl2sql/
├── config.py                    # All tunables (ports, thresholds, schema)
├── logger.py                    # Structured JSON logger with stage timing
├── main.py                      # Backend entrypoint (uvicorn + preflight checks)
├── requirements.txt
├── db/
│   ├── ingest.py                # CSV → Parquet + synthetic generator
│   └── query.py                 # Thread-safe DuckDB executor
├── rag/
│   ├── cache.py                 # Semantic query cache (ChromaDB)
│   └── schema_index.py          # Column-level RAG (ChromaDB)
├── llm/
│   ├── prompt_builder.py        # Prompt assembly + few-shot examples
│   └── ollama_client.py         # Ollama HTTP client
├── validation/
│   └── sql_guard.py             # sqlglot AST validation + self-correction
├── ui/
│   ├── api.py                   # FastAPI endpoints
│   └── streamlit_app.py         # Streamlit dark-mode UI
├── tests/
│   └── test_sql_guard.py        # 14 pytest unit tests
└── data/                        # (git-ignored)
    ├── aadhaar/                 # Partitioned Parquet files
    ├── chroma_store/            # ChromaDB persistent storage
    └── schema.json              # Auto-generated column metadata
```

---

## Configuration

Edit `config.py` to change:

| Setting | Default | Description |
|---------|---------|-------------|
| `FASTAPI_PORT` | 8000 | Backend API port |
| `STREAMLIT_PORT` | 8501 | Streamlit UI port |
| `OLLAMA_MODEL` | qwen2.5-coder:3b | LLM model |
| `CACHE_SIMILARITY_THRESHOLD` | 0.95 | Cache hit threshold |
| `RAG_TOP_K` | 7 | Columns injected into prompt |
| `MAX_CORRECTION_ATTEMPTS` | 2 | SQL self-correction retries |
| `PARTITION_COL` | DISTRICT_NAME_ENG | Parquet partition key |
