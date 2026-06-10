"""
db/ingest.py — CSV → partitioned Parquet ingestion + synthetic mock generator.

Two entry-points:
  1. ingest_csv(source_path)       — real dataset ingestion
  2. generate_mock_dataset()       — synthetic 80M-row generator (gated)
  3. extract_schema()              — writes data/schema.json from Parquet
"""
import json
import sys
import time
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

# Ensure project root is on path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    COLUMN_SCHEMA,
    DATA_DIR,
    INGEST_CHUNK_ROWS,
    MOCK_ROW_COUNT,
    PARTITION_COL,
    SCHEMA_JSON,
    TABLE_NAME,
)
from logger import get_logger, log_event

log = get_logger("nl2sql.ingest")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _build_pyarrow_schema() -> pa.Schema:
    """Map COLUMN_SCHEMA to a PyArrow schema."""
    type_map = {
        "VARCHAR": pa.string(),
        "DATE":    pa.date32(),
        "INTEGER": pa.int32(),
        "DOUBLE":  pa.float64(),
        "BIGINT":  pa.int64(),
    }
    fields = [pa.field(col, type_map.get(dtype, pa.string()))
              for col, (dtype, _) in COLUMN_SCHEMA.items()]
    return pa.Schema.from_pandas(pd.DataFrame(columns=[c for c in COLUMN_SCHEMA]),
                                  preserve_index=False) if False else pa.schema(fields)


# ── 1. Real CSV Ingestion ─────────────────────────────────────────────────────

def ingest_csv(source_path: str | Path, partition_col: str = PARTITION_COL) -> None:
    """
    Read *source_path* CSV in 1M-row chunks and write Hive-style partitioned
    Parquet files into DATA_DIR.

    Usage::

        python -m db.ingest --source /path/to/data.csv
    """
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_path}")

    _ensure_data_dir()
    schema = _build_pyarrow_schema()

    log_event("ingest_start", source=str(source_path))
    t0 = time.perf_counter()
    total_rows = 0

    reader = pd.read_csv(
        source_path,
        chunksize=INGEST_CHUNK_ROWS,
        dtype=str,           # read everything as string first; cast below
        low_memory=False,
    )

    for chunk_idx, chunk in enumerate(reader):
        # Normalise column names to uppercase
        chunk.columns = [c.strip().upper() for c in chunk.columns]

        # Coerce types — handles real CSV format
        if "AGE" in chunk.columns:
            chunk["AGE"] = pd.to_numeric(chunk["AGE"], errors="coerce").astype("Int32")
        if "INCOME" in chunk.columns:
            chunk["INCOME"] = pd.to_numeric(chunk["INCOME"], errors="coerce")
        if "IS_RURAL" in chunk.columns:
            # Real data uses 1/0 integers; normalise to string '1'/'0' for uniformity
            chunk["IS_RURAL"] = chunk["IS_RURAL"].astype(str)
        if "DOB" in chunk.columns:
            # Real format: '03-May-2006' (dd-Mon-YYYY)
            chunk["DOB"] = pd.to_datetime(
                chunk["DOB"], format="%d-%b-%Y", errors="coerce"
            ).dt.date

        table = pa.Table.from_pandas(chunk, preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=str(DATA_DIR),
            partition_cols=[partition_col] if partition_col in chunk.columns else [],
            existing_data_behavior="overwrite_or_ignore",
            compression="ZSTD",
        )
        total_rows += len(chunk)
        elapsed = time.perf_counter() - t0
        log.info(f"Chunk {chunk_idx+1}: {total_rows:,} rows written ({elapsed:.1f}s)")

    log_event("ingest_done", total_rows=total_rows,
              elapsed_s=round(time.perf_counter() - t0, 1))
    print(f"[OK] Ingestion complete - {total_rows:,} rows -> {DATA_DIR}")
    extract_schema()


# ── 2. Synthetic Mock Generator ───────────────────────────────────────────────

def generate_mock_dataset(n_rows: int = MOCK_ROW_COUNT, chunk_size: int = 500_000) -> None:
    """
    Generate a realistic synthetic dataset and write it as partitioned Parquet.
    **This function will NOT execute unless you explicitly call it.**
    Estimated time: ~8-12 min on a mid-range CPU for 80M rows.
    """
    try:
        from faker import Faker
        import random, numpy as np
    except ImportError:
        print("Install faker: pip install faker numpy")
        sys.exit(1)

    fake = Faker("en_IN")
    _ensure_data_dir()

    DISTRICTS = [
        "Jaipur", "Jodhpur", "Udaipur", "Kota", "Ajmer", "Bikaner",
        "Alwar", "Bharatpur", "Sikar", "Nagaur", "Barmer", "Pali",
        "Sri Ganganagar", "Jhunjhunu", "Churu", "Bhilwara", "Tonk",
        "Sawai Madhopur", "Bundi", "Baran",
    ]
    CASTES_CAT = ["GEN", "OBC", "SC", "ST"]
    GENDERS = ["M", "F"]
    MARITAL = ["Married", "Unmarried", "Widowed", "Divorced"]
    OCCUPATIONS = [
        "Farmer", "Labourer", "Business", "Service", "Housewife",
        "Student", "Unemployed", "Artisan", "Shopkeeper", "Driver",
    ]
    EDUCATION_LEVELS = [
        "Illiterate", "Primary", "Middle", "Secondary", "Senior Secondary",
        "Graduate", "Post Graduate", "Diploma", "Professional",
    ]
    BANKS = [
        "State Bank of India", "Bank of Baroda", "Punjab National Bank",
        "Rajasthan Marudhara Gramin Bank", "Baroda Rajasthan Kshetriya Gramin Bank",
        "HDFC Bank", "ICICI Bank",
    ]

    log_event("mock_start", n_rows=n_rows)
    t0 = time.perf_counter()
    total_written = 0

    for chunk_start in range(0, n_rows, chunk_size):
        actual_size = min(chunk_size, n_rows - chunk_start)
        rng = np.random.default_rng(chunk_start)

        districts = rng.choice(DISTRICTS, size=actual_size)
        is_rural = rng.choice(["R", "U"], size=actual_size, p=[0.72, 0.28])
        ages = rng.integers(0, 95, size=actual_size)
        genders = rng.choice(GENDERS, size=actual_size)
        incomes = rng.lognormal(mean=10.5, sigma=1.2, size=actual_size).round(2)

        rows = {
            "DISTRICT_NAME_ENG": districts,
            "IS_RURAL":          is_rural,
            "BLOCK_NAME_ENG":    [fake.city() if r == "R" else "" for r in is_rural],
            "CITY_NAME_ENG":     [fake.city() if r == "U" else "" for r in is_rural],
            "WARD_NAME_ENG":     [f"Ward {rng.integers(1,50)}" if r == "U" else "" for r in is_rural],
            "GP_NAME_ENG":       [fake.last_name() + " GP" if r == "R" else "" for r in is_rural],
            "VILL_NAME_ENG":     [fake.city() if r == "R" else "" for r in is_rural],
            "ENROLLMENT_ID":     [f"RJ{rng.integers(10**9, 10**10)}" for _ in range(actual_size)],
            "MEMBER_ID":         [f"M{rng.integers(10**7, 10**8)}" for _ in range(actual_size)],
            "MEM_TYPE":          rng.choice(["HOF", "MEMBER"], size=actual_size, p=[0.25, 0.75]),
            "RELATION_WITH_HOF": rng.choice(
                ["SELF", "WIFE", "SON", "DAUGHTER", "FATHER", "MOTHER", "BROTHER", "SISTER"],
                size=actual_size
            ),
            "NAME_EN":           [fake.name() for _ in range(actual_size)],
            "FATHER_NAME_EN":    [fake.name_male() for _ in range(actual_size)],
            "MOTHER_NAME_EN":    [fake.name_female() for _ in range(actual_size)],
            "MARITAL_STATUS":    rng.choice(MARITAL, size=actual_size),
            "SPOUCE_NAME_EN":    [fake.name() if m == "Married" else "" for m in
                                  rng.choice(MARITAL, size=actual_size)],
            "DOB":               pd.to_datetime(
                ["2024-01-01"] * actual_size
            ) - pd.to_timedelta(ages * 365, unit="D"),
            "AGE":               ages,
            "GENDER":            genders,
            "CASTE_CATEGORY":    rng.choice(CASTES_CAT, size=actual_size, p=[0.30, 0.43, 0.17, 0.10]),
            "CASTE":             [fake.last_name() for _ in range(actual_size)],
            "BANK":              rng.choice(BANKS, size=actual_size),
            "IFSC_CODE":         [f"SBIN{rng.integers(1000000, 9999999)}" for _ in range(actual_size)],
            "ACCOUNT_NO":        [str(rng.integers(10**11, 10**12)) for _ in range(actual_size)],
            "MOBILE_NO":         [f"9{rng.integers(10**9, 10**10-1)}" for _ in range(actual_size)],
            "INCOME":            incomes,
            "OCCUPATION":        rng.choice(OCCUPATIONS, size=actual_size),
            "MINORITY":          rng.choice(["Y", "N"], size=actual_size, p=[0.15, 0.85]),
            "EDUCATION":         rng.choice(EDUCATION_LEVELS, size=actual_size),
        }

        df = pd.DataFrame(rows)
        df["DOB"] = df["DOB"].dt.date

        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=str(DATA_DIR),
            partition_cols=[PARTITION_COL],
            existing_data_behavior="overwrite_or_ignore",
            compression="ZSTD",
        )
        total_written += actual_size
        pct = total_written / n_rows * 100
        elapsed = time.perf_counter() - t0
        print(f"\r  {total_written:>12,} / {n_rows:,} rows  ({pct:.1f}%)  {elapsed:.0f}s", end="")

    print(f"\n[OK] Mock dataset done - {total_written:,} rows -> {DATA_DIR}")
    log_event("mock_done", total_rows=total_written,
              elapsed_s=round(time.perf_counter() - t0, 1))
    extract_schema()


# ── 3. Schema Extraction ──────────────────────────────────────────────────────

def extract_schema() -> dict:
    """
    Use DuckDB to introspect the Parquet files and write data/schema.json.
    Also augments the schema with human-readable descriptions from COLUMN_SCHEMA.
    """
    SCHEMA_JSON.parent.mkdir(parents=True, exist_ok=True)
    parquet_glob = str(DATA_DIR / "**" / "*.parquet")

    con = duckdb.connect(":memory:")
    try:
        rows = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{parquet_glob}', hive_partitioning=true)"
        ).fetchall()
    finally:
        con.close()

    schema = {}
    for col_name, col_type, *_ in rows:
        col_upper = col_name.upper()
        _, description = COLUMN_SCHEMA.get(col_upper, (col_type, f"Column {col_upper}"))
        schema[col_upper] = {
            "type": col_type,
            "description": description,
        }

    with open(SCHEMA_JSON, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    log_event("schema_extracted", columns=list(schema.keys()), path=str(SCHEMA_JSON))
    print(f"[OK] Schema written to {SCHEMA_JSON} ({len(schema)} columns)")
    return schema


# ── 4. Pre-Aggregated Rollups ──────────────────────────────────────────────────

def generate_rollups() -> None:
    """
    Generate high-level pre-aggregated metrics into district_rollup.parquet.
    This bypasses querying raw 80M rows for simple district-level statistics.
    """
    parquet_glob = str(DATA_DIR / "**" / "*.parquet")
    rollup_path = str(DATA_DIR.parent / "district_rollup.parquet")
    
    # Do not include the rollup itself if it already exists
    con = duckdb.connect(":memory:")
    con.execute("PRAGMA threads = 8;")
    con.execute("PRAGMA memory_limit = '8GB';")
    
    query = f"""
    COPY (
        SELECT DISTRICT_NAME_ENG, IS_RURAL, GENDER, CASTE_CATEGORY, COUNT(*) AS total_members
        FROM read_parquet('{parquet_glob}', hive_partitioning=true)
        WHERE DISTRICT_NAME_ENG IS NOT NULL
        GROUP BY DISTRICT_NAME_ENG, IS_RURAL, GENDER, CASTE_CATEGORY
    ) TO '{rollup_path}' (FORMAT PARQUET, COMPRESSION 'ZSTD');
    """
    
    log_event("rollup_start")
    t0 = time.perf_counter()
    con.execute(query)
    elapsed = time.perf_counter() - t0
    
    log_event("rollup_done", elapsed_s=round(elapsed, 1), path=rollup_path)
    print(f"[OK] Rollup written to {rollup_path} ({elapsed:.1f}s)")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Jan-Aadhaar data ingestion")
    sub = parser.add_subparsers(dest="command")

    p_csv = sub.add_parser("csv", help="Ingest a real CSV file")
    p_csv.add_argument("--source", required=True, help="Path to source CSV")
    p_csv.add_argument("--partition-col", default=PARTITION_COL)

    p_mock = sub.add_parser("mock", help="Generate synthetic mock dataset")
    p_mock.add_argument("--rows", type=int, default=MOCK_ROW_COUNT)

    p_schema = sub.add_parser("schema", help="Extract schema from existing Parquet files")

    p_rollup = sub.add_parser("rollup", help="Generate district rollup Parquet file")

    args = parser.parse_args()

    if args.command == "csv":
        ingest_csv(args.source, args.partition_col)
    elif args.command == "mock":
        print(f"⚠️  About to generate {args.rows:,} synthetic rows into {DATA_DIR}")
        confirm = input("Type YES to proceed: ")
        if confirm.strip().upper() == "YES":
            generate_mock_dataset(args.rows)
        else:
            print("Aborted.")
    elif args.command == "schema":
        extract_schema()
    elif args.command == "rollup":
        generate_rollups()
    else:
        parser.print_help()
