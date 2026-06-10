from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import pandas as pd
from sqlalchemy import text

from database.connection import get_engine
from database.models import Base, Citizen


# Excel column → DB column mapping
_COL_MAP = {
    "DISTRICT_NAME_ENG": "district",
    "IS_RURAL":          "is_rural",
    "BLOCK_NAME_ENG":    "block",
    "CITY_NAME_ENG":     "city",
    "WARD_NAME_ENG":     "ward",
    "GP_NAME_ENG":       "gram_panchayat",
    "VILL_NAME_ENG":     "village",
    "ENROLLMENT_ID":     "enrollment_id",
    "MEMBER_ID":         "member_id",
    "MEM_TYPE":          "member_type",
    "RELATION_WITH_HOF": "relation_with_hof",
    "NAME_EN":           "member_name",
    "FATHER_NAME_EN":    "father_name",
    "MOTHER_NAME_EN":    "mother_name",
    "MARITAL_STATUS":    "marital_status",
    "SPOUCE_NAME_EN":    "spouse_name",
    "DOB":               "date_of_birth",
    "AGE":               "age",
    "GENDER":            "gender",
    "CASTE_CATEGORY":    "caste_category",
    "CASTE":             "caste",
    "BANK":              "bank_name",
    "IFSC_CODE":         "ifsc_code",
    "ACCOUNT_NO":        "bank_account",
    "MOBILE_NO":         "mobile_number",
    "INCOME":            "income",
    "OCCUPATION":        "occupation",
    "MINORITY":          "minority",
    "EDUCATION":         "education",
}

DISTRICT_VALUE_NORMALIZATION = {"Balotara": "Balotra"}

REQUIRED_COLUMNS = {"DISTRICT_NAME_ENG", "ENROLLMENT_ID", "MEMBER_ID", "NAME_EN", "AGE", "GENDER"}


@dataclass(frozen=True)
class DatasetImportReport:
    source_name: str
    rows_loaded: int


def import_excel_dataset(
    source: str | Path | BinaryIO = "dummy_dataset/Dummy_Data_Set.xlsx",
    source_name: str | None = None,
    database_url: str | None = None,
) -> DatasetImportReport:
    # ── 1. Read ──────────────────────────────────────────────────────────────
    if isinstance(source, (str, Path)) and str(source).lower().endswith(".csv"):
        df = pd.read_csv(source)
    else:
        df = pd.read_excel(source)

    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("The provided dataset is empty.")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}.")

    # ── 2. Rename columns ────────────────────────────────────────────────────
    df = df.rename(columns=_COL_MAP)

    # ── 3. Light cleaning ────────────────────────────────────────────────────
    # Normalise district spelling variations
    if "district" in df.columns:
        df["district"] = df["district"].replace(DISTRICT_VALUE_NORMALIZATION)

    # Gender: ensure Title Case
    if "gender" in df.columns:
        df["gender"] = df["gender"].apply(lambda v: str(v).strip().title() if pd.notna(v) else None)

    # is_rural: convert to 0/1 integer; keep NULL as None
    if "is_rural" in df.columns:
        def _to_rural(v):
            if pd.isna(v):
                return None
            try:
                return int(float(v))
            except (ValueError, TypeError):
                return None
        df["is_rural"] = df["is_rural"].apply(_to_rural)

    # age / income / member_id: integer, coerce errors to None
    for int_col in ("age", "income", "member_id"):
        if int_col in df.columns:
            df[int_col] = pd.to_numeric(df[int_col], errors="coerce").astype("Int64")

    # date_of_birth: keep as plain string (avoids timezone drama)
    if "date_of_birth" in df.columns:
        df["date_of_birth"] = df["date_of_birth"].astype(str).where(df["date_of_birth"].notna(), None)

    # Caste: strip leading numbers that creep in (e.g. "58 Jat" → "Jat")
    if "caste" in df.columns:
        import re
        df["caste"] = df["caste"].apply(
            lambda v: re.sub(r"^\d+\s*", "", str(v).strip()).title() if pd.notna(v) else None
        )

    # Keep only columns that exist in the Citizen model
    citizen_cols = {c.name for c in Citizen.__table__.columns if c.name != "id"}
    df = df[[c for c in df.columns if c in citizen_cols]]

    # Replace pandas NA / NaN with None so SQLAlchemy writes NULL
    df = df.where(pd.notna(df), None)

    # ── 4. Write to SQLite ──────────────────────────────────────────────────
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM citizen;"))

    df.to_sql("citizen", engine, if_exists="append", index=False, method="multi", chunksize=500)

    resolved_name = source_name or getattr(source, "name", None) or str(source)
    return DatasetImportReport(
        source_name=Path(str(resolved_name)).name,
        rows_loaded=len(df),
    )
