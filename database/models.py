from __future__ import annotations

from sqlalchemy import Boolean, Date, Float, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Citizen(Base):
    """Single flat table — mirrors Dummy_Data_Set.xlsx one-to-one."""
    __tablename__ = "citizen"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Geography
    district:        Mapped[str | None] = mapped_column(String(80),  index=True)
    is_rural:        Mapped[int | None] = mapped_column(Integer,     index=True)   # 1=rural, 0=urban
    block:           Mapped[str | None] = mapped_column(String(80),  index=True)
    city:            Mapped[str | None] = mapped_column(String(80),  index=True)
    ward:            Mapped[str | None] = mapped_column(String(40),  index=True)
    gram_panchayat:  Mapped[str | None] = mapped_column(String(100), index=True)
    village:         Mapped[str | None] = mapped_column(String(100), index=True)

    # Identity
    enrollment_id:          Mapped[str | None] = mapped_column(String(40), index=True)
    member_id:              Mapped[int | None] = mapped_column(Integer,    index=True)
    jan_aadhaar_member_id:  Mapped[str | None] = mapped_column(String(60), index=True)

    # Demographics
    member_type:      Mapped[str | None] = mapped_column(String(20), index=True)
    relation_with_hof: Mapped[str | None] = mapped_column(String(40), index=True)
    member_name:      Mapped[str | None] = mapped_column(String(120), index=True)
    father_name:      Mapped[str | None] = mapped_column(String(120))
    mother_name:      Mapped[str | None] = mapped_column(String(120))
    marital_status:   Mapped[str | None] = mapped_column(String(32), index=True)
    spouse_name:      Mapped[str | None] = mapped_column(String(120))
    date_of_birth:    Mapped[str | None] = mapped_column(String(20))
    age:              Mapped[int | None] = mapped_column(Integer,     index=True)
    gender:           Mapped[str | None] = mapped_column(String(16),  index=True)
    caste_category:   Mapped[str | None] = mapped_column(String(32),  index=True)
    caste:            Mapped[str | None] = mapped_column(String(180), index=True)

    # Bank
    bank_name:    Mapped[str | None] = mapped_column(String(120), index=True)
    ifsc_code:    Mapped[str | None] = mapped_column(String(16))
    bank_account: Mapped[str | None] = mapped_column(String(32),  index=True)

    # Contact / social
    mobile_number: Mapped[str | None] = mapped_column(String(16))
    income:        Mapped[int | None] = mapped_column(Integer,    index=True)
    occupation:    Mapped[str | None] = mapped_column(String(80), index=True)
    minority:      Mapped[str | None] = mapped_column(String(40), index=True)
    education:     Mapped[str | None] = mapped_column(String(80), index=True)


# Composite indexes for the most common filter patterns
Index("ix_citizen_gender_age",          Citizen.gender, Citizen.age)
Index("ix_citizen_district_gender",     Citizen.district, Citizen.gender)
Index("ix_citizen_caste_category_age",  Citizen.caste_category, Citizen.age)
