"""Postgres persistence for product rows + extracted shoe attributes.

Qdrant holds the vectors (one per image); Postgres holds the human-readable
product record and its attributes, keyed by product_id. Mode A retrieval reads
from Postgres for filtering; the ranker reads titles/prices from here.
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Integer,
    String,
    Text,
    create_engine,
    inspect,
    select,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    pass


class Product(Base):
    __tablename__ = "products"

    product_id: Mapped[str] = mapped_column(String, primary_key=True)
    handle: Mapped[str] = mapped_column(String, index=True)
    title: Mapped[str] = mapped_column(Text)
    price: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    product_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    # shoes | bags | jewelry, from app.categories. Indexed: every retrieval
    # path filters on it so categories never get scored against each other.
    category: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True)
    variant_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    image_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    in_stock: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    # {"38": {"variant_id": "...", "available": true}, ...} for products sold by
    # size; {} for bags and jewellery. Carries the per-size variant so add-to-cart
    # adds the size the shopper actually chose, not the product's default variant.
    sizes: Mapped[dict] = mapped_column(JSON, default=dict)

    # Attribute axes we filter on, denormalised for fast Mode A queries.
    shoe_type: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True)
    formality: Mapped[int] = mapped_column(Integer, default=3, index=True)
    season: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True)

    # Full ShoeAttributes dump for the ranker + presentation.
    attributes: Mapped[dict] = mapped_column(JSON, default=dict)


_engine = None
_SessionLocal: Optional[sessionmaker] = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)
    return _engine


def get_session() -> Session:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), future=True)
    return _SessionLocal()


def init_db() -> None:
    Base.metadata.create_all(get_engine())
    _ensure_columns()


# Columns added after the table first shipped, as (name, DDL type, index SQL).
# create_all() creates missing tables but never alters existing ones, so a
# database populated before one of these existed would fail every query against
# it. Still not enough surface to justify Alembic -- but enough that each new
# column is a row here rather than another bespoke function.
_ADDED_COLUMNS: list[tuple[str, str, Optional[str]]] = [
    (
        "category",
        "VARCHAR",
        "CREATE INDEX IF NOT EXISTS ix_products_category ON products (category)",
    ),
    # Per-size variants; see Product.sizes. No index: it is read per candidate
    # row after the SQL filter, never selected on.
    ("sizes", "JSON", None),
]


def _ensure_columns() -> None:
    engine = get_engine()
    insp = inspect(engine)
    if "products" not in insp.get_table_names():
        return  # create_all just made it, with every column
    existing = {c["name"] for c in insp.get_columns("products")}
    for name, ddl_type, index_sql in _ADDED_COLUMNS:
        if name in existing:
            continue
        with engine.begin() as conn:
            conn.execute(text(f"ALTER TABLE products ADD COLUMN {name} {ddl_type}"))
            if index_sql:
                conn.execute(text(index_sql))


def upsert_product(session: Session, **fields) -> None:
    obj = session.get(Product, fields["product_id"])
    if obj is None:
        obj = Product(**fields)
        session.add(obj)
    else:
        for k, v in fields.items():
            setattr(obj, k, v)


def get_products_by_ids(session: Session, ids: list[str]) -> dict[str, Product]:
    if not ids:
        return {}
    rows = session.scalars(select(Product).where(Product.product_id.in_(ids))).all()
    return {p.product_id: p for p in rows}


def count_products(session: Session) -> int:
    from sqlalchemy import func

    return session.scalar(select(func.count()).select_from(Product)) or 0
