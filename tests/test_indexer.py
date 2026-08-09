"""Stage 1 verifiable output: attributes for the catalog look sane."""
from __future__ import annotations

from app import db


def test_indexes_all_products(indexed, sqlite_db):
    assert indexed["stats"].indexed == 20  # nothing skipped; all have an image
    with db.get_session() as session:
        assert db.count_products(session) == 20


def test_extracted_attributes_are_reasonable(indexed):
    extractor = indexed["extractor"]
    # Product 1: black leather chelsea boot.
    from tests.conftest import img_bytes

    attrs = extractor.extract_shoe(img_bytes(["shoe", "boot", "leather", "black", "formal"]))
    assert attrs.type.value == "boot"
    assert attrs.material.value == "leather"
    assert "black" in attrs.dominant_colors


def test_out_of_stock_flagged(indexed, sqlite_db):
    with db.get_session() as session:
        oos = session.get(db.Product, "19")
        assert oos is not None and oos.in_stock is False


def test_skip_product_without_image(indexed):
    from app.indexing import Indexer

    indexer = Indexer(
        store=indexed["store"],
        embedder=indexed["embedder"],
        extractor=indexed["extractor"],
    )
    # index_product returns False when there is no image to embed.
    assert (
        indexer.index_product(
            {"product_id": "x", "handle": "x", "title": "x", "images": []}
        )
        is False
    )
