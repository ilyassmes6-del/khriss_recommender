"""Mode B (similar shoes) and Mode A (candidate filtering/diversity) tests."""
from __future__ import annotations

from app import retrieval
from app.config import settings
from app.models import OutfitAttributes
from tests.conftest import img_bytes


# ---------------------------------------------------------------------------
# Mode B
# ---------------------------------------------------------------------------
def test_similar_shoes_returns_same_type_first(indexed):
    extractor = indexed["extractor"]
    query = img_bytes(["shoe", "boot", "leather", "black"])
    attrs = extractor.extract_shoe(query)
    results = retrieval.similar_shoes(query, attrs)

    assert results, "expected some matches"
    assert len(results) <= settings.mode_b_top_k
    # The closest match to a black leather boot should itself be a boot.
    top = results[0]
    assert "Boot" in top.title


def test_similar_shoes_excludes_out_of_stock(indexed):
    extractor = indexed["extractor"]
    query = img_bytes(["shoe", "sneaker", "canvas", "white"])
    attrs = extractor.extract_shoe(query)
    results = retrieval.similar_shoes(query, attrs)
    ids = {r.product_id for r in results}
    assert "19" not in ids  # the OOS white canvas sneaker


def test_similar_shoes_dedupes_by_product(indexed):
    # Every product contributes exactly one result even though the store may
    # hold several vectors per product.
    extractor = indexed["extractor"]
    query = img_bytes(["shoe", "sneaker", "white"])
    attrs = extractor.extract_shoe(query)
    results = retrieval.similar_shoes(query, attrs)
    ids = [r.product_id for r in results]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Mode A candidate filtering
# ---------------------------------------------------------------------------
def test_outfit_candidates_respect_formality(indexed):
    # A formal outfit should not surface very-casual sandals (formality ~1-2).
    outfit = OutfitAttributes(formality=4)
    query = img_bytes(["outfit", "person", "formal", "elegant"])
    candidates = retrieval.outfit_candidates(outfit, query)
    formalities = [c["attributes"].get("formality") for c in candidates]
    assert all(abs(f - 4) <= settings.mode_a_formality_tolerance for f in formalities)


def test_outfit_candidates_price_band(indexed):
    outfit = OutfitAttributes(formality=3)
    query = img_bytes(["outfit", "person", "casual"])
    candidates = retrieval.outfit_candidates(outfit, query, price_band=(50.0, 100.0))
    for c in candidates:
        assert 50.0 <= float(c["price"]) <= 100.0
