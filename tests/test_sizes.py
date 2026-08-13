"""Shoe sizing: the shopper picks a size, and only wearable shoes come back.

Four promises are under test here:
  * a shoe that does not stock the chosen size is never shown, and no relaxation
    pass can bring it back;
  * add-to-cart resolves to the variant for that size, not the product default;
  * bags and jewellery are untouched -- picking a shoe size must not shrink them;
  * the size map is read off Shopify's variant options, not guessed.
"""
from __future__ import annotations

import json

from app import categories, db, ranker, retrieval
from app.models import OutfitAttributes
from app.shopify_client import normalize_product
from tests.conftest import FakeLLM, img_bytes


# ---------------------------------------------------------------------------
# Reading sizes off Shopify
# ---------------------------------------------------------------------------
def _node(pid, options, variants):
    return {
        "id": f"gid://shopify/Product/{pid}",
        "handle": f"p{pid}",
        "title": f"Product {pid}",
        "productType": "Talons",
        "tags": [],
        "images": {"nodes": [{"url": "http://img/1.jpg"}]},
        "options": options,
        "variants": {"nodes": variants},
    }


def _variant(vid, size, available):
    return {
        "id": f"gid://shopify/ProductVariant/{vid}",
        "price": "429.00",
        "inventoryQuantity": 3 if available else 0,
        "availableForSale": available,
        "selectedOptions": [{"name": "Taille", "value": size}],
    }


def test_sizes_are_read_from_the_taille_option():
    """The live catalog names the axis "Taille" with values 36-41."""
    node = _node(
        "1",
        [{"name": "Taille", "values": ["36", "37"]}],
        [_variant("v36", "36", True), _variant("v37", "37", False)],
    )
    p = normalize_product(node)
    assert p["sizes"] == {
        "36": {"variant_id": "v36", "available": True},
        "37": {"variant_id": "v37", "available": False},
    }


def test_a_product_with_no_size_option_maps_to_no_sizes():
    """Bags come back as a single Default Title variant -- sizeless, not size 'Default Title'."""
    node = _node(
        "2",
        [{"name": "Title", "values": ["Default Title"]}],
        [
            {
                "id": "gid://shopify/ProductVariant/v1",
                "price": "349.00",
                "inventoryQuantity": 5,
                "availableForSale": True,
                "selectedOptions": [{"name": "Title", "value": "Default Title"}],
            }
        ],
    )
    assert normalize_product(node)["sizes"] == {}


# ---------------------------------------------------------------------------
# Resolving the variant for a size
# ---------------------------------------------------------------------------
def _sized(**over):
    rec = {
        "product_id": "s1",
        "variant_id": "default-variant",
        "category": categories.SHOES,
        "sizes": {
            "38": {"variant_id": "v38", "available": True},
            "41": {"variant_id": "v41", "available": False},
        },
    }
    rec.update(over)
    return rec


def test_variant_resolves_to_the_chosen_size():
    """The stored variant_id is whichever was available at index time; a shopper
    who picked 38 must not get that one in their cart."""
    assert retrieval.variant_for_size(_sized(), "38") == ("v38", "38")


def test_sold_out_size_yields_no_variant_rather_than_the_wrong_one():
    assert retrieval.variant_for_size(_sized(), "41") == (None, None)


def test_sizeless_product_keeps_its_only_variant():
    bag = {"variant_id": "bag-v1", "category": categories.BAGS, "sizes": {}}
    assert retrieval.variant_for_size(bag, "38") == ("bag-v1", None)


def test_no_size_chosen_keeps_the_default_variant():
    assert retrieval.variant_for_size(_sized(), None) == ("default-variant", None)


_IMG = img_bytes(["shoe"])


# ---------------------------------------------------------------------------
# Filtering, including the relaxation ladder
# ---------------------------------------------------------------------------
def _row(session, pid, category, sizes, formality=3, season="all-season"):
    db.upsert_product(
        session,
        product_id=pid,
        handle=pid,
        title=f"Item {pid}",
        price="100.00",
        category=category,
        variant_id=f"{pid}-default",
        image_url="u",
        in_stock=True,
        sizes=sizes,
        shoe_type="heel",
        formality=formality,
        season=season,
        attributes={},
    )


def test_size_filter_survives_the_relaxation_ladder(sqlite_db):
    """_category_rows widens formality/season when a category comes back empty.

    Size must not widen with them: a shoe that does not come in the shopper's
    size is unwearable, not merely a worse match, and surfacing it on the
    fallback pass would turn the whole feature backwards.
    """
    with db.get_session() as session:
        # Formality 5 and winter -- outside the window the query will ask for,
        # so every relaxation step gets exercised.
        _row(session, "wrong-size", categories.SHOES,
             {"41": {"variant_id": "v41", "available": True}},
             formality=5, season="winter")
        session.commit()

        rows = retrieval._category_rows(
            session, categories.SHOES, lo=2, hi=4, seasons=["summer"], size="38"
        )

    assert rows == [], "a shoe with no size 38 came back from a relaxed pass"


def test_size_filter_keeps_the_matching_shoe(sqlite_db):
    with db.get_session() as session:
        _row(session, "right-size", categories.SHOES,
             {"38": {"variant_id": "v38", "available": True}})
        session.commit()
        rows = retrieval._category_rows(
            session, categories.SHOES, lo=2, hi=4, seasons=None, size="38"
        )
    assert [r.product_id for r in rows] == ["right-size"]


def test_sold_out_size_is_filtered_even_though_sql_matched(sqlite_db, fake_embedder,
                                                           monkeypatch):
    """The SQL prefilter is a LIKE over serialized JSON, so it matches a size
    that is present but sold out; the parsed re-check is what drops it."""
    from tests.conftest import FakeVectorStore

    store = FakeVectorStore()
    monkeypatch.setattr("app.retrieval.get_store", lambda dim=512: store)
    monkeypatch.setattr("app.retrieval.get_embedder", lambda: fake_embedder)

    with db.get_session() as session:
        _row(session, "oos-38", categories.SHOES,
             {"38": {"variant_id": "v38", "available": False}})
        session.commit()

    picked = retrieval.outfit_candidates(
        OutfitAttributes(formality=3), _IMG,
        image_vec=fake_embedder.embed_image(_IMG), size="38",
    )
    assert [c["product_id"] for c in picked] == []


def test_bags_and_jewelry_ignore_the_shoe_size(sqlite_db, fake_embedder, monkeypatch):
    """Picking a shoe size must not shrink the sizeless half of the catalog."""
    from tests.conftest import FakeVectorStore

    store = FakeVectorStore()
    monkeypatch.setattr("app.retrieval.get_store", lambda dim=512: store)
    monkeypatch.setattr("app.retrieval.get_embedder", lambda: fake_embedder)

    with db.get_session() as session:
        _row(session, "bag1", categories.BAGS, {})
        _row(session, "jewel1", categories.JEWELRY, {})
        _row(session, "shoe-38", categories.SHOES,
             {"38": {"variant_id": "v38", "available": True}})
        _row(session, "shoe-41", categories.SHOES,
             {"41": {"variant_id": "v41", "available": True}})
        session.commit()

    picked = retrieval.outfit_candidates(
        OutfitAttributes(formality=3), _IMG,
        image_vec=fake_embedder.embed_image(_IMG), size="38",
    )
    got = {c["product_id"] for c in picked}
    assert "bag1" in got and "jewel1" in got, "sizeless pieces were filtered by size"
    assert "shoe-38" in got
    assert "shoe-41" not in got


# ---------------------------------------------------------------------------
# The ranker hands back the size-resolved variant
# ---------------------------------------------------------------------------
def test_ranker_returns_the_variant_for_the_chosen_size():
    candidates = [
        {
            "product_id": "s1", "title": "Talons", "handle": "talons",
            "price": "429.00", "image_url": "u", "variant_id": "default-variant",
            "category": categories.SHOES,
            "sizes": {
                "38": {"variant_id": "v38", "available": True},
                "39": {"variant_id": "v39", "available": True},
            },
            "attributes": {},
        }
    ]
    reply = json.dumps(
        {"ranked": [{"product_id": "s1", "rationale": "Jolies.", "coherence": 0.9}]}
    )

    r38 = ranker.rank_outfit(
        OutfitAttributes(), candidates, client=FakeLLM([reply]), size="38"
    )
    r39 = ranker.rank_outfit(
        OutfitAttributes(), candidates, client=FakeLLM([reply]), size="39"
    )

    assert r38[0].variant_id == "v38" and r38[0].size == "38"
    assert r39[0].variant_id == "v39" and r39[0].size == "39"
    assert r38[0].variant_id != r39[0].variant_id, "same variant for two sizes"
