"""Category scoping: the rules that keep a bag from being matched to a boot.

Three promises are under test here:
  * a product with no recognisable product_type is skipped, not guessed at;
  * "similar" only ever returns the uploaded item's own category;
  * "complete the look" returns at most one piece per category.
"""
from __future__ import annotations

import json

import pytest

from app import categories, db, ranker, retrieval
from app.extractor import ClipExtractor
from app.indexing import Indexer
from tests.conftest import FakeHttp, FakeLLM, FakeVectorStore, img_bytes


# ---------------------------------------------------------------------------
# product_type -> category
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "product_type,expected",
    [
        ("Talons", categories.SHOES),
        ("BOTTINES", categories.SHOES),
        ("  Basket  ", categories.SHOES),   # trimmed
        ("SAC", categories.BAGS),
        ("Bagues", categories.JEWELRY),
        ("Boucles d'oreilles", categories.JEWELRY),
        ("Parfum", None),                   # not in the catalog taxonomy
        ("", None),
        (None, None),
    ],
)
def test_category_from_product_type(product_type, expected):
    assert categories.from_product_type(product_type) == expected


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
def _mixed_catalog():
    """One item per category, plus one with an unmappable product_type."""
    def item(pid, title, concepts, category):
        key = f"img-{pid}"
        return (
            {
                "product_id": pid, "handle": title.lower().replace(" ", "-"),
                "title": title, "product_type": "x", "category": category,
                "tags": [], "images": [key], "price": "100.00",
                "variant_id": f"v{pid}", "in_stock": True,
            },
            key,
            img_bytes(concepts),
        )

    return [
        item("s1", "Escarpin Noir", ["shoe", "heel", "black"], categories.SHOES),
        item("b1", "Sac Noir", ["shoe", "black", "leather"], categories.BAGS),
        item("j1", "Bague Or", ["shoe", "gold"], categories.JEWELRY),
        item("u1", "Article Inconnu", ["shoe", "white"], None),
    ]


@pytest.fixture
def mixed(sqlite_db, fake_embedder, tmp_path, monkeypatch):
    monkeypatch.setattr("app.indexing.CHECKPOINT_FILE", str(tmp_path / "ckpt.json"))
    rows = _mixed_catalog()
    store = FakeVectorStore()
    indexer = Indexer(
        store=store,
        embedder=fake_embedder,
        extractor=ClipExtractor(embedder=fake_embedder),
        http=FakeHttp({key: b for _, key, b in rows}),
    )
    stats = indexer.run([p for p, _, _ in rows], resume=False)
    monkeypatch.setattr("app.retrieval.get_store", lambda dim=None: store)
    monkeypatch.setattr("app.retrieval.get_embedder", lambda: fake_embedder)
    return {"store": store, "stats": stats, "embedder": fake_embedder}


def test_uncategorised_product_is_skipped_not_guessed(mixed):
    """A product we cannot categorise must not enter the index at all."""
    with db.get_session() as session:
        stored = {p.product_id: p.category for p in session.query(db.Product).all()}

    assert "u1" not in stored, "an unmappable product_type must not be indexed"
    assert stored == {"s1": categories.SHOES, "b1": categories.BAGS, "j1": categories.JEWELRY}


def test_similar_search_never_leaves_the_category(mixed):
    """A bag query must not return shoes, however similar they look."""
    # These fixtures share concepts on purpose: without scoping, the black
    # shoe and the black bag are near-identical to the embedder.
    query = img_bytes(["shoe", "black", "leather"])
    attrs = ClipExtractor(embedder=mixed["embedder"]).extract_item(query, categories.BAGS)

    results = retrieval.similar_shoes(query, attrs, category=categories.BAGS)

    assert results, "the bag category should still return its own products"
    assert {r.product_id for r in results} == {"b1"}


def test_similar_search_unscoped_would_cross_categories(mixed):
    """Guards the test above: without a category, the mix does leak."""
    query = img_bytes(["shoe", "black", "leather"])
    attrs = ClipExtractor(embedder=mixed["embedder"]).extract_item(query, categories.BAGS)

    results = retrieval.similar_shoes(query, attrs, category=None)

    assert len({r.product_id for r in results}) > 1, (
        "unscoped search should span categories -- otherwise the scoped test proves nothing"
    )


# ---------------------------------------------------------------------------
# One piece per category
# ---------------------------------------------------------------------------
def _candidate(pid, category):
    return {
        "product_id": pid,
        "title": f"Item {pid}",
        "handle": pid,
        "price": "100.00",
        "image_url": "u",
        "variant_id": f"v{pid}",
        "category": category,
        "attributes": {},
    }


def test_ranker_keeps_at_most_one_item_per_category():
    """A model that returns three shoes must still yield one look, not three shoes."""
    from app.models import OutfitAttributes

    candidates = [
        _candidate("s1", categories.SHOES),
        _candidate("s2", categories.SHOES),
        _candidate("s3", categories.SHOES),
        _candidate("b1", categories.BAGS),
    ]
    # Deliberately non-compliant reply: three shoes first, then the bag.
    reply = json.dumps(
        {"ranked": [
            {"product_id": "s1", "rationale": "Jolies chaussures.", "coherence": 0.9},
            {"product_id": "s2", "rationale": "Encore des chaussures.", "coherence": 0.8},
            {"product_id": "s3", "rationale": "Toujours des chaussures.", "coherence": 0.7},
            {"product_id": "b1", "rationale": "Un sac assorti.", "coherence": 0.6},
        ]}
    )

    results = ranker.rank_outfit(
        OutfitAttributes(), candidates, client=FakeLLM([reply])
    )

    picked = [r.product_id for r in results]
    assert picked == ["s1", "b1"], f"expected one per category, got {picked}"


def test_ranker_prompt_states_the_category_of_each_candidate():
    """The model cannot honour one-per-category without knowing the categories."""
    from app.models import OutfitAttributes

    llm = FakeLLM([json.dumps({"ranked": []})])
    ranker.rank_outfit(
        OutfitAttributes(),
        [_candidate("s1", categories.SHOES), _candidate("b1", categories.BAGS)],
        client=llm,
    )

    prompt = json.dumps(llm.calls[0])
    assert categories.SHOES in prompt and categories.BAGS in prompt


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------
def test_full_run_removes_products_that_left_the_catalog(mixed, fake_embedder, tmp_path, monkeypatch):
    """Drafts and de-listed items must disappear, not linger as recommendations.

    run() only upserts, so without a prune step a product switched to draft
    stays searchable forever -- which is exactly how 188 drafts ended up
    recommendable in production.
    """
    monkeypatch.setattr("app.indexing.CHECKPOINT_FILE", str(tmp_path / "ckpt2.json"))
    store = mixed["store"]

    with db.get_session() as session:
        before = {p.product_id for p in session.query(db.Product).all()}
    assert before == {"s1", "b1", "j1"}

    # The feed now carries only the shoe: the bag and the ring were unlisted.
    key = "img-s1"
    feed = [{
        "product_id": "s1", "handle": "escarpin-noir", "title": "Escarpin Noir",
        "product_type": "x", "category": categories.SHOES, "tags": [],
        "images": [key], "price": "100.00", "variant_id": "vs1", "in_stock": True,
    }]
    indexer = Indexer(
        store=store,
        embedder=fake_embedder,
        extractor=ClipExtractor(embedder=fake_embedder),
        http=FakeHttp({key: img_bytes(["shoe", "heel", "black"])}),
    )
    stats = indexer.run(feed, resume=False, prune=True)

    assert stats.pruned == 2
    with db.get_session() as session:
        after = {p.product_id for p in session.query(db.Product).all()}
    assert after == {"s1"}
    assert store.get_first_vector("b1") is None, "vectors must go too, not just rows"


def test_incremental_run_never_prunes(mixed, fake_embedder, tmp_path, monkeypatch):
    """An incremental pass sees a slice of the feed and must not delete."""
    monkeypatch.setattr("app.indexing.CHECKPOINT_FILE", str(tmp_path / "ckpt3.json"))
    indexer = Indexer(
        store=mixed["store"],
        embedder=fake_embedder,
        extractor=ClipExtractor(embedder=fake_embedder),
        http=FakeHttp({}),
    )
    stats = indexer.run([], resume=True, prune=False)

    assert stats.pruned == 0
    with db.get_session() as session:
        assert {p.product_id for p in session.query(db.Product).all()} == {"s1", "b1", "j1"}


def test_sparse_category_still_reaches_the_shortlist(sqlite_db, fake_embedder, monkeypatch):
    """A category the formality/season filter would empty must not vanish.

    The filters were tuned for 320 shoes. Against 11 handbags they routinely
    match nothing, and a shortlist with no bag in it means "compléter le look"
    quietly returns shoes only -- no error, no signal, feature just gone.
    """
    from app.models import OutfitAttributes

    with db.get_session() as session:
        # Shoes sit inside the window; the only bag sits well outside it.
        db.upsert_product(
            session, product_id="s1", handle="s1", title="Escarpin", price="100",
            product_type="Talons", category=categories.SHOES, variant_id="v1",
            image_url="u", in_stock=True, shoe_type="heel", formality=3,
            season="all-season", attributes={},
        )
        db.upsert_product(
            session, product_id="b1", handle="b1", title="Pochette", price="200",
            product_type="SAC", category=categories.BAGS, variant_id="v2",
            image_url="u", in_stock=True, shoe_type="clutch", formality=5,
            season="winter", attributes={},
        )
        session.commit()

    store = FakeVectorStore()
    monkeypatch.setattr("app.retrieval.get_store", lambda dim=None: store)
    monkeypatch.setattr("app.retrieval.get_embedder", lambda: fake_embedder)

    outfit = OutfitAttributes(formality=3)
    got = retrieval.outfit_candidates(
        outfit, img_bytes(["outfit"]), image_vec=fake_embedder.embed_texts(["casual"])[0]
    )

    by_cat = {c["category"] for c in got}
    assert categories.BAGS in by_cat, (
        f"the bag was filtered out and never reached the ranker: {by_cat}"
    )
    assert categories.SHOES in by_cat
