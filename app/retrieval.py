"""Retrieval for both modes.

Mode B (similar shoes): pure vector search, no LLM. Fast and free.
Mode A (complete the outfit): attribute filter -> candidate pool -> CLIP-embedding
diversity pruning -> hand off to the ranker.
"""
from __future__ import annotations

import numpy as np
from sqlalchemy import select

from app import categories, db
from app.clip_model import Embedder, get_embedder
from app.config import settings
from app.models import ItemAttributes, OutfitAttributes, ProductResult
from app.qdrant_store import VectorStore, get_store


# ---------------------------------------------------------------------------
# Mode B — visually similar shoes
# ---------------------------------------------------------------------------
def similar_shoes(
    image_bytes: bytes,
    query_attrs: ItemAttributes,
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
    query_vec=None,
    category: str | None = None,
) -> list[ProductResult]:
    store = store or get_store()
    embedder = embedder or get_embedder()
    if query_vec is None:
        query_vec = embedder.embed_image(image_bytes)

    # Scope to the uploaded item's own category. Cross-category cosine is
    # dominated by colour -- a black clutch and a black boot look "similar" to
    # CLIP -- so an unscoped search returns a grid nobody asked for.
    hits = store.search(
        query_vec,
        limit=settings.mode_b_top_k * 2,
        in_stock_only=True,
        category=category,
    )

    query_type = query_attrs.type_value()
    scored: list[tuple[float, str, dict]] = []
    for pid, score, payload in hits:
        boosted = score
        if query_type and payload.get("type") == query_type:
            boosted += settings.same_type_boost
        scored.append((boosted, pid, payload))

    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[: settings.mode_b_top_k]
    return _payloads_to_results([(pid, s, p) for s, pid, p in top])


# ---------------------------------------------------------------------------
# Mode A — shoes that complete an outfit
# ---------------------------------------------------------------------------
def outfit_candidates(
    outfit: OutfitAttributes,
    image_bytes: bytes,
    price_band: tuple[float, float] | None = None,
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
    image_vec=None,
) -> list[dict]:
    """Filtered, diversity-pruned shortlist for the ranker.

    Returns a list of candidate dicts carrying the product record + attributes.
    """
    store = store or get_store()
    embedder = embedder or get_embedder()

    tol = settings.mode_a_formality_tolerance
    lo, hi = outfit.formality - tol, outfit.formality + tol
    compatible_seasons = _season_matches(outfit.season.value if outfit.season else None)

    with db.get_session() as session:
        rows = []
        for cat in categories.CATEGORIES:
            rows.extend(_category_rows(session, cat, lo, hi, compatible_seasons))
        # Detach the fields we need before the session closes.
        candidates = [_row_to_candidate(r) for r in rows]

    if price_band:
        low, high = price_band
        candidates = [c for c in candidates if _price_in(c["price"], low, high)]

    # Rank the pool by visual affinity to the outfit.
    outfit_vec = image_vec if image_vec is not None else embedder.embed_image(image_bytes)
    candidates = _rank_by_vector(candidates, outfit_vec, store)

    # Balance the shortlist across categories, and prune near-duplicates *within*
    # each category. A flat top-N would be all shoes (the catalog is ~320 shoes
    # to 11 bags), and a global diversity pass lets shoes spend the budget before
    # bags and jewellery are considered -- so both are done per category here,
    # giving the ranker a few genuinely distinct options for every slot.
    per_category = max(1, settings.mode_a_candidate_pool // len(categories.CATEGORIES))
    picked: list[dict] = []
    for cat in categories.CATEGORIES:
        in_cat = [c for c in candidates if c.get("category") == cat]
        in_cat = _diversify(in_cat, store)
        picked.extend(in_cat[:per_category])
    return picked


def _category_rows(session, category: str, lo: int, hi: int, seasons):
    """In-stock products for one category, relaxing filters if they empty it.

    The formality window and season match were tuned against a catalog of 320
    shoes. Applied to 11 handbags they routinely match nothing, and the
    balanced shortlist below would then silently contain no bag at all --
    "compléter le look" degrading back to three pairs of shoes with no error
    anywhere. Narrow first, widen only when a category would otherwise be
    unrepresented.
    """
    base = select(db.Product).where(
        db.Product.in_stock.is_(True), db.Product.category == category
    )

    stmt = base.where(db.Product.formality >= lo, db.Product.formality <= hi)
    if seasons:
        stmt = stmt.where(db.Product.season.in_(seasons))
    rows = session.scalars(stmt).all()
    if rows:
        return rows

    # Drop the season constraint before the formality one: wearing a summer bag
    # in autumn reads as a smaller mistake than a black-tie clutch with jeans.
    rows = session.scalars(
        base.where(db.Product.formality >= lo, db.Product.formality <= hi)
    ).all()
    if rows:
        return rows

    return session.scalars(base).all()


def _rank_by_vector(
    candidates: list[dict], outfit_vec: np.ndarray, store: VectorStore
) -> list[dict]:
    """Order candidates by how well their image vector fits the outfit vector."""
    for c in candidates:
        vec = _product_vector(store, c["product_id"])
        c["_vec"] = vec
        c["affinity"] = float(vec @ outfit_vec) if vec is not None else -1.0
    candidates.sort(key=lambda c: c["affinity"], reverse=True)
    return candidates


def _diversify(candidates: list[dict], store: VectorStore) -> list[dict]:
    """Greedily drop candidates too similar to an already-kept one."""
    kept: list[dict] = []
    kept_vecs: list[np.ndarray] = []
    thr = settings.mode_a_diversity_threshold
    for c in candidates:
        vec = c.get("_vec")
        if vec is None:
            kept.append(c)
            continue
        if all(float(vec @ kv) < thr for kv in kept_vecs):
            kept.append(c)
            kept_vecs.append(vec)
    return kept


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _product_vector(store: VectorStore, product_id: str) -> np.ndarray | None:
    """Fetch one stored image vector for a product (for diversity math)."""
    return store.get_first_vector(product_id)


def _row_to_candidate(r: db.Product) -> dict:
    return {
        "product_id": r.product_id,
        "title": r.title,
        "handle": r.handle,
        "price": r.price,
        "image_url": r.image_url,
        "variant_id": r.variant_id,
        "category": r.category,
        "attributes": r.attributes or {},
    }


def _payloads_to_results(hits: list[tuple[str, float, dict]]) -> list[ProductResult]:
    ids = [pid for pid, _, _ in hits]
    with db.get_session() as session:
        rows = db.get_products_by_ids(session, ids)
        out = []
        for pid, score, _payload in hits:
            r = rows.get(pid)
            if r is None:
                continue
            out.append(
                ProductResult(
                    product_id=pid,
                    title=r.title,
                    handle=r.handle,
                    price=r.price,
                    image_url=r.image_url,
                    variant_id=r.variant_id,
                    category=r.category,
                    score=round(float(score), 4),
                )
            )
    return out


def _season_matches(season: str | None) -> list[str] | None:
    if not season or season == "all-season":
        return None  # no season constraint
    # A shoe tagged all-season is compatible with any outfit season.
    return [season, "all-season"]


def _price_in(price: str | None, low: float, high: float) -> bool:
    try:
        p = float(price)
    except (TypeError, ValueError):
        return True  # don't drop unknown-price products
    return low <= p <= high
