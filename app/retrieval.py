"""Retrieval for both modes.

Mode B (similar shoes): pure vector search, no LLM. Fast and free.
Mode A (complete the outfit): attribute filter -> candidate pool -> CLIP-embedding
diversity pruning -> hand off to the ranker.
"""
from __future__ import annotations

import numpy as np
from sqlalchemy import select

from app import db
from app.clip_model import Embedder, get_embedder
from app.config import settings
from app.models import OutfitAttributes, ProductResult, ShoeAttributes
from app.qdrant_store import VectorStore, get_store


# ---------------------------------------------------------------------------
# Mode B — visually similar shoes
# ---------------------------------------------------------------------------
def similar_shoes(
    image_bytes: bytes,
    query_attrs: ShoeAttributes,
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
) -> list[ProductResult]:
    store = store or get_store()
    embedder = embedder or get_embedder()
    query_vec = embedder.embed_image(image_bytes)

    # Over-fetch so the same-type boost can reorder without starving the top-12.
    hits = store.search(query_vec, limit=settings.mode_b_top_k * 2, in_stock_only=True)

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
        stmt = (
            select(db.Product)
            .where(db.Product.in_stock.is_(True))
            .where(db.Product.formality >= lo)
            .where(db.Product.formality <= hi)
        )
        if compatible_seasons:
            stmt = stmt.where(db.Product.season.in_(compatible_seasons))
        rows = session.scalars(stmt).all()
        # Detach the fields we need before the session closes.
        candidates = [_row_to_candidate(r) for r in rows]

    if price_band:
        low, high = price_band
        candidates = [c for c in candidates if _price_in(c["price"], low, high)]

    # Rank the pool by visual affinity to the outfit, then prune near-duplicates.
    outfit_vec = embedder.embed_image(image_bytes)
    candidates = _rank_by_vector(candidates, outfit_vec, store)
    candidates = _diversify(candidates, store)
    return candidates[: settings.mode_a_candidate_pool]


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
