"""Request-time orchestration: route -> extract -> retrieve -> (rank).

Holds the warm singletons (embedder, router, extractor, vector store) so a
request never pays model-load cost. `warm_up()` is called once at startup.
"""
from __future__ import annotations

from app import categories, ranker, retrieval
from app.clip_model import get_embedder
from app.extractor import get_extractor
from app.models import RecommendResponse
from app.qdrant_store import get_store
from app.router import ModeRouter

_router: ModeRouter | None = None


def warm_up() -> None:
    """Load CLIP + build router/extractor label banks once."""
    global _router
    get_embedder()  # triggers model load
    get_extractor()  # builds CLIP label banks (or OpenRouter client)
    _router = ModeRouter()
    get_store(dim=get_embedder().dim)


def _get_router() -> ModeRouter:
    global _router
    if _router is None:
        _router = ModeRouter()
    return _router


def recommend(
    image_bytes: bytes, llm_client=None, size: str | None = None
) -> RecommendResponse:
    # One CLIP pass for the whole request. Routing, extraction and retrieval all
    # score against the *same* image, so embedding per step meant up to five
    # identical forward passes -- which on a shared vCPU was the entire wait.
    # Borrow the router's embedder rather than the module singleton: it is the
    # same object in production, and the one the tests already wire up.
    router = _get_router()
    vec = router.embedder.embed_image(image_bytes)

    route = router.route(image_bytes, vec=vec)
    extractor = get_extractor()

    if route.mode == "shoe":
        # "shoe" means "a single item" -- the router says which category it is,
        # and both the vocabulary and the searched slice follow from that.
        category = route.category or categories.SHOES
        attrs = extractor.extract_item(image_bytes, category, vec=vec)
        results = retrieval.similar_shoes(
            image_bytes, attrs, query_vec=vec, category=category, size=size
        )
        return RecommendResponse(
            mode="shoe",
            confidence=route.confidence,
            query_attributes=attrs.model_dump(mode="json"),
            results=results,
            size=size,
        )

    if route.mode == "outfit":
        attrs = extractor.extract_outfit(image_bytes, vec=vec)
        candidates = retrieval.outfit_candidates(
            attrs, image_bytes, image_vec=vec, size=size
        )
        results = ranker.rank_outfit(attrs, candidates, client=llm_client, size=size)
        return RecommendResponse(
            mode="outfit",
            confidence=route.confidence,
            query_attributes=attrs.model_dump(mode="json"),
            results=results,
            size=size,
        )

    # Ambiguous -> run both paths, storefront renders tabs.
    category = route.category or categories.SHOES
    shoe_attrs = extractor.extract_item(image_bytes, category, vec=vec)
    outfit_attrs = extractor.extract_outfit(image_bytes, vec=vec)
    shoe_results = retrieval.similar_shoes(
        image_bytes, shoe_attrs, query_vec=vec, category=category, size=size
    )
    candidates = retrieval.outfit_candidates(
        outfit_attrs, image_bytes, image_vec=vec, size=size
    )
    outfit_results = ranker.rank_outfit(
        outfit_attrs, candidates, client=llm_client, size=size
    )
    return RecommendResponse(
        mode="both",
        confidence=route.confidence,
        query_attributes=outfit_attrs.model_dump(mode="json"),
        results=outfit_results,
        shoe_results=shoe_results,
        outfit_results=outfit_results,
        size=size,
    )
