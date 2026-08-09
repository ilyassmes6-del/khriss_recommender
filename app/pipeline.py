"""Request-time orchestration: route -> extract -> retrieve -> (rank).

Holds the warm singletons (embedder, router, extractor, vector store) so a
request never pays model-load cost. `warm_up()` is called once at startup.
"""
from __future__ import annotations

from app import ranker, retrieval
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


def recommend(image_bytes: bytes, llm_client=None) -> RecommendResponse:
    route = _get_router().route(image_bytes)
    extractor = get_extractor()

    if route.mode == "shoe":
        attrs = extractor.extract_shoe(image_bytes)
        results = retrieval.similar_shoes(image_bytes, attrs)
        return RecommendResponse(
            mode="shoe",
            confidence=route.confidence,
            query_attributes=attrs.model_dump(mode="json"),
            results=results,
        )

    if route.mode == "outfit":
        attrs = extractor.extract_outfit(image_bytes)
        candidates = retrieval.outfit_candidates(attrs, image_bytes)
        results = ranker.rank_outfit(attrs, candidates, client=llm_client)
        return RecommendResponse(
            mode="outfit",
            confidence=route.confidence,
            query_attributes=attrs.model_dump(mode="json"),
            results=results,
        )

    # Ambiguous -> run both paths, storefront renders tabs.
    shoe_attrs = extractor.extract_shoe(image_bytes)
    outfit_attrs = extractor.extract_outfit(image_bytes)
    shoe_results = retrieval.similar_shoes(image_bytes, shoe_attrs)
    candidates = retrieval.outfit_candidates(outfit_attrs, image_bytes)
    outfit_results = ranker.rank_outfit(outfit_attrs, candidates, client=llm_client)
    return RecommendResponse(
        mode="both",
        confidence=route.confidence,
        query_attributes=outfit_attrs.model_dump(mode="json"),
        results=outfit_results,
        shoe_results=shoe_results,
        outfit_results=outfit_results,
    )
