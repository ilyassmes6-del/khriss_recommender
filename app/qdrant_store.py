"""Qdrant vector store.

One vector per product image (a shoe shot from three angles => three vectors),
each pointing back to the same product_id via its payload. Query-time dedupe
keeps the best-scoring vector per product_id.

Vectors are L2-normalised and stored under cosine distance, so the score Qdrant
returns is cosine similarity in [-1, 1] (typically [0, 1] for real images).
"""
from __future__ import annotations

import uuid
from typing import Iterable, Optional

import numpy as np

from app.config import settings

# qdrant-client is a real dependency in production, but the offline test suite
# swaps in a fake store and never touches it. Import lazily so the module (and
# everything that imports it) loads without the package present.
try:  # pragma: no cover - import guard
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qm
except ImportError:  # pragma: no cover
    QdrantClient = None  # type: ignore
    qm = None  # type: ignore


class VectorStore:
    def __init__(self, client: Optional[QdrantClient] = None, dim: int = 512):
        self.collection = settings.qdrant_collection
        self.dim = dim
        self.client = client or QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )

    def ensure_collection(self, dim: Optional[int] = None) -> None:
        dim = dim or self.dim
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            )
            # Payload indexes for filtered search.
            for field, schema in [
                ("in_stock", qm.PayloadSchemaType.BOOL),
                ("type", qm.PayloadSchemaType.KEYWORD),
                ("product_id", qm.PayloadSchemaType.KEYWORD),
            ]:
                self.client.create_payload_index(self.collection, field, schema)

    @staticmethod
    def _point_id(product_id: str, image_index: int) -> str:
        # Deterministic per (product, image) so re-indexing overwrites in place.
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{product_id}:{image_index}"))

    def upsert_product_vectors(
        self, product_id: str, vectors: list[np.ndarray], payload: dict
    ) -> None:
        points = []
        for i, vec in enumerate(vectors):
            body = dict(payload)
            body["product_id"] = product_id
            points.append(
                qm.PointStruct(
                    id=self._point_id(product_id, i),
                    vector=vec.astype("float32").tolist(),
                    payload=body,
                )
            )
        if points:
            self.client.upsert(self.collection, points=points, wait=True)

    def delete_product(self, product_id: str) -> None:
        self.client.delete(
            self.collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="product_id", match=qm.MatchValue(value=product_id)
                        )
                    ]
                )
            ),
            wait=True,
        )

    def search(
        self,
        query: np.ndarray,
        limit: int,
        in_stock_only: bool = True,
        over_fetch: int = 4,
    ) -> list[tuple[str, float, dict]]:
        """Return (product_id, score, payload), deduped to best vector/product."""
        flt = None
        if in_stock_only:
            flt = qm.Filter(
                must=[
                    qm.FieldCondition(key="in_stock", match=qm.MatchValue(value=True))
                ]
            )
        # Over-fetch because several vectors may map to one product.
        hits = self.client.search(
            self.collection,
            query_vector=query.astype("float32").tolist(),
            limit=limit * over_fetch,
            query_filter=flt,
            with_payload=True,
        )
        best: dict[str, tuple[float, dict]] = {}
        for h in hits:
            pid = h.payload["product_id"]
            if pid not in best or h.score > best[pid][0]:
                best[pid] = (h.score, h.payload)
        ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)
        return [(pid, score, payload) for pid, (score, payload) in ranked[:limit]]

    def get_first_vector(self, product_id: str) -> Optional[np.ndarray]:
        """Return one stored image vector for a product (used for diversity math)."""
        try:
            recs = self.client.retrieve(
                self.collection,
                ids=[self._point_id(product_id, 0)],
                with_vectors=True,
            )
        except Exception:
            return None
        if not recs or recs[0].vector is None:
            return None
        return np.asarray(recs[0].vector, dtype="float32")

    def count(self) -> int:
        try:
            return self.client.count(self.collection, exact=True).count
        except Exception:
            return 0


_store: Optional[VectorStore] = None


def get_store(dim: int = 512) -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore(dim=dim)
    return _store
