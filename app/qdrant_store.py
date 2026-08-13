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

        # Payload indexes are ensured on every call, not just at creation: a
        # collection built before a field existed would otherwise never get its
        # index, and the filter would fall back to a full scan on exactly the
        # deployment that already has data. Re-creating an existing index is a
        # no-op, so this is safe to repeat.
        for field, schema in [
            ("in_stock", qm.PayloadSchemaType.BOOL),
            ("type", qm.PayloadSchemaType.KEYWORD),
            ("product_id", qm.PayloadSchemaType.KEYWORD),
            ("category", qm.PayloadSchemaType.KEYWORD),
            # List of sizes this product currently has in stock. A keyword index
            # over an array matches if *any* element matches, which is exactly
            # "does this shoe come in the shopper's size".
            ("sizes_in_stock", qm.PayloadSchemaType.KEYWORD),
        ]:
            try:
                self.client.create_payload_index(self.collection, field, schema)
            except Exception:
                pass  # already indexed

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

    def set_product_payload(self, product_id: str, payload: dict) -> bool:
        """Update payload fields on a product's points, leaving vectors alone.

        Metadata that changes far more often than the images -- stock, per-size
        availability -- must not cost a CLIP pass over the catalog to refresh.
        Returns False when the product has no points yet (nothing to update), so
        callers can tell a refresh from a product that still needs indexing.
        """
        flt = qm.Filter(
            must=[
                qm.FieldCondition(
                    key="product_id", match=qm.MatchValue(value=product_id)
                )
            ]
        )
        hits, _ = self.client.scroll(
            self.collection, scroll_filter=flt, limit=1, with_payload=False
        )
        if not hits:
            return False
        self.client.set_payload(
            self.collection, payload=payload, points=flt, wait=True
        )
        return True

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
        category: Optional[str] = None,
        size: Optional[str] = None,
    ) -> list[tuple[str, float, dict]]:
        """Return (product_id, score, payload), deduped to best vector/product.

        `category` restricts the search to one slice of the index. Cosine
        similarity across categories is close to meaningless -- a black bag and
        a black shoe score highly on colour alone -- so callers scope it.

        `size` keeps only products stocking that size. Filtering here rather than
        after the search means `limit` returns that many *wearable* results,
        instead of a top-N that a size filter then guts.
        """
        must = []
        if in_stock_only:
            must.append(
                qm.FieldCondition(key="in_stock", match=qm.MatchValue(value=True))
            )
        if category:
            must.append(
                qm.FieldCondition(key="category", match=qm.MatchValue(value=category))
            )
        if size:
            must.append(
                qm.FieldCondition(
                    key="sizes_in_stock", match=qm.MatchValue(value=size)
                )
            )
        flt = qm.Filter(must=must) if must else None
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
