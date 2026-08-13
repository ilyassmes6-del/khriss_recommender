"""Core indexing logic shared by indexer.py (CLI) and the products/update webhook.

For each product: download every image, run the SHOE extractor on the first
image for attributes, embed every image into Qdrant (one vector per image,
all pointing back to the product), and upsert the Postgres row. Products with no
image are skipped.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Optional

import httpx

from app import db
from app.clip_model import Embedder, get_embedder
from app.extractor import AttributeExtractor, get_extractor
from app.models import ShoeAttributes
from app.qdrant_store import VectorStore, get_store

# Overridable so the checkpoint can live on a mounted volume in Docker.
CHECKPOINT_FILE = os.environ.get("KHRISS_CHECKPOINT", "index_checkpoint.json")


@dataclass
class IndexStats:
    indexed: int = 0
    skipped_no_image: int = 0
    failed: int = 0
    pruned: int = 0


@dataclass
class RefreshStats:
    refreshed: int = 0
    not_indexed: int = 0
    failed: int = 0


def _stock_payload(product: dict) -> dict:
    """The vector-store payload fields that track availability.

    `sizes_in_stock` is a flat list rather than the full size map because Qdrant
    filters on it: a keyword index over an array matches when any element does,
    which is exactly the "comes in my size" question. The variant IDs behind
    those sizes live in Postgres, where add-to-cart reads them.
    """
    sizes = product.get("sizes") or {}
    return {
        "in_stock": bool(product.get("in_stock", True)),
        "sizes_in_stock": sorted(s for s, v in sizes.items() if v.get("available")),
    }


class Indexer:
    """Embeds products and keeps their metadata current.

    The model is loaded lazily. `refresh_metadata` touches no images and no
    vectors, so a stock/size refresh should not pay a CLIP load and a label-bank
    build just to construct the object that performs it.
    """

    def __init__(
        self,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
        extractor: AttributeExtractor | None = None,
        http: httpx.Client | None = None,
    ):
        self._embedder = embedder
        self._store = store
        self._extractor = extractor
        self.http = http or httpx.Client(timeout=30.0, follow_redirects=True)

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            # get_store needs the vector width, which only the embedder knows.
            self._store = get_store(dim=self.embedder.dim)
        return self._store

    @property
    def extractor(self) -> AttributeExtractor:
        if self._extractor is None:
            self._extractor = get_extractor()
        return self._extractor

    # --- single product ---------------------------------------------------
    def index_product(self, product: dict) -> bool:
        """Index one normalised product dict. Returns False if skipped."""
        images = product.get("images") or []
        if not images:
            return False

        # An unmapped product_type means we cannot tell what this item is, and
        # a mis-filed product pollutes results in both directions -- skip it
        # rather than describe a handbag with the shoe vocabulary.
        category = product.get("category")
        if not category:
            return False

        image_bytes = [self._download(u) for u in images]
        image_bytes = [b for b in image_bytes if b is not None]
        if not image_bytes:
            return False

        # Attributes come from the first (primary) image, read through the
        # vocabulary for this item's own category.
        attrs = self.extractor.extract_item(image_bytes[0], category)
        vectors = [self.embedder.embed_image(b) for b in image_bytes]

        payload = {
            "category": category,
            "type": attrs.type_value(),
            "formality": attrs.formality,
            "season": attrs.season.value if attrs.season else None,
            "dominant_colors": attrs.dominant_colors,
            **_stock_payload(product),
        }
        self.store.upsert_product_vectors(product["product_id"], vectors, payload)

        with db.get_session() as session:
            db.upsert_product(
                session,
                product_id=product["product_id"],
                handle=product["handle"],
                title=product["title"],
                price=product.get("price"),
                product_type=product.get("product_type"),
                category=category,
                variant_id=product.get("variant_id"),
                image_url=images[0],
                in_stock=bool(product.get("in_stock", True)),
                sizes=product.get("sizes") or {},
                shoe_type=attrs.type_value(),
                formality=attrs.formality,
                season=attrs.season.value if attrs.season else None,
                attributes=attrs.model_dump(mode="json"),
            )
            session.commit()
        return True

    def refresh_metadata(self, product: dict) -> bool:
        """Update stock and per-size availability without re-embedding.

        Images are what make indexing expensive, and they are also what almost
        never changes: stock moves daily, a product's photos do not. This path
        touches only the Postgres row and the Qdrant payload, so refreshing the
        whole catalog costs a Shopify page-through rather than a CLIP pass.

        Returns False for a product that is not indexed yet -- it needs the full
        `index_product` path, and silently "refreshing" it would leave a product
        the shopper can never be shown.
        """
        pid = product["product_id"]
        if not self.store.set_product_payload(pid, _stock_payload(product)):
            return False
        with db.get_session() as session:
            obj = session.get(db.Product, pid)
            if obj is None:
                return False
            obj.in_stock = bool(product.get("in_stock", True))
            obj.sizes = product.get("sizes") or {}
            obj.variant_id = product.get("variant_id")
            obj.price = product.get("price")
            session.commit()
        return True

    def delete_product(self, product_id: str) -> None:
        self.store.delete_product(product_id)
        with db.get_session() as session:
            obj = session.get(db.Product, product_id)
            if obj:
                session.delete(obj)
                session.commit()

    # --- full / incremental run ------------------------------------------
    def run(
        self,
        products,
        resume: bool = True,
        progress: Optional[Callable[[dict, bool], None]] = None,
        prune: bool = False,
    ) -> IndexStats:
        stats = IndexStats()
        done = _load_checkpoint() if resume else set()
        kept: set[str] = set()
        for product in products:
            pid = product["product_id"]
            if pid in done:
                kept.add(pid)  # indexed on an earlier pass, still in the feed
                continue
            try:
                ok = self.index_product(product)
                if ok:
                    stats.indexed += 1
                    kept.add(pid)
                else:
                    stats.skipped_no_image += 1
            except Exception:
                stats.failed += 1
                if progress:
                    progress(product, False)
                continue
            done.add(pid)
            _save_checkpoint(done)
            if progress:
                progress(product, True)

        if prune:
            stats.pruned = self.prune(kept)
        return stats

    def refresh_all(
        self,
        products,
        progress: Optional[Callable[[dict, bool], None]] = None,
    ) -> RefreshStats:
        """Refresh stock/size metadata across the feed, without re-embedding.

        Deliberately does not consult the checkpoint: the checkpoint records
        what has been *embedded*, and this pass is about the metadata attached
        to those embeddings, which goes stale on its own schedule.
        """
        stats = RefreshStats()
        for product in products:
            try:
                if self.refresh_metadata(product):
                    stats.refreshed += 1
                else:
                    stats.not_indexed += 1
            except Exception:
                stats.failed += 1
                if progress:
                    progress(product, False)
                continue
            if progress:
                progress(product, True)
        return stats

    def prune(self, keep_ids: set[str]) -> int:
        """Delete indexed products that this run did not keep.

        run() only ever upserts, so anything that leaves the catalog -- a
        product switched to draft, deleted, or one we now skip because its
        product_type maps to no category -- would otherwise stay searchable
        forever. Only safe after a full pass, which sees the whole feed.
        """
        with db.get_session() as session:
            existing = {p.product_id for p in session.query(db.Product).all()}
            stale = existing - keep_ids
            for pid in stale:
                self.store.delete_product(pid)
                obj = session.get(db.Product, pid)
                if obj is not None:
                    session.delete(obj)
            session.commit()
        return len(stale)

    def _download(self, url: str) -> Optional[bytes]:
        try:
            r = self.http.get(url)
            r.raise_for_status()
            return r.content
        except Exception:
            return None


def _load_checkpoint() -> set[str]:
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    try:
        with open(CHECKPOINT_FILE) as f:
            return set(json.load(f))
    except Exception:
        return set()


def _save_checkpoint(done: set[str]) -> None:
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(sorted(done), f)
    os.replace(tmp, CHECKPOINT_FILE)


def clear_checkpoint() -> None:
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
