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


class Indexer:
    def __init__(
        self,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
        extractor: AttributeExtractor | None = None,
        http: httpx.Client | None = None,
    ):
        self.embedder = embedder or get_embedder()
        self.store = store or get_store(dim=self.embedder.dim)
        self.extractor = extractor or get_extractor()
        self.http = http or httpx.Client(timeout=30.0, follow_redirects=True)

    # --- single product ---------------------------------------------------
    def index_product(self, product: dict) -> bool:
        """Index one normalised product dict. Returns False if skipped."""
        images = product.get("images") or []
        if not images:
            return False

        image_bytes = [self._download(u) for u in images]
        image_bytes = [b for b in image_bytes if b is not None]
        if not image_bytes:
            return False

        # Attributes come from the first (primary) image.
        attrs: ShoeAttributes = self.extractor.extract_shoe(image_bytes[0])
        vectors = [self.embedder.embed_image(b) for b in image_bytes]

        payload = {
            "type": attrs.type_value(),
            "formality": attrs.formality,
            "season": attrs.season.value if attrs.season else None,
            "dominant_colors": attrs.dominant_colors,
            "in_stock": bool(product.get("in_stock", True)),
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
                variant_id=product.get("variant_id"),
                image_url=images[0],
                in_stock=bool(product.get("in_stock", True)),
                shoe_type=attrs.type_value(),
                formality=attrs.formality,
                season=attrs.season.value if attrs.season else None,
                attributes=attrs.model_dump(mode="json"),
            )
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
    ) -> IndexStats:
        stats = IndexStats()
        done = _load_checkpoint() if resume else set()
        for product in products:
            pid = product["product_id"]
            if pid in done:
                continue
            try:
                ok = self.index_product(product)
                if ok:
                    stats.indexed += 1
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
        return stats

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
