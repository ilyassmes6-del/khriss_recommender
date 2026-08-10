"""Offline test harness.

Everything heavy (OpenCLIP, Qdrant server, Postgres, Anthropic, Shopify) is
replaced by an in-process fake so `pytest` runs with no network and no torch:

* FakeEmbedder — a toy CLIP over a fixed concept vocabulary. Images are JSON
  bytes listing their concepts; text is matched by substring. Cosine between an
  image and a text prompt is therefore *meaningful and deterministic*, which is
  what lets us assert on routing / extraction / retrieval behaviour.
* FakeVectorStore — in-memory cosine search satisfying VectorStore's interface.
* FakeShopify / FakeLLM / FakeHttp — canned catalog and LLM replies.
* SQLite stands in for Postgres.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from app import db
from app.config import settings

# ---------------------------------------------------------------------------
# Concept vocabulary. Dimensions of the toy embedding space.
# ---------------------------------------------------------------------------
VOCAB = [
    "shoe", "outfit", "clothing", "person",
    "sneaker", "boot", "loafer", "heel", "sandal", "oxford",
    "leather", "suede", "canvas", "mesh",
    "black", "white", "red", "brown", "blue", "tan",
    "casual", "formal", "athletic", "elegant", "sporty",
    "summer", "winter", "dress", "jeans",
]
_DIM = len(VOCAB)


def _vec(tokens: list[str]) -> np.ndarray:
    v = np.zeros(_DIM, dtype="float32")
    for i, w in enumerate(VOCAB):
        if any(w in t for t in tokens):
            v[i] = 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


def img_bytes(concepts: list[str]) -> bytes:
    """Build a fake image whose 'pixels' are its concept list."""
    return json.dumps({"concepts": concepts}).encode()


class FakeEmbedder:
    dim = _DIM

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.stack([_vec([t.lower()]) for t in texts])

    def embed_image(self, image_bytes: bytes) -> np.ndarray:
        data = json.loads(image_bytes)
        return _vec([c.lower() for c in data["concepts"]])


# ---------------------------------------------------------------------------
# In-memory vector store
# ---------------------------------------------------------------------------
class FakeVectorStore:
    def __init__(self, dim: int = _DIM):
        self.dim = dim
        self.collection = "shoes"
        # product_id -> list[(vector, payload)]
        self._data: dict[str, list[tuple[np.ndarray, dict]]] = {}

    def ensure_collection(self, dim=None):  # noqa: D401 - interface parity
        pass

    def upsert_product_vectors(self, product_id, vectors, payload):
        self._data[product_id] = [(np.asarray(v, dtype="float32"), dict(payload)) for v in vectors]

    def delete_product(self, product_id):
        self._data.pop(product_id, None)

    def get_first_vector(self, product_id):
        recs = self._data.get(product_id)
        return recs[0][0] if recs else None

    def search(self, query, limit, in_stock_only=True, over_fetch=4, category=None):
        best: dict[str, tuple[float, dict]] = {}
        for pid, recs in self._data.items():
            for vec, payload in recs:
                if in_stock_only and not payload.get("in_stock", True):
                    continue
                # Mirrors the Qdrant payload filter: retrieval scopes every
                # search to one category so types never rank against each other.
                if category and payload.get("category") != category:
                    continue
                score = float(np.asarray(query, dtype="float32") @ vec)
                if pid not in best or score > best[pid][0]:
                    best[pid] = (score, payload)
        ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)
        return [(pid, s, p) for pid, (s, p) in ranked[:limit]]

    def count(self):
        return sum(len(v) for v in self._data.values())


# ---------------------------------------------------------------------------
# Fake external services
# ---------------------------------------------------------------------------
class FakeHttp:
    """Maps image 'urls' (which are concept keys) to fake image bytes."""

    def __init__(self, mapping: dict[str, bytes]):
        self.mapping = mapping

    def get(self, url):
        class _R:
            def __init__(self, content):
                self.content = content

            def raise_for_status(self):
                pass

        return _R(self.mapping[url])


class FakeShopify:
    def __init__(self, products):
        self._products = products

    def iter_products(self):
        yield from self._products

    def get_product(self, gid):
        pid = gid.rsplit("/", 1)[-1]
        return next((p for p in self._products if p["product_id"] == pid), None)


class FakeLLM:
    """OpenRouter LLMClient stand-in: returns queued replies, one per chat()."""

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self._replies.pop(0) if self._replies else "{}"


# ---------------------------------------------------------------------------
# Fixture catalog: ~20 fake shoes across types/colours/formality/season.
# Each product's image concepts drive the fake embedder deterministically.
# ---------------------------------------------------------------------------
def _p(pid, title, concepts, price, in_stock=True, category="shoes"):
    key = f"img-{pid}"
    return {
        "product": {
            "product_id": pid,
            "handle": title.lower().replace(" ", "-"),
            "title": title,
            "product_type": "Shoes",
            # Set from Shopify's product_type in production; the indexer skips
            # products without one rather than guessing the category.
            "category": category,
            "tags": [],
            "images": [key],
            "price": price,
            "variant_id": f"v{pid}",
            "in_stock": in_stock,
        },
        "image_key": key,
        "image_bytes": img_bytes(["shoe"] + concepts),
    }


_RAW = [
    _p("1", "Black Leather Chelsea Boot", ["boot", "leather", "black", "formal", "winter"], "180.00"),
    _p("2", "Brown Suede Chukka Boot", ["boot", "suede", "brown", "casual", "winter"], "150.00"),
    _p("3", "White Canvas Sneaker", ["sneaker", "canvas", "white", "casual", "summer"], "80.00"),
    _p("4", "Red Mesh Running Sneaker", ["sneaker", "mesh", "red", "athletic", "sporty"], "120.00"),
    _p("5", "Black Leather Oxford", ["oxford", "leather", "black", "formal"], "200.00"),
    _p("6", "Tan Leather Loafer", ["loafer", "leather", "tan", "casual"], "160.00"),
    _p("7", "Black Suede Heel", ["heel", "suede", "black", "elegant", "formal"], "140.00"),
    _p("8", "Blue Canvas Sneaker", ["sneaker", "canvas", "blue", "casual", "summer"], "75.00"),
    _p("9", "Brown Leather Boot", ["boot", "leather", "brown", "casual", "winter"], "170.00"),
    _p("10", "White Leather Sneaker", ["sneaker", "leather", "white", "casual"], "110.00"),
    _p("11", "Black Sandal", ["sandal", "black", "summer", "casual"], "60.00"),
    _p("12", "Tan Summer Sandal", ["sandal", "tan", "summer", "casual"], "55.00"),
    _p("13", "Black Athletic Sneaker", ["sneaker", "mesh", "black", "athletic", "sporty"], "130.00"),
    _p("14", "Brown Oxford Dress Shoe", ["oxford", "leather", "brown", "formal"], "210.00"),
    _p("15", "White Sporty Sneaker", ["sneaker", "mesh", "white", "athletic", "sporty"], "125.00"),
    _p("16", "Red Elegant Heel", ["heel", "leather", "red", "elegant", "formal"], "175.00"),
    _p("17", "Blue Suede Loafer", ["loafer", "suede", "blue", "casual"], "155.00"),
    _p("18", "Black Winter Boot", ["boot", "leather", "black", "winter", "casual"], "190.00"),
    # An out-of-stock item that must never appear in results.
    _p("19", "White Canvas Sneaker OOS", ["sneaker", "canvas", "white", "casual"], "70.00", in_stock=False),
    _p("20", "Brown Casual Loafer", ["loafer", "leather", "brown", "casual"], "145.00"),
]


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def catalog():
    return _RAW


@pytest.fixture
def http_map(catalog):
    return {c["image_key"]: c["image_bytes"] for c in catalog}


@pytest.fixture
def sqlite_db(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'khriss_test.db'}"
    monkeypatch.setattr(settings, "database_url", url)
    db._engine = None
    db._SessionLocal = None
    db.init_db()
    yield
    db._engine = None
    db._SessionLocal = None


@pytest.fixture
def indexed(catalog, http_map, fake_embedder, sqlite_db, tmp_path, monkeypatch):
    """Build an indexed system: fake store + sqlite populated from the catalog."""
    from app.extractor import ClipExtractor
    from app.indexing import Indexer

    monkeypatch.setattr("app.indexing.CHECKPOINT_FILE", str(tmp_path / "ckpt.json"))
    store = FakeVectorStore()
    extractor = ClipExtractor(embedder=fake_embedder)
    indexer = Indexer(
        store=store,
        embedder=fake_embedder,
        extractor=extractor,
        http=FakeHttp(http_map),
    )
    products = [c["product"] for c in catalog]
    stats = indexer.run(products, resume=False)

    # Wire retrieval's module-level singletons to our fakes.
    monkeypatch.setattr("app.retrieval.get_store", lambda dim=_DIM: store)
    monkeypatch.setattr("app.retrieval.get_embedder", lambda: fake_embedder)
    return {"store": store, "extractor": extractor, "embedder": fake_embedder, "stats": stats}
