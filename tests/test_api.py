"""HTTP surface: upload guards, response shape, webhook HMAC, health."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from app import db, pipeline
from app.config import settings
from app.models import ProductResult, RecommendResponse


@pytest.fixture
def client(sqlite_db, monkeypatch):
    # Don't load real CLIP or hit Qdrant during app startup / health.
    monkeypatch.setattr(pipeline, "warm_up", lambda: None)

    class _FakeStore:
        def count(self):
            return 42

    monkeypatch.setattr("app.main.get_store", lambda *a, **k: _FakeStore())

    from app.main import app

    with TestClient(app) as c:
        yield c


def _stub_recommend(monkeypatch, mode="shoe"):
    resp = RecommendResponse(
        mode=mode,
        confidence=0.9,
        query_attributes={"type": {"value": "boot", "confidence": 0.8}},
        results=[
            ProductResult(
                product_id="1", title="Boot", handle="boot", price="180.00",
                image_url="u", variant_id="v1", score=0.95,
            )
        ],
    )
    monkeypatch.setattr("app.main.pipeline.recommend", lambda data: resp)


def test_recommend_happy_path(client, monkeypatch):
    _stub_recommend(monkeypatch)
    r = client.post(
        "/recommend",
        files={"image": ("shoe.jpg", b"\xff\xd8fakejpeg", "image/jpeg")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "shoe"
    assert body["results"][0]["variant_id"] == "v1"
    assert body["results"][0]["image_url"] == "u"


def test_recommend_rejects_non_image(client):
    r = client.post(
        "/recommend",
        files={"image": ("note.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 415


def test_recommend_rejects_oversized(client, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_bytes", 10)
    r = client.post(
        "/recommend",
        files={"image": ("big.jpg", b"x" * 50, "image/jpeg")},
    )
    assert r.status_code == 413


def test_recommend_rejects_empty(client):
    r = client.post(
        "/recommend",
        files={"image": ("empty.jpg", b"", "image/jpeg")},
    )
    assert r.status_code == 400


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["vector_count"] == 42
    assert "ranker_model" in body


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------
def _sign(secret: str, body: bytes) -> str:
    return base64.b64encode(
        hmac.new(secret.encode(), body, hashlib.sha256).digest()
    ).decode()


def test_webhook_rejects_bad_hmac(client, monkeypatch):
    monkeypatch.setattr(settings, "shopify_webhook_secret", "s3cret")
    body = json.dumps({"id": 123}).encode()
    r = client.post(
        "/webhooks/products-update",
        content=body,
        headers={"X-Shopify-Hmac-Sha256": "wrong"},
    )
    assert r.status_code == 401


def test_webhook_accepts_valid_hmac(client, monkeypatch):
    monkeypatch.setattr(settings, "shopify_webhook_secret", "s3cret")
    called = {}
    monkeypatch.setattr("app.main._reindex_one", lambda gid: called.setdefault("gid", gid))

    body = json.dumps(
        {"id": 123, "admin_graphql_api_id": "gid://shopify/Product/123"}
    ).encode()
    r = client.post(
        "/webhooks/products-update",
        content=body,
        headers={"X-Shopify-Hmac-Sha256": _sign("s3cret", body)},
    )
    assert r.status_code == 200
    assert called["gid"] == "gid://shopify/Product/123"
