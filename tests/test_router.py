"""Mode routing tests."""
from __future__ import annotations

from app.config import settings
from app.router import ModeRouter
from tests.conftest import img_bytes


def test_routes_clear_shoe(fake_embedder):
    router = ModeRouter(embedder=fake_embedder)
    result = router.route(img_bytes(["shoe", "sneaker", "white"]))
    assert result.mode == "shoe"
    assert result.shoe_score > result.outfit_score


def test_routes_clear_outfit(fake_embedder):
    router = ModeRouter(embedder=fake_embedder)
    result = router.route(img_bytes(["outfit", "person", "dress", "casual"]))
    assert result.mode == "outfit"
    assert result.outfit_score > result.shoe_score


def test_ambiguous_returns_both(fake_embedder, monkeypatch):
    # Force the ambiguity branch: any margin below this huge threshold -> both.
    monkeypatch.setattr(settings, "mode_confidence_threshold", 2.0)
    router = ModeRouter(embedder=fake_embedder)
    result = router.route(img_bytes(["shoe", "outfit"]))
    assert result.mode == "both"
    # Both underlying scores are still exposed for the storefront tabs.
    assert 0.0 <= result.shoe_score <= 1.0
    assert 0.0 <= result.outfit_score <= 1.0
