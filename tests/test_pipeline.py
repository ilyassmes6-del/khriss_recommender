"""End-to-end pipeline (route -> extract -> retrieve -> rank) with fakes."""
from __future__ import annotations

import json

import pytest

from app import pipeline
from app.config import settings
from app.router import ModeRouter
from tests.conftest import FakeLLM, img_bytes


@pytest.fixture
def wired(indexed, fake_embedder, monkeypatch):
    """Point the pipeline's singletons at the fakes from the `indexed` fixture."""
    monkeypatch.setattr(pipeline, "_router", ModeRouter(embedder=fake_embedder))
    monkeypatch.setattr("app.pipeline.get_extractor", lambda: indexed["extractor"])
    return indexed


def test_pipeline_shoe_mode(wired):
    resp = pipeline.recommend(img_bytes(["shoe", "boot", "leather", "black"]))
    assert resp.mode == "shoe"
    assert resp.results
    # Mode B carries no rationale.
    assert all(r.rationale is None for r in resp.results)
    assert "type" in resp.query_attributes


def test_pipeline_outfit_mode(wired):
    reply = json.dumps(
        {"ranked": [{"product_id": "5", "rationale": "Sharp oxford.", "coherence": 0.9}]}
    )
    resp = pipeline.recommend(
        img_bytes(["outfit", "person", "formal", "elegant"]),
        llm_client=FakeLLM([reply]),
    )
    assert resp.mode == "outfit"
    assert resp.results and resp.results[0].rationale == "Sharp oxford."


def test_pipeline_both_mode(wired, monkeypatch):
    monkeypatch.setattr(settings, "mode_confidence_threshold", 2.0)
    reply = json.dumps(
        {"ranked": [{"product_id": "6", "rationale": "Loafer works.", "coherence": 0.6}]}
    )
    resp = pipeline.recommend(
        img_bytes(["shoe", "outfit"]),
        llm_client=FakeLLM([reply]),
    )
    assert resp.mode == "both"
    assert resp.shoe_results is not None
    assert resp.outfit_results is not None
