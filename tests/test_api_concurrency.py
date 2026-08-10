"""The event loop must survive a slow recommendation.

CLIP inference is blocking CPU work. Called straight from the coroutine it
stalls every other request -- uploads, /health, CORS preflights -- until the
proxy times them out and returns 502, while the app still logs 200 for each.
That failure is invisible to a single-request test, so this one holds a
recommendation open and checks the server answers something else meanwhile.
"""
from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient

from app import pipeline
from tests.conftest import img_bytes


@pytest.fixture
def slow_app(sqlite_db, monkeypatch):
    """A recommendation that blocks until the test releases it."""
    released = threading.Event()
    entered = threading.Event()

    # Same stubs as tests/test_api.py: no real CLIP load, no Qdrant on /health.
    monkeypatch.setattr(pipeline, "warm_up", lambda: None)

    class _FakeStore:
        def count(self):
            return 42

    monkeypatch.setattr("app.main.get_store", lambda *a, **k: _FakeStore())

    def blocking_recommend(image_bytes, llm_client=None):
        entered.set()
        if not released.wait(timeout=10):
            raise AssertionError("recommendation was never released")
        from app.models import RecommendResponse

        return RecommendResponse(mode="shoe", confidence=1.0, query_attributes={}, results=[])

    monkeypatch.setattr(pipeline, "recommend", blocking_recommend)
    monkeypatch.setattr("app.main.pipeline.recommend", blocking_recommend)

    from app.main import app

    return {"app": app, "entered": entered, "released": released}


def test_health_answers_while_a_recommendation_is_in_flight(slow_app):
    client = TestClient(slow_app["app"])
    result = {}

    def upload():
        result["resp"] = client.post(
            "/recommend",
            files={"image": ("a.jpg", img_bytes(["shoe"]), "image/jpeg")},
        )

    worker = threading.Thread(target=upload, daemon=True)
    worker.start()

    assert slow_app["entered"].wait(timeout=5), "the upload never reached the pipeline"

    # The blocking call is mid-flight. On the old code this hangs until the
    # upload finishes; the endpoint must not own the event loop.
    health = client.get("/health")
    assert health.status_code == 200

    slow_app["released"].set()
    worker.join(timeout=10)
    assert result["resp"].status_code == 200
