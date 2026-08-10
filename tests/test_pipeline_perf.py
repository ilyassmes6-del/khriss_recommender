"""One CLIP pass per request.

Routing, extraction and retrieval all score against the same uploaded image.
Each used to embed it independently, so an ambiguous upload paid for five
identical forward passes -- on a shared vCPU that was the whole response time.
These count the passes, because the cost is invisible from the response.
"""
from __future__ import annotations

import pytest

from app import pipeline
from tests.conftest import img_bytes


class CountingEmbedder:
    """Wraps the fake embedder and tallies image forward passes."""

    def __init__(self, inner):
        self._inner = inner
        self.image_calls = 0
        self.dim = inner.dim

    def embed_texts(self, texts):
        return self._inner.embed_texts(texts)

    def embed_image(self, image_bytes):
        self.image_calls += 1
        return self._inner.embed_image(image_bytes)


@pytest.fixture
def counting(indexed, monkeypatch):
    counter = CountingEmbedder(indexed["embedder"])
    monkeypatch.setattr("app.retrieval.get_embedder", lambda: counter)

    from app.router import ModeRouter

    # recommend() takes its embedder from the router, so this is the one that
    # counts -- and it is the same wiring test_pipeline.py uses.
    monkeypatch.setattr(pipeline, "_router", ModeRouter(embedder=counter))
    monkeypatch.setattr(pipeline, "get_extractor", lambda: indexed["extractor"])
    indexed["extractor"].embedder = counter
    counter.image_calls = 0  # ignore setup
    return counter


def _reply(ids):
    import json

    return json.dumps({"ranked": [{"product_id": i, "rationale": "x"} for i in ids]})


def test_shoe_upload_embeds_once(counting, indexed):
    from tests.conftest import FakeLLM

    pipeline.recommend(
        img_bytes(["shoe", "sneaker", "white", "canvas"]), llm_client=FakeLLM([_reply(["3"])])
    )
    assert counting.image_calls == 1, (
        f"one upload should mean one CLIP pass, got {counting.image_calls}"
    )


def test_outfit_upload_embeds_once(counting):
    from tests.conftest import FakeLLM

    pipeline.recommend(
        img_bytes(["outfit", "person", "jeans", "casual"]),
        llm_client=FakeLLM([_reply(["3", "8", "10"])]),
    )
    assert counting.image_calls == 1, (
        f"one upload should mean one CLIP pass, got {counting.image_calls}"
    )


def test_ambiguous_upload_still_embeds_once(counting):
    """The 'both' path runs shoe *and* outfit routes -- the worst case."""
    from tests.conftest import FakeLLM

    pipeline.recommend(
        img_bytes(["shoe", "outfit", "person"]),
        llm_client=FakeLLM([_reply(["3", "8", "10"])]),
    )
    assert counting.image_calls == 1, (
        f"the 'both' path used to embed five times, got {counting.image_calls}"
    )
