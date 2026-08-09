"""Mode A ranking: JSON contract, hallucination guard, repair retry."""
from __future__ import annotations

import json

from app import ranker
from app.models import OutfitAttributes
from tests.conftest import FakeLLM


def _candidates():
    return [
        {"product_id": "5", "title": "Black Leather Oxford", "handle": "blk-ox",
         "price": "200.00", "variant_id": "v5", "image_url": "u5",
         "attributes": {"type": "oxford", "formality": 4}},
        {"product_id": "6", "title": "Tan Leather Loafer", "handle": "tan-lf",
         "price": "160.00", "variant_id": "v6", "image_url": "u6",
         "attributes": {"type": "loafer", "formality": 3}},
    ]


def test_ranker_returns_validated_results():
    reply = json.dumps(
        {"ranked": [
            {"product_id": "5", "rationale": "A crisp oxford sharpens the look.", "coherence": 0.9},
            {"product_id": "6", "rationale": "A loafer keeps it relaxed.", "coherence": 0.7},
        ]}
    )
    client = FakeLLM([reply])
    results = ranker.rank_outfit(OutfitAttributes(formality=4), _candidates(), client=client)
    assert [r.product_id for r in results] == ["5", "6"]
    assert results[0].rationale
    assert results[0].score == 0.9


def test_ranker_drops_hallucinated_ids():
    reply = json.dumps(
        {"ranked": [
            {"product_id": "999", "rationale": "not a real product", "coherence": 0.99},
            {"product_id": "5", "rationale": "real one", "coherence": 0.8},
        ]}
    )
    client = FakeLLM([reply])
    results = ranker.rank_outfit(OutfitAttributes(formality=4), _candidates(), client=client)
    assert [r.product_id for r in results] == ["5"]  # 999 dropped


def test_ranker_repairs_bad_json():
    good = json.dumps({"ranked": [{"product_id": "6", "rationale": "ok", "coherence": 0.5}]})
    client = FakeLLM(["this is not json at all", good])
    results = ranker.rank_outfit(OutfitAttributes(formality=3), _candidates(), client=client)
    assert [r.product_id for r in results] == ["6"]
    assert len(client.calls) == 2  # first call + one repair


def test_ranker_gives_up_gracefully():
    client = FakeLLM(["garbage", "still garbage"])
    results = ranker.rank_outfit(OutfitAttributes(formality=3), _candidates(), client=client)
    assert results == []


def test_ranker_empty_candidates():
    client = FakeLLM([])
    assert ranker.rank_outfit(OutfitAttributes(), [], client=client) == []
