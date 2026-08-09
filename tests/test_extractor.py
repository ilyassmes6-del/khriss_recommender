"""Extractor behaviour: CLIP zero-shot axes + the LLM interface parity."""
from __future__ import annotations

import json

from app.extractor import ClipExtractor, LLMExtractor
from tests.conftest import FakeLLM, FakeEmbedder, img_bytes


def test_clip_extractor_axes():
    ext = ClipExtractor(embedder=FakeEmbedder())
    attrs = ext.extract_shoe(img_bytes(["shoe", "sneaker", "mesh", "red", "athletic"]))
    assert attrs.type.value == "sneaker"
    assert attrs.material.value == "mesh"
    assert "red" in attrs.dominant_colors
    assert 1 <= attrs.formality <= 5


def test_clip_extractor_outfit_axes():
    ext = ClipExtractor(embedder=FakeEmbedder())
    attrs = ext.extract_outfit(img_bytes(["outfit", "person", "dress", "elegant", "formal"]))
    # Multi-label garments come back as a list.
    assert isinstance(attrs.garments_present, list)
    assert isinstance(attrs.dominant_colors, list)


def test_llm_extractor_parity():
    """LLM extractor returns the same Pydantic type, parsed from JSON."""
    reply = json.dumps(
        {
            "type": "boot",
            "material": "leather",
            "sole_type": "rubber sole",
            "heel_height": "flat",
            "toe_shape": "round toe",
            "pattern": "solid color",
            "season": "winter",
            "formality": 4,
            "dominant_colors": ["black", "brown"],
            "style_tags": ["classic"],
        }
    )
    ext = LLMExtractor(client=FakeLLM([reply]), embedder=FakeEmbedder())
    attrs = ext.extract_shoe(img_bytes(["shoe", "boot"]))
    assert attrs.type.value == "boot"
    assert attrs.formality == 4
    assert attrs.dominant_colors == ["black", "brown"]
