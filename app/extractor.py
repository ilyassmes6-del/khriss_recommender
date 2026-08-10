"""Attribute extraction behind a single interface.

Two implementations satisfy `AttributeExtractor`:

* ClipExtractor  — zero-shot cosine scoring against the label schemas. No API
  cost. This is what the indexer runs on every catalog image and what Mode B/A
  run on queries by default.
* LLMExtractor  — OpenRouter vision (Claude Haiku 4.5 by default), selected with
  EXTRACTOR=llm, for stores that want higher-fidelity tags at ~$0.001/image.

Both return the same Pydantic types (ShoeAttributes / OutfitAttributes), so
nothing downstream cares which one produced them.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Protocol

import numpy as np

from app import labels
from app.clip_model import Embedder, get_embedder
from app.config import settings
from app.models import ItemAttributes, LabeledAxis, OutfitAttributes, ShoeAttributes


class AttributeExtractor(Protocol):
    # `vec` lets a caller that has already embedded this image hand the vector
    # in rather than paying for a second CLIP pass. One /recommend request
    # touched the same image up to five times before this existed.
    def extract_shoe(self, image_bytes: bytes, vec=None) -> ShoeAttributes: ...

    def extract_outfit(self, image_bytes: bytes, vec=None) -> OutfitAttributes: ...

    def extract_item(
        self, image_bytes: bytes, category: str, vec=None
    ) -> ItemAttributes: ...


# ---------------------------------------------------------------------------
# CLIP zero-shot extractor
# ---------------------------------------------------------------------------
def _softmax(x: np.ndarray) -> np.ndarray:
    # Scale by CLIP's logit temperature (100) so cosine gaps become decisive.
    x = x * 100.0
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


class _LabelBank:
    """Pre-computed mean template embedding per label for one schema."""

    def __init__(self, embedder: Embedder, schema: dict, templates: list[str]):
        self.axes: dict[str, list[str]] = {}
        self.embeddings: dict[str, np.ndarray] = {}
        for axis, candidates in schema.items():
            self.axes[axis] = candidates
            prompts, spans = [], []
            for label in candidates:
                start = len(prompts)
                prompts.extend(t.format(label=label) for t in templates)
                spans.append((start, len(prompts)))
            vecs = embedder.embed_texts(prompts)
            # Mean across templates, then re-normalise -> one vector per label.
            per_label = np.stack([vecs[s:e].mean(axis=0) for s, e in spans])
            norm = np.linalg.norm(per_label, axis=-1, keepdims=True)
            self.embeddings[axis] = per_label / np.where(norm == 0, 1.0, norm)

    def score_axis(self, axis: str, image_vec: np.ndarray) -> np.ndarray:
        return self.embeddings[axis] @ image_vec  # cosine, both normalised


class ClipExtractor:
    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or get_embedder()
        self._shoe = _LabelBank(self.embedder, labels.SHOE_SCHEMA, labels.SHOE_TEMPLATES)
        self._shoe_multi = _LabelBank(
            self.embedder, labels.SHOE_MULTI, labels.SHOE_TEMPLATES
        )
        self._outfit = _LabelBank(
            self.embedder, labels.OUTFIT_SCHEMA, labels.OUTFIT_TEMPLATES
        )
        self._outfit_multi = _LabelBank(
            self.embedder, labels.OUTFIT_MULTI, labels.OUTFIT_TEMPLATES
        )
        # Per-category banks are built lazily: a store with no jewellery should
        # not pay to embed a jewellery vocabulary at every boot.
        self._item_banks: dict[str, _LabelBank] = {}
        self._item_multi_banks: dict[str, _LabelBank] = {}
        self._formality_cache: dict[tuple, list] = {}

    def _top_axis(self, bank: _LabelBank, axis: str, vec: np.ndarray) -> LabeledAxis:
        sims = bank.score_axis(axis, vec)
        probs = _softmax(sims)
        idx = int(np.argmax(sims))
        return LabeledAxis(value=bank.axes[axis][idx], confidence=float(probs[idx]))

    def _multi(self, bank: _LabelBank, axis: str, vec: np.ndarray) -> list[str]:
        sims = bank.score_axis(axis, vec)
        order = np.argsort(sims)[::-1]
        out: list[str] = []
        for i in order[: labels.MULTI_LABEL_MAX]:
            # Compare raw cosine against the floor; softmax would be too flat
            # for large label sets like the colour palette.
            if sims[i] >= labels.MULTI_LABEL_FLOOR:
                out.append(bank.axes[axis][int(i)])
        return out

    def _formality_anchors(self, templates: list[str]) -> list[tuple[int, np.ndarray]]:
        """One unit vector per formality level, built once per template set.

        These anchors depend only on the templates, never on the image, but
        were being re-embedded on every product -- a text-encoder pass per
        item, which is most of the per-product cost once the label banks are
        warm.
        """
        key = tuple(templates)
        cached = self._formality_cache.get(key)
        if cached is not None:
            return cached

        prompts, spans = [], []
        for score, desc in labels.FORMALITY_ANCHORS.items():
            start = len(prompts)
            prompts.extend(t.format(label=desc) for t in templates)
            spans.append((score, start, len(prompts)))
        text_vecs = self.embedder.embed_texts(prompts)

        anchors = []
        for score, s, e in spans:
            a = text_vecs[s:e].mean(axis=0)
            anchors.append((score, a / (np.linalg.norm(a) or 1.0)))
        self._formality_cache[key] = anchors
        return anchors

    def _formality(self, vec: np.ndarray, templates: list[str]) -> int:
        best_score, best_sim = 3, -1.0
        for score, anchor in self._formality_anchors(templates):
            sim = float(anchor @ vec)
            if sim > best_sim:
                best_sim, best_score = sim, score
        return best_score

    def extract_shoe(self, image_bytes: bytes, vec=None) -> ShoeAttributes:
        if vec is None:
            vec = self.embedder.embed_image(image_bytes)
        return ShoeAttributes(
            type=self._top_axis(self._shoe, "type", vec),
            material=self._top_axis(self._shoe, "material", vec),
            sole_type=self._top_axis(self._shoe, "sole_type", vec),
            heel_height=self._top_axis(self._shoe, "heel_height", vec),
            toe_shape=self._top_axis(self._shoe, "toe_shape", vec),
            pattern=self._top_axis(self._shoe, "pattern", vec),
            season=self._top_axis(self._shoe, "season", vec),
            formality=self._formality(vec, labels.SHOE_TEMPLATES),
            dominant_colors=self._multi(self._shoe_multi, "dominant_colors", vec),
            style_tags=self._multi(self._shoe_multi, "style_tags", vec),
        )

    def extract_item(
        self, image_bytes: bytes, category: str, vec=None
    ) -> ItemAttributes:
        """Describe a single item using its own category's vocabulary.

        Scoring a bracelet against the shoe schema returns a confident "mule",
        which then drives retrieval. The schema is chosen by category, which
        comes from Shopify's product_type rather than from CLIP.
        """
        if vec is None:
            vec = self.embedder.embed_image(image_bytes)

        bank = self._bank_for(category)
        multi = self._multi_bank_for(category)
        schema = labels.SCHEMA_BY_CATEGORY[category]
        templates = labels.TEMPLATES_BY_CATEGORY[category]

        shared = {"type", "material", "pattern", "season"}
        extras = {
            axis: self._top_axis(bank, axis, vec)
            for axis in schema
            if axis not in shared
        }
        return ItemAttributes(
            category=category,
            type=self._top_axis(bank, "type", vec) if "type" in schema else None,
            material=self._top_axis(bank, "material", vec) if "material" in schema else None,
            pattern=self._top_axis(bank, "pattern", vec) if "pattern" in schema else None,
            season=self._top_axis(bank, "season", vec) if "season" in schema else None,
            formality=self._formality(vec, templates),
            dominant_colors=self._multi(multi, "dominant_colors", vec),
            style_tags=self._multi(multi, "style_tags", vec),
            extras=extras,
        )

    def _bank_for(self, category: str) -> _LabelBank:
        if category not in self._item_banks:
            self._item_banks[category] = _LabelBank(
                self.embedder,
                labels.SCHEMA_BY_CATEGORY[category],
                labels.TEMPLATES_BY_CATEGORY[category],
            )
        return self._item_banks[category]

    def _multi_bank_for(self, category: str) -> _LabelBank:
        if category not in self._item_multi_banks:
            self._item_multi_banks[category] = _LabelBank(
                self.embedder,
                labels.MULTI_BY_CATEGORY[category],
                labels.TEMPLATES_BY_CATEGORY[category],
            )
        return self._item_multi_banks[category]

    def extract_outfit(self, image_bytes: bytes, vec=None) -> OutfitAttributes:
        if vec is None:
            vec = self.embedder.embed_image(image_bytes)
        return OutfitAttributes(
            silhouette=self._top_axis(self._outfit, "silhouette", vec),
            pattern=self._top_axis(self._outfit, "pattern", vec),
            season=self._top_axis(self._outfit, "season", vec),
            occasion=self._top_axis(self._outfit, "occasion", vec),
            formality=self._formality(vec, labels.OUTFIT_TEMPLATES),
            dominant_colors=self._multi(self._outfit_multi, "dominant_colors", vec),
            garments_present=self._multi(self._outfit_multi, "garments_present", vec),
            style_tags=self._multi(self._outfit_multi, "style_tags", vec),
        )


# ---------------------------------------------------------------------------
# OpenRouter vision extractor (optional, selected by EXTRACTOR=llm)
# ---------------------------------------------------------------------------
_SHOE_KEYS = list(labels.SHOE_SCHEMA) + list(labels.SHOE_MULTI) + ["formality"]
_OUTFIT_KEYS = list(labels.OUTFIT_SCHEMA) + list(labels.OUTFIT_MULTI) + ["formality"]


class LLMExtractor:
    """Vision extraction via OpenRouter (Claude Haiku 4.5 by default).

    Returns the same Pydantic types as ClipExtractor. We still keep a CLIP
    embedder around because the *indexer* needs image vectors for Qdrant
    regardless of which extractor produced the tags.
    """

    def __init__(self, client=None, embedder: Embedder | None = None):
        if client is None:
            from app.llm import get_llm

            client = get_llm()
        self.client = client
        self.embedder = embedder or get_embedder()

    def _ask(self, image_bytes: bytes, schema_hint: str) -> dict:
        from app.llm import image_message

        model = settings.vision_model or settings.ranker_model
        text = self.client.chat(
            model=model,
            messages=[image_message(schema_hint, image_bytes)],
            max_tokens=512,
            temperature=0.0,
        )
        return _loads_json(text)

    def extract_shoe(self, image_bytes: bytes, vec=None) -> ShoeAttributes:
        hint = _schema_prompt("shoe", labels.SHOE_SCHEMA, labels.SHOE_MULTI)
        raw = self._ask(image_bytes, hint)
        return _coerce_shoe(raw)

    def extract_outfit(self, image_bytes: bytes, vec=None) -> OutfitAttributes:
        hint = _schema_prompt("outfit", labels.OUTFIT_SCHEMA, labels.OUTFIT_MULTI)
        raw = self._ask(image_bytes, hint)
        return _coerce_outfit(raw)


def _schema_prompt(kind: str, single: dict, multi: dict) -> str:
    single_desc = "\n".join(f"- {k}: one of {v}" for k, v in single.items())
    multi_desc = "\n".join(f"- {k}: up to 3 of {v}" for k, v in multi.items())
    return (
        f"Analyse this {kind} photo and return STRICT JSON only, no prose.\n"
        f"Single-choice fields:\n{single_desc}\n"
        f"Multi-choice fields (arrays):\n{multi_desc}\n"
        f"- formality: integer 1 (very casual) to 5 (black tie)\n"
        f"Return an object with exactly these keys."
    )


def _loads_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def _axis(raw: dict, key: str) -> LabeledAxis | None:
    v = raw.get(key)
    if not v:
        return None
    return LabeledAxis(value=str(v), confidence=0.9)  # Haiku gives no score


def _coerce_shoe(raw: dict) -> ShoeAttributes:
    return ShoeAttributes(
        type=_axis(raw, "type"),
        material=_axis(raw, "material"),
        sole_type=_axis(raw, "sole_type"),
        heel_height=_axis(raw, "heel_height"),
        toe_shape=_axis(raw, "toe_shape"),
        pattern=_axis(raw, "pattern"),
        season=_axis(raw, "season"),
        formality=int(raw.get("formality", 3)),
        dominant_colors=list(raw.get("dominant_colors", []))[:3],
        style_tags=list(raw.get("style_tags", []))[:3],
    )


def _coerce_outfit(raw: dict) -> OutfitAttributes:
    return OutfitAttributes(
        silhouette=_axis(raw, "silhouette"),
        pattern=_axis(raw, "pattern"),
        season=_axis(raw, "season"),
        occasion=_axis(raw, "occasion"),
        formality=int(raw.get("formality", 3)),
        dominant_colors=list(raw.get("dominant_colors", []))[:3],
        garments_present=list(raw.get("garments_present", []))[:3],
        style_tags=list(raw.get("style_tags", []))[:3],
    )


@lru_cache
def get_extractor() -> AttributeExtractor:
    if settings.extractor.lower() in {"llm", "haiku", "openrouter"}:
        return LLMExtractor()
    return ClipExtractor()
