"""Routing: is this an OUTFIT, or a single item — and if so, which category?

Zero-shot OpenCLIP against one prompt set per destination. We average the
cosine of the image against each set, softmax across all of them, and compare
the top two. When the margin is under MODE_CONFIDENCE_THRESHOLD we return
"both" and let the storefront show two tabs rather than guess wrong.

The item categories mirror app.categories, because the winner decides which
slice of the index gets searched and which label schema describes the query.
"""
from __future__ import annotations

import numpy as np

from app import categories
from app.clip_model import Embedder, get_embedder
from app.config import settings
from app.models import RouteResult

OUTFIT_PROMPTS = [
    "a photo of a person wearing an outfit",
    "a full body photo of someone's clothing",
]

# One prompt set per category. Kept deliberately close to how a product photo
# actually looks, since that is what the shopper is most likely uploading.
CATEGORY_PROMPTS: dict[str, list[str]] = {
    categories.SHOES: [
        "a photo of a shoe",
        "a close-up product photo of footwear",
    ],
    categories.BAGS: [
        "a photo of a handbag",
        "a close-up product photo of a bag or purse",
    ],
    categories.JEWELRY: [
        "a photo of a piece of jewellery",
        "a close-up product photo of a ring, bracelet, necklace or earrings",
    ],
}


class ModeRouter:
    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or get_embedder()
        self.outfit_vec = _mean_unit(self.embedder.embed_texts(OUTFIT_PROMPTS))
        self.category_vecs = {
            cat: _mean_unit(self.embedder.embed_texts(prompts))
            for cat, prompts in CATEGORY_PROMPTS.items()
        }

    def route(self, image_bytes: bytes, vec=None) -> RouteResult:
        # Callers that already embedded this image pass `vec` in; see
        # pipeline.recommend, which embeds once and shares it.
        if vec is None:
            vec = self.embedder.embed_image(image_bytes)

        names = ["outfit"] + list(self.category_vecs)
        sims = [float(self.outfit_vec @ vec)] + [
            float(v @ vec) for v in self.category_vecs.values()
        ]
        probs = _softmax(np.array(sims), settings.route_temperature)

        order = np.argsort(probs)[::-1]
        top, second = int(order[0]), int(order[1])
        margin = float(probs[top] - probs[second])

        outfit_p = float(probs[0])
        # Best single-item score, whichever category won it.
        item_idx = int(np.argmax(probs[1:])) + 1
        item_category = names[item_idx]

        if margin < settings.mode_confidence_threshold:
            # Ambiguous. Show both readings; the item side uses whichever
            # category scored highest, so the tab is still category-scoped.
            return RouteResult(
                mode="both",
                confidence=margin,
                shoe_score=float(probs[item_idx]),
                outfit_score=outfit_p,
                category=item_category,
            )

        if names[top] == "outfit":
            return RouteResult(
                mode="outfit",
                confidence=outfit_p,
                shoe_score=float(probs[item_idx]),
                outfit_score=outfit_p,
                category=None,
            )

        return RouteResult(
            mode="shoe",  # "single item" -- name kept for API compatibility
            confidence=float(probs[top]),
            shoe_score=float(probs[top]),
            outfit_score=outfit_p,
            category=names[top],
        )


def _mean_unit(arr: np.ndarray) -> np.ndarray:
    m = arr.mean(axis=0)
    return m / (np.linalg.norm(m) or 1.0)


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    x = x * temperature
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()
