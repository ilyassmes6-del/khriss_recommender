"""Mode routing: is the uploaded image a SHOE or an OUTFIT?

Zero-shot OpenCLIP against two prompt sets. We average the cosine of the image
against each set, softmax the two aggregate scores, and compare the margin
against MODE_CONFIDENCE_THRESHOLD. Below the threshold we return "both" and let
the storefront render two tabs rather than guess wrong.
"""
from __future__ import annotations

import numpy as np

from app.clip_model import Embedder, get_embedder
from app.config import settings
from app.models import RouteResult

SHOE_PROMPTS = [
    "a photo of a shoe",
    "a close-up product photo of footwear",
]
OUTFIT_PROMPTS = [
    "a photo of a person wearing an outfit",
    "a photo of clothing",
]


class ModeRouter:
    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or get_embedder()
        shoe = self.embedder.embed_texts(SHOE_PROMPTS)
        outfit = self.embedder.embed_texts(OUTFIT_PROMPTS)
        # Mean prompt vector per set, re-normalised.
        self.shoe_vec = _mean_unit(shoe)
        self.outfit_vec = _mean_unit(outfit)

    def route(self, image_bytes: bytes) -> RouteResult:
        vec = self.embedder.embed_image(image_bytes)
        shoe_sim = float(self.shoe_vec @ vec)
        outfit_sim = float(self.outfit_vec @ vec)

        probs = _softmax(np.array([shoe_sim, outfit_sim]), settings.route_temperature)
        shoe_p, outfit_p = float(probs[0]), float(probs[1])
        margin = abs(shoe_p - outfit_p)

        if margin < settings.mode_confidence_threshold:
            mode = "both"
            confidence = margin  # deliberately low to signal ambiguity
        elif shoe_p > outfit_p:
            mode, confidence = "shoe", shoe_p
        else:
            mode, confidence = "outfit", outfit_p

        return RouteResult(
            mode=mode,
            confidence=confidence,
            shoe_score=shoe_p,
            outfit_score=outfit_p,
        )


def _mean_unit(arr: np.ndarray) -> np.ndarray:
    m = arr.mean(axis=0)
    return m / (np.linalg.norm(m) or 1.0)


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    x = x * temperature
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()
