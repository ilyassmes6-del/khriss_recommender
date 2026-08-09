"""OpenCLIP wrapper: load once at startup, keep warm, embed images and text.

All embeddings are L2-normalised so a dot product == cosine similarity, which is
what Qdrant's cosine distance and our diversity/routing math assume.

The heavy torch/open_clip import is done lazily inside `load()` so the rest of
the codebase (and the offline test suite) can import this module without pulling
in torch. Tests inject a `FakeEmbedder` instead of calling `load()`.
"""
from __future__ import annotations

import io
from functools import lru_cache
from typing import Protocol

import numpy as np

from app.config import settings


class Embedder(Protocol):
    """Minimal surface the pipeline depends on. Real + fake both satisfy it."""

    dim: int

    def embed_image(self, image_bytes: bytes) -> np.ndarray: ...

    def embed_texts(self, texts: list[str]) -> np.ndarray: ...


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return x / norm


class ClipEmbedder:
    """Real OpenCLIP ViT-B-32 (laion2b) embedder, CPU inference."""

    def __init__(self) -> None:
        import open_clip  # heavy import, kept local
        import torch

        self._torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            settings.clip_model_name, pretrained=settings.clip_pretrained
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(settings.clip_model_name)
        # ViT-B-32 embeds to 512 dims.
        self.dim = self.model.visual.output_dim

    def embed_image(self, image_bytes: bytes) -> np.ndarray:
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.preprocess(img).unsqueeze(0)
        with self._torch.no_grad():
            feats = self.model.encode_image(tensor)
        vec = feats.cpu().numpy()[0].astype("float32")
        return _l2_normalize(vec)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        tokens = self.tokenizer(texts)
        with self._torch.no_grad():
            feats = self.model.encode_text(tokens)
        arr = feats.cpu().numpy().astype("float32")
        return _l2_normalize(arr)


@lru_cache
def get_embedder() -> Embedder:
    """Process-wide singleton. First call pays the model-load cost (~1-3s)."""
    return ClipEmbedder()
