"""OpenRouter chat client (OpenAI-compatible).

All LLM calls in khriss — the single Mode A ranking call, and the optional
vision attribute extractor — go through OpenRouter using an OpenRouter API key.
OpenRouter serves an OpenAI-compatible `/chat/completions` endpoint, so we speak
that shape over httpx (already a project dependency). The default model is Claude
Haiku 4.5 via its OpenRouter slug, but RANKER_MODEL / VISION_MODEL accept any
OpenRouter model.
"""
from __future__ import annotations

import base64
from functools import lru_cache
from typing import Optional

import httpx

from app.config import settings


class LLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        http: Optional[httpx.Client] = None,
    ):
        self.api_key = api_key if api_key is not None else settings.openrouter_api_key
        self.base_url = (base_url or settings.openrouter_base_url).rstrip("/")
        self._http = http or httpx.Client(timeout=60.0)

    def chat(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = 700,
        temperature: float = 0.2,
    ) -> str:
        """Return the assistant text for an OpenAI-style messages list."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # OpenRouter optionally attributes traffic via these headers.
        if settings.openrouter_referer:
            headers["HTTP-Referer"] = settings.openrouter_referer
        if settings.openrouter_title:
            headers["X-Title"] = settings.openrouter_title

        resp = self._http.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"] or ""


def text_message(role: str, text: str) -> dict:
    return {"role": role, "content": text}


def image_message(text: str, image_bytes: bytes) -> dict:
    """An OpenAI-style user message carrying a text part + an inline image."""
    media = _media_type(image_bytes)
    b64 = base64.standard_b64encode(image_bytes).decode()
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media};base64,{b64}"},
            },
        ],
    }


def _media_type(b: bytes) -> str:
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


@lru_cache
def get_llm() -> LLMClient:
    return LLMClient()
