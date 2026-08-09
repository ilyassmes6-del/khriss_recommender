"""Shopify webhook HMAC verification."""
from __future__ import annotations

import base64
import hashlib
import hmac

from app.config import settings


def verify_hmac(raw_body: bytes, header_hmac: str | None) -> bool:
    """Validate the X-Shopify-Hmac-Sha256 header against the raw request body.

    Uses a constant-time comparison. `raw_body` MUST be the exact bytes Shopify
    sent — do not re-serialise the JSON first.
    """
    if not header_hmac or not settings.shopify_webhook_secret:
        return False
    digest = hmac.new(
        settings.shopify_webhook_secret.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).digest()
    expected = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected, header_hmac)
