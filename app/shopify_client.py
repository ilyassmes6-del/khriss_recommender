"""Shopify Admin GraphQL client.

Pulls the full catalog with cursor pagination (250/page) and normalises each
product into a flat dict the indexer consumes. Only read_products scope is
required (it covers the variants' inventoryQuantity too).

Auth: apps created in the Dev Dashboard (the replacement for the legacy custom
apps retired in January 2026) are never handed a permanent token. The app's
client id and secret are exchanged for one that lives 24 hours, so tokens are
minted on demand and refreshed here rather than pasted into .env.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Iterator, Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# Re-mint this long before the stated expiry, so a request never rides a token
# that lapses mid-flight.
_REFRESH_MARGIN_SECONDS = 300

_PRODUCTS_QUERY = """
query Products($cursor: String) {
  products(first: 250, after: $cursor) {
    pageInfo { hasNextPage endCursor }
    nodes {
      id
      handle
      title
      productType
      tags
      images(first: 10) { nodes { url } }
      variants(first: 25) {
        nodes {
          id
          price
          inventoryQuantity
          availableForSale
        }
      }
    }
  }
}
"""

_SINGLE_PRODUCT_QUERY = """
query Product($id: ID!) {
  product(id: $id) {
    id
    handle
    title
    productType
    tags
    images(first: 10) { nodes { url } }
    variants(first: 25) {
      nodes { id price inventoryQuantity availableForSale }
    }
  }
}
"""


class AccessTokenProvider:
    """Mints and caches an Admin API token via the client credentials grant.

    The token is fetched lazily on first use -- a bad secret should surface as a
    failed request, not as an import-time crash in the indexer.
    """

    def __init__(self, client: Optional[httpx.Client] = None):
        self._client = client or httpx.Client(timeout=30.0)
        self._lock = threading.Lock()
        self._token: Optional[str] = None
        self._expires_at: float = 0.0

    def get(self, *, force_refresh: bool = False) -> str:
        # A legacy shpat_ token never expires, so there is nothing to exchange.
        if settings.shopify_admin_token:
            return settings.shopify_admin_token

        with self._lock:
            fresh = self._token and time.monotonic() < self._expires_at
            if fresh and not force_refresh:
                return self._token
            self._token = self._exchange()
            return self._token

    def _exchange(self) -> str:
        if not (settings.shopify_client_id and settings.shopify_client_secret):
            raise RuntimeError(
                "No Shopify credentials. Set SHOPIFY_CLIENT_ID and "
                "SHOPIFY_CLIENT_SECRET from the Dev Dashboard (app > Parametres "
                "> Identifiants), or SHOPIFY_ADMIN_TOKEN for a legacy custom app."
            )

        resp = self._client.post(
            f"https://{settings.shopify_shop}/admin/oauth/access_token",
            data={
                "grant_type": "client_credentials",
                "client_id": settings.shopify_client_id,
                "client_secret": settings.shopify_client_secret,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        token = data.get("access_token")
        if not token:
            raise RuntimeError(f"Token exchange returned no access_token: {data}")

        # An app with no access scopes published still mints a valid token --
        # one that is denied on every query. Say so here, where it is legible.
        if not data.get("scope"):
            logger.warning(
                "Shopify token carries no scopes: publish read_products on the "
                "app in the Dev Dashboard, otherwise every query is denied."
            )

        lifetime = int(data.get("expires_in", 86399))
        self._expires_at = time.monotonic() + max(lifetime - _REFRESH_MARGIN_SECONDS, 0)
        return token


class ShopifyClient:
    def __init__(self, client: Optional[httpx.Client] = None):
        self.endpoint = (
            f"https://{settings.shopify_shop}/admin/api/"
            f"{settings.shopify_api_version}/graphql.json"
        )
        self._client = client or httpx.Client(timeout=30.0)
        self._tokens = AccessTokenProvider(self._client)

    def _send(self, query: str, variables: dict, token: str) -> httpx.Response:
        return self._client.post(
            self.endpoint,
            headers={
                "X-Shopify-Access-Token": token,
                "Content-Type": "application/json",
            },
            json={"query": query, "variables": variables},
        )

    def _post(self, query: str, variables: dict) -> dict:
        resp = self._send(query, variables, self._tokens.get())
        if resp.status_code == 401:
            # Revoked, or expired sooner than advertised: one fresh mint, once.
            resp = self._send(query, variables, self._tokens.get(force_refresh=True))
        resp.raise_for_status()
        data = resp.json()
        if "errors" in data and data["errors"]:
            raise RuntimeError(f"Shopify GraphQL error: {data['errors']}")
        return data["data"]

    def iter_products(self) -> Iterator[dict]:
        """Yield normalised product dicts across all pages."""
        cursor: Optional[str] = None
        while True:
            data = self._post(_PRODUCTS_QUERY, {"cursor": cursor})
            block = data["products"]
            for node in block["nodes"]:
                yield normalize_product(node)
            if not block["pageInfo"]["hasNextPage"]:
                break
            cursor = block["pageInfo"]["endCursor"]

    def get_product(self, product_gid: str) -> Optional[dict]:
        data = self._post(_SINGLE_PRODUCT_QUERY, {"id": product_gid})
        node = data.get("product")
        return normalize_product(node) if node else None


def normalize_product(node: dict) -> dict:
    """Flatten a GraphQL product node into the shape the indexer expects."""
    images = [img["url"] for img in node.get("images", {}).get("nodes", [])]
    variants = node.get("variants", {}).get("nodes", [])

    # In stock if any variant is available for sale (or has positive inventory).
    in_stock = any(
        v.get("availableForSale") or (v.get("inventoryQuantity") or 0) > 0
        for v in variants
    )
    # Pick the first available variant for add-to-cart; fall back to first.
    chosen = next((v for v in variants if v.get("availableForSale")), None)
    if chosen is None and variants:
        chosen = variants[0]

    price = chosen.get("price") if chosen else None
    variant_id = _numeric_id(chosen["id"]) if chosen else None

    return {
        "product_id": _numeric_id(node["id"]),
        "handle": node["handle"],
        "title": node["title"],
        "product_type": node.get("productType"),
        "tags": node.get("tags", []),
        "images": images,
        "price": price,
        "variant_id": variant_id,
        "in_stock": in_stock,
    }


def _numeric_id(gid: str) -> str:
    """gid://shopify/Product/12345 -> "12345"."""
    return gid.rsplit("/", 1)[-1]
