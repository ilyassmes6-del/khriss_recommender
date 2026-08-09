"""Token lifecycle for the client credentials grant.

Dev Dashboard tokens live 24 hours, so the failure these cover is silent: a
long-running indexer that keeps presenting a token which lapsed hours ago.
"""
from __future__ import annotations

import httpx
import pytest

from app import shopify_client
from app.shopify_client import AccessTokenProvider


@pytest.fixture
def creds(monkeypatch):
    monkeypatch.setattr(shopify_client.settings, "shopify_admin_token", "")
    monkeypatch.setattr(shopify_client.settings, "shopify_client_id", "cid")
    monkeypatch.setattr(shopify_client.settings, "shopify_client_secret", "csec")
    monkeypatch.setattr(shopify_client.settings, "shopify_shop", "s.myshopify.com")


def _provider(handler):
    """AccessTokenProvider wired to a scripted transport."""
    return AccessTokenProvider(httpx.Client(transport=httpx.MockTransport(handler)))


def _minted(counter, scope="read_products", expires_in=86399):
    def handler(request):
        counter.append(request)
        return httpx.Response(
            200,
            json={
                "access_token": f"tok-{len(counter)}",
                "scope": scope,
                "expires_in": expires_in,
            },
        )

    return handler


def test_token_is_cached_between_calls(creds):
    calls = []
    provider = _provider(_minted(calls))

    assert provider.get() == "tok-1"
    assert provider.get() == "tok-1"
    assert len(calls) == 1, "a cached token must not be re-exchanged"


def test_token_is_reminted_once_it_goes_stale(creds, monkeypatch):
    calls = []
    provider = _provider(_minted(calls))

    now = 1_000.0
    monkeypatch.setattr(shopify_client.time, "monotonic", lambda: now)
    assert provider.get() == "tok-1"

    # Past expiry minus the refresh margin: the next call must mint again.
    now = 1_000.0 + 86399
    assert provider.get() == "tok-2"
    assert len(calls) == 2


def test_refresh_margin_beats_the_stated_expiry(creds, monkeypatch):
    """Refresh early, so no request rides a token that lapses mid-flight."""
    calls = []
    provider = _provider(_minted(calls, expires_in=1_000))

    now = 0.0
    monkeypatch.setattr(shopify_client.time, "monotonic", lambda: now)
    provider.get()

    now = 1_000 - shopify_client._REFRESH_MARGIN_SECONDS + 1
    provider.get()
    assert len(calls) == 2, "token should be replaced before it actually expires"


def test_force_refresh_bypasses_a_still_valid_cache(creds):
    calls = []
    provider = _provider(_minted(calls))

    assert provider.get() == "tok-1"
    assert provider.get(force_refresh=True) == "tok-2"


def test_legacy_static_token_skips_the_exchange(creds, monkeypatch):
    monkeypatch.setattr(shopify_client.settings, "shopify_admin_token", "shpat_legacy")
    calls = []
    provider = _provider(_minted(calls))

    assert provider.get() == "shpat_legacy"
    assert calls == [], "a permanent token has nothing to exchange"


def test_missing_credentials_name_both_variables(creds, monkeypatch):
    monkeypatch.setattr(shopify_client.settings, "shopify_client_secret", "")
    provider = _provider(_minted([]))

    with pytest.raises(RuntimeError, match="SHOPIFY_CLIENT_SECRET"):
        provider.get()


def test_scopeless_token_is_flagged(creds, caplog):
    """An app with no published scopes mints a token denied on every query."""
    provider = _provider(_minted([], scope=""))

    with caplog.at_level("WARNING"):
        provider.get()

    assert "no scopes" in caplog.text


def test_expired_token_is_retried_once(creds):
    """A 401 mid-flight should re-mint and replay, not surface to the caller."""
    seen = []

    def handler(request):
        if request.url.path.endswith("/access_token"):
            seen.append("mint")
            return httpx.Response(
                200,
                json={
                    "access_token": f"tok-{seen.count('mint')}",
                    "scope": "read_products",
                    "expires_in": 86399,
                },
            )
        seen.append(request.headers["X-Shopify-Access-Token"])
        if seen.count("tok-1") == 1 and "tok-2" not in seen:
            return httpx.Response(401, json={"errors": "expired"})
        return httpx.Response(200, json={"data": {"product": None}})

    client = shopify_client.ShopifyClient(
        httpx.Client(transport=httpx.MockTransport(handler))
    )
    assert client.get_product("gid://shopify/Product/1") is None
    assert "tok-2" in seen, "the 401 should have triggered a fresh token"
