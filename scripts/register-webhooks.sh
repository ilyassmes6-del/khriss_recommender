#!/usr/bin/env bash
# Register the product webhooks that keep the index current.
#
# All three topics point at the same endpoint: _reindex_one() in app/main.py is
# topic-agnostic -- it fetches the product by id and either indexes it or drops
# it (gone, out of stock, or no image). So create/update/delete need no
# separate handling.
#
# Idempotent: existing subscriptions for the same topic+address are left alone.
#
# Usage: scripts/register-webhooks.sh https://your-api.up.railway.app
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

API_BASE="${1:-}"
[[ -n "$API_BASE" ]] || { echo "usage: $0 https://your-api-host" >&2; exit 1; }
ADDRESS="${API_BASE%/}/webhooks/products-update"

[[ -f .env ]] || { echo "error: .env not found in $ROOT" >&2; exit 1; }
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ "$line" =~ ^[A-Z_]+= ]] || continue
  export "${line%%=*}=${line#*=}"
done < .env

: "${SHOPIFY_SHOP:?set SHOPIFY_SHOP in .env}"
: "${SHOPIFY_API_VERSION:?set SHOPIFY_API_VERSION in .env}"
: "${SHOPIFY_CLIENT_ID:?set SHOPIFY_CLIENT_ID in .env}"
: "${SHOPIFY_CLIENT_SECRET:?set SHOPIFY_CLIENT_SECRET in .env}"

echo "shop:    $SHOPIFY_SHOP"
echo "address: $ADDRESS"
echo

# Dev Dashboard apps hold no permanent token; mint a 24h one.
TOKEN=$(curl -s -X POST "https://${SHOPIFY_SHOP}/admin/oauth/access_token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials" \
  -d "client_id=${SHOPIFY_CLIENT_ID}" \
  -d "client_secret=${SHOPIFY_CLIENT_SECRET}" \
  | sed -n 's/.*"access_token" *: *"\([^"]*\)".*/\1/p')

[[ -n "$TOKEN" ]] || { echo "error: token exchange failed (check client id/secret, and that the app is installed)" >&2; exit 1; }

EXISTING=$(curl -s -H "X-Shopify-Access-Token: $TOKEN" \
  "https://${SHOPIFY_SHOP}/admin/api/${SHOPIFY_API_VERSION}/webhooks.json?limit=250")

for TOPIC in products/create products/update products/delete; do
  # Match topic and address together: the same topic pointed elsewhere is a
  # different subscription and should not suppress this one.
  if echo "$EXISTING" | tr '}' '}\n' | grep -F "\"topic\":\"$TOPIC\"" | grep -qF "$ADDRESS"; then
    printf '  SKIP  %-18s already registered\n' "$TOPIC"
    continue
  fi

  BODY=$(printf '{"webhook":{"topic":"%s","address":"%s","format":"json"}}' "$TOPIC" "$ADDRESS")
  RESP=$(curl -s -w '\n%{http_code}' -X POST \
    "https://${SHOPIFY_SHOP}/admin/api/${SHOPIFY_API_VERSION}/webhooks.json" \
    -H "X-Shopify-Access-Token: $TOKEN" \
    -H "Content-Type: application/json" \
    -d "$BODY")
  CODE=$(echo "$RESP" | tail -1)

  case "$CODE" in
    201) printf '  OK    %-18s registered\n' "$TOPIC" ;;
    422) printf '  WARN  %-18s rejected: %s\n' "$TOPIC" "$(echo "$RESP" | sed '$d' | head -c 200)" ;;
    *)   printf '  FAIL  %-18s http %s: %s\n' "$TOPIC" "$CODE" "$(echo "$RESP" | sed '$d' | head -c 200)" ;;
  esac
done

echo
echo "Set SHOPIFY_WEBHOOK_SECRET to the app's client secret -- app-created"
echo "webhooks are signed with it, and verify_hmac rejects everything without it."
