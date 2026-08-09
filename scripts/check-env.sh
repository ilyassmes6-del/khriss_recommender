#!/usr/bin/env bash
# Smoke-test the credentials in .env against the real services.
#
# Every check is read-only and costs nothing: no tokens are spent, no rows
# written, no collections created. Values are never echoed -- only PASS/FAIL,
# HTTP codes, and the first 4 characters of a secret when identification helps.
#
# Note: docker-compose.yml overrides QDRANT_URL and DATABASE_URL for the `api`
# service (to the compose service names), so those two .env values only apply
# when you run the app directly on the host.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[[ -f .env ]] || { echo "error: .env not found in $ROOT (run scripts/setup-env.sh)" >&2; exit 1; }

# Load .env without executing it: export KEY=value lines, ignore comments/blanks.
# Read in a loop rather than `source <(...)`: macOS ships bash 3.2, where
# sourcing a process substitution silently yields nothing.
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ "$line" =~ ^[A-Z_]+= ]] || continue
  export "${line%%=*}=${line#*=}"
done < .env

TMPD="$(mktemp -d)"
trap 'rm -rf "$TMPD"' EXIT

PASS=0; FAIL=0; SKIP=0
ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; PASS=$((PASS+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAIL=$((FAIL+1)); }
skip() { printf '  \033[33mSKIP\033[0m  %s\n' "$1"; SKIP=$((SKIP+1)); }
mask() { local v="${1:-}"; [[ -z "$v" ]] && { echo "(unset)"; return; }; echo "${v:0:4}..."; }

# unset VAR -> report as SKIP (not configured yet) rather than a failure.
is_placeholder() {
  local v="${1:-}"
  [[ -z "$v" ]] && return 0
  [[ "$v" == *xxxx* || "$v" == your-* || "$v" == *your-store* ]] && return 0
  return 1
}

echo
echo "=== Shopify Admin API ==="
# Dev Dashboard apps hold no permanent token: exchange the client id/secret for
# a 24h one, exactly as app/shopify_client.py does at runtime. A legacy shpat_
# token in SHOPIFY_ADMIN_TOKEN still wins, so honour it when present.
TOKEN="${SHOPIFY_ADMIN_TOKEN:-}"
if is_placeholder "$TOKEN"; then
  TOKEN=""
  if is_placeholder "${SHOPIFY_CLIENT_ID:-}" || is_placeholder "${SHOPIFY_CLIENT_SECRET:-}"; then
    skip "SHOPIFY_CLIENT_ID / SHOPIFY_CLIENT_SECRET not set (Dev Dashboard > app > Parametres)"
  else
    tcode=$(curl -s -o "$TMPD"/tok.json -w '%{http_code}' --max-time 20 \
      -X POST "https://${SHOPIFY_SHOP:-}/admin/oauth/access_token" \
      -H "Content-Type: application/x-www-form-urlencoded" \
      -d "grant_type=client_credentials" \
      -d "client_id=${SHOPIFY_CLIENT_ID}" \
      -d "client_secret=${SHOPIFY_CLIENT_SECRET}")
    if [[ "$tcode" == "200" ]]; then
      TOKEN=$(sed -n 's/.*"access_token" *: *"\([^"]*\)".*/\1/p' "$TMPD"/tok.json)
      GRANTED=$(sed -n 's/.*"scope" *: *"\([^"]*\)".*/\1/p' "$TMPD"/tok.json)
      ok "token exchange -> 200 (24h token minted for ${SHOPIFY_SHOP:-})"
      if [[ -z "$GRANTED" ]]; then
        bad "token has no scopes -- publish read_products under 'Champs d'acces' in the Dev Dashboard"
      else
        ok "granted scopes: $GRANTED"
      fi
    else
      bad "token exchange -> $tcode (client id/secret rejected, or app not installed on ${SHOPIFY_SHOP:-})"
    fi
  fi
fi

if [[ -z "$TOKEN" ]]; then
  skip "no Shopify token to test with"
else
  code=$(curl -s -o "$TMPD"/shop.json -w '%{http_code}' --max-time 20 \
    -H "X-Shopify-Access-Token: ${TOKEN}" \
    "https://${SHOPIFY_SHOP:-}/admin/api/${SHOPIFY_API_VERSION:-}/shop.json")
  case "$code" in
    200) ok "shop.json -> 200 (token $(mask "$TOKEN") valid for ${SHOPIFY_SHOP:-})" ;;
    401|403) bad "shop.json -> $code (token $(mask "$TOKEN") rejected)" ;;
    404) bad "shop.json -> 404 (bad shop domain or API version ${SHOPIFY_API_VERSION:-})" ;;
    *)   bad "shop.json -> $code" ;;
  esac

  # Probe the exact GraphQL selection the indexer uses. A REST 200 proves
  # nothing here: a denied field comes back as a 200 carrying an "errors" key.
  if [[ "$code" == "200" ]]; then
    curl -s -o "$TMPD"/gql.json --max-time 20 \
      -X POST "https://${SHOPIFY_SHOP:-}/admin/api/${SHOPIFY_API_VERSION:-}/graphql.json" \
      -H "X-Shopify-Access-Token: ${TOKEN}" \
      -H "Content-Type: application/json" \
      -d '{"query":"{ products(first:1){ nodes { id variants(first:1){ nodes { availableForSale inventoryQuantity } } } } }"}'
    if grep -q '"errors"' "$TMPD"/gql.json 2>/dev/null; then
      bad "graphql products query denied: $(tr -d '\n' < "$TMPD"/gql.json | cut -c1-160)"
    else
      ok "graphql products+variants query -> granted (read_products, read_inventory)"
    fi
  fi
fi

if is_placeholder "${SHOPIFY_WEBHOOK_SECRET:-}"; then
  skip "SHOPIFY_WEBHOOK_SECRET not set (webhook HMAC verification will reject everything)"
else
  printf '  \033[36mINFO\033[0m  %s\n' "SHOPIFY_WEBHOOK_SECRET present $(mask "$SHOPIFY_WEBHOOK_SECRET") (correctness only provable on a live webhook)"
fi

echo
echo "=== OpenRouter ==="
if is_placeholder "${OPENROUTER_API_KEY:-}"; then
  skip "OPENROUTER_API_KEY not set"
else
  code=$(curl -s -o "$TMPD"/or.json -w '%{http_code}' --max-time 20 \
    -H "Authorization: Bearer ${OPENROUTER_API_KEY}" \
    "${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}/key")
  [[ "$code" == "200" ]] && ok "/key -> 200 (key $(mask "$OPENROUTER_API_KEY") valid)" \
                         || bad "/key -> $code (key rejected)"

  # Confirm the configured ranker slug actually resolves.
  if [[ "$code" == "200" ]]; then
    if curl -s --max-time 20 "${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}/models" \
         | grep -q "\"${RANKER_MODEL:-}\""; then
      ok "RANKER_MODEL '${RANKER_MODEL:-}' resolves"
    else
      bad "RANKER_MODEL '${RANKER_MODEL:-}' not found in /models"
    fi
  fi
fi

echo
echo "=== Qdrant ==="
if [[ -z "${QDRANT_URL:-}" ]]; then
  skip "QDRANT_URL not set"
else
  code=$(curl -s -o "$TMPD"/qd.json -w '%{http_code}' --max-time 20 \
    -H "api-key: ${QDRANT_API_KEY:-}" "${QDRANT_URL}/collections")
  case "$code" in
    200)
      ok "/collections -> 200 (reachable, key $(mask "${QDRANT_API_KEY:-}") accepted)"
      if grep -q "\"${QDRANT_COLLECTION:-shoes}\"" "$TMPD"/qd.json 2>/dev/null; then
        ok "collection '${QDRANT_COLLECTION:-shoes}' exists"
      else
        skip "collection '${QDRANT_COLLECTION:-shoes}' not created yet (the indexer creates it)"
      fi ;;
    401|403) bad "/collections -> $code (QDRANT_API_KEY rejected)" ;;
    000)     bad "/collections -> unreachable (${QDRANT_URL})" ;;
    *)       bad "/collections -> $code" ;;
  esac
fi

echo
echo "=== Postgres ==="
# SQLAlchemy's dialect suffix is not valid libpq URI syntax; psql needs it gone.
PG_DSN="${DATABASE_URL:-}"
PG_DSN="${PG_DSN/+psycopg/}"
PG_HOST=$(echo "$PG_DSN" | sed -E 's|.*@([^:/]+).*|\1|')
if ! command -v psql >/dev/null 2>&1; then
  skip "psql not installed"
elif [[ "$PG_HOST" == "postgres" ]]; then
  # Compose service name: not resolvable from the host, and compose maps no port.
  if docker compose ps postgres 2>/dev/null | grep -q running; then
    if docker compose exec -T postgres psql -U khriss -d khriss -c 'select 1' >/dev/null 2>&1; then
      ok "reachable via 'docker compose exec postgres' (host '$PG_HOST' is compose-only)"
    else
      bad "container is up but psql failed inside it"
    fi
  else
    skip "host '$PG_HOST' is a compose service name and the stack isn't running (start with 'docker compose up -d postgres')"
  fi
else
  if psql "$PG_DSN" -c 'select 1' >/dev/null 2>&1; then
    ok "connected to $PG_HOST"
  else
    bad "could not connect to $PG_HOST"
  fi
fi

echo
echo "=== Serving ==="
if is_placeholder "${ALLOWED_ORIGINS:-}"; then
  skip "ALLOWED_ORIGINS still a placeholder (browser calls from your storefront will be CORS-blocked)"
else
  ok "ALLOWED_ORIGINS=${ALLOWED_ORIGINS}"
fi

echo
echo "-- $PASS passed, $FAIL failed, $SKIP skipped --"
[[ "$FAIL" -eq 0 ]]
