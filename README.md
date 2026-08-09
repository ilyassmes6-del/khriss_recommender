# khriss — AI shoe recommendation for a Shopify shoe store

A shopper uploads a photo. khriss detects what it is and responds in one of two
modes:

- **Mode A — the photo is an OUTFIT** (a person, or clothing without shoes as the
  subject): it recommends in-stock shoes from your catalog that *complete the
  look*, each with a one-sentence stylist rationale.
- **Mode B — the photo is a SHOE**: it returns *visually similar* in-stock shoes,
  ranked by similarity.

No image generation anywhere. Every recommendation is one of your own product
photos, linked to its product page with add-to-cart.

---

## How it works (architecture)

```
upload ──▶ MODE ROUTING (OpenCLIP zero-shot: shoe vs outfit)
             │
   ┌─────────┴──────────┐
   ▼                    ▼
 SHOE                 OUTFIT
 (Mode B)             (Mode A)
   │                    │
 embed query          extract outfit attributes (CLIP zero-shot)
 cosine top-k         filter in-stock shoes: formality ±1, season, price
 in Qdrant           ~30 candidates ─▶ CLIP diversity prune
 dedupe per product   │
   │                  ▼
   │                RANK (one LLM call via OpenRouter — Haiku 4.5)
   │                  top 3 + rationale + coherence
   ▼                  ▼
        PRESENTATION (product cards, add-to-cart)
```

Key decisions, and why:

- **OpenCLIP `ViT-B-32` (`laion2b_s34b_b79k`)** is the single vision backbone for
  routing, attribute extraction, and similarity. One model, loaded once and kept
  warm, does everything. CPU inference is ~100 ms/image — no GPU required.
- **Qdrant holds one vector per product image.** A shoe shot from three angles
  becomes three vectors that all point back to the same `product_id`; we dedupe
  to one product at query time keeping its best-scoring vector. This makes
  similarity robust to camera angle without inflating the result list.
- **Postgres holds the human-readable product row + extracted attributes**, so
  Mode A can filter on formality/season/price with a plain SQL query and the
  ranker can read titles/prices without touching the vector store.
- **The LLM is used sparingly and only in Mode A.** Mode B is pure vector search:
  fast and free, no API call. Mode A makes exactly one LLM call — routed through
  **OpenRouter** (Claude Haiku 4.5 by default) — to rank a pre-filtered,
  pre-diversified shortlist, never to browse the whole catalog. Swapping the
  model is a one-line `RANKER_MODEL` change to any OpenRouter slug.
- **Everything tunable lives in config.** Label schemas are an editable module
  (`app/labels.py`); thresholds and weights are environment variables.

---

## Prerequisites

| You need | Where to get it |
|---|---|
| **Python 3.11** | Only if running the indexer/tests outside Docker. `python.org` or `pyenv`. Docker already ships 3.11. |
| **Docker + Docker Compose** | `docker.com`. The whole stack (API + Qdrant + Postgres) runs from `docker-compose.yml`. |
| **A Shopify app with `read_products` scope** | [Dev Dashboard](https://dev.shopify.com/dashboard) → create an app → **Versions** → put `read_products` in **Champs d'accès** → **Publier**. (Verified sufficient: it covers the variants' `inventoryQuantity` too, so no `read_inventory` needed.) Install it on your store, then copy the **Client ID** and **Client secret** from **Paramètres → Identifiants**. Shopify retired store-admin custom apps in January 2026: there is no `shpat_…` token to copy, and the app mints its own (see below). |
| **An OpenRouter API key** | `openrouter.ai/keys`. Only used for Mode A ranking (and optional vision extraction). Models are namespaced slugs like `anthropic/claude-haiku-4.5`. |

---

## Configuration (`.env`)

Copy `.env.example` to `.env` and fill it in. Every variable:

| Variable | Meaning |
|---|---|
| `SHOPIFY_SHOP` | Your myshopify domain, e.g. `my-shoes.myshopify.com` (no `https://`). |
| `SHOPIFY_CLIENT_ID` | Dev Dashboard app's Client ID (**Paramètres → Identifiants**). |
| `SHOPIFY_CLIENT_SECRET` | Dev Dashboard app's Client secret. Exchanged for a 24h Admin API token at runtime; `app/shopify_client.py` caches it and re-mints before it lapses, so nothing to rotate by hand. |
| `SHOPIFY_ADMIN_TOKEN` | Optional. A legacy custom app's permanent `shpat_…` token, which overrides the exchange. Leave blank unless you already had one. |
| `SHOPIFY_API_VERSION` | Admin API version, e.g. `2024-04`. Bump quarterly. |
| `SHOPIFY_WEBHOOK_SECRET` | Shared secret used to verify `products/update` webhook HMAC signatures. |
| `OPENROUTER_API_KEY` | OpenRouter key (`sk-or-…`) for the Mode A ranker (and optional vision extractor). |
| `OPENROUTER_BASE_URL` | OpenAI-compatible endpoint. Default `https://openrouter.ai/api/v1`. |
| `OPENROUTER_REFERER` / `OPENROUTER_TITLE` | Optional OpenRouter attribution headers. |
| `RANKER_MODEL` | OpenRouter model slug for the single Mode A ranking call. Default `anthropic/claude-haiku-4.5`. |
| `VISION_MODEL` | Optional separate slug for the vision extractor; blank reuses `RANKER_MODEL`. |
| `EXTRACTOR` | `clip` (free, zero-shot; default) or `llm` (OpenRouter vision, higher fidelity, ~$0.001/image). |
| `QDRANT_URL` | Qdrant endpoint. In Docker: `http://qdrant:6333`. |
| `QDRANT_API_KEY` | Qdrant API key (blank for the local container). |
| `DATABASE_URL` | Postgres DSN. In Docker: `postgresql+psycopg://khriss:khriss@postgres:5432/khriss`. |
| `ALLOWED_ORIGINS` | Comma-separated browser origins allowed by CORS — your storefront domain(s). |
| `MODE_CONFIDENCE_THRESHOLD` | If the shoe/outfit probability margin is below this, return **both** modes (tabs). Default `0.10`. |

Additional tuning knobs (sensible defaults in `app/config.py`, override via env):
`MODE_B_TOP_K`, `MODE_A_CANDIDATE_POOL`, `MODE_A_DIVERSITY_THRESHOLD`,
`MODE_A_FORMALITY_TOLERANCE`, `SAME_TYPE_BOOST`, `ROUTE_TEMPERATURE`.

---

## Run it

```bash
cp .env.example .env      # then edit .env
docker compose up --build
```

This starts Postgres, Qdrant, and the API on `http://localhost:8000`. The API
loads OpenCLIP once at startup (weights are baked into the image) and keeps it
warm. Check it's healthy:

```bash
curl -s localhost:8000/health | jq
```

### First full index

Run the indexer once to populate Qdrant + Postgres from your catalog:

```bash
docker compose run --rm api python indexer.py full
```

`full` clears the resume checkpoint and indexes everything; if it's interrupted,
re-run `python indexer.py incremental` and it picks up where it left off (the
checkpoint file lives on the mounted `./data` volume). **Roughly how long for
2,000 products?** CLIP inference is ~100 ms/image on CPU; add image download and
DB writes and you land around **0.3–0.5 s per product**, so a 2,000-product
catalog takes roughly **15–25 minutes** with `EXTRACTOR=clip`. `EXTRACTOR=llm`
adds an OpenRouter vision round-trip per product (~1 s each), so budget closer to
an hour.

### Build incrementally (recommended first-run sanity checks)

The system is designed to be validated stage by stage:

1. **Indexer first — eyeball attributes for 10 products (writes nothing):**
   ```bash
   docker compose run --rm api python indexer.py dump --limit 10
   ```
   Confirm the `type`, `material`, `dominant_colors`, `formality` look right for
   each shoe. Tune `app/labels.py` if not.
2. **Mode B — upload a shoe, check the matches make sense** (see curl below).
3. **Mode routing — upload an outfit vs a shoe**, confirm `mode` in the response.
4. **Mode A ranking — upload an outfit**, confirm 3 shoes come back with
   rationales.
5. **Theme extension last** — deploy the app block into the storefront.

---

## API

### `POST /recommend` (multipart image upload)

```bash
# Mode B — a shoe photo
curl -s -X POST localhost:8000/recommend \
  -F "image=@/path/to/shoe.jpg" | jq
```

Mode B response shape:

```json
{
  "mode": "shoe",
  "confidence": 0.94,
  "query_attributes": {
    "type": {"value": "boot", "confidence": 0.81},
    "material": {"value": "leather", "confidence": 0.63},
    "formality": 4,
    "dominant_colors": ["black"],
    "style_tags": ["classic"]
  },
  "results": [
    {
      "product_id": "123",
      "title": "Black Leather Chelsea Boot",
      "handle": "black-leather-chelsea-boot",
      "price": "180.00",
      "image_url": "https://cdn.shopify.com/....jpg",
      "variant_id": "456",
      "score": 0.92
    }
  ]
}
```

```bash
# Mode A — an outfit photo
curl -s -X POST localhost:8000/recommend \
  -F "image=@/path/to/outfit.jpg" | jq
```

Mode A response shape (note the per-shoe `rationale`):

```json
{
  "mode": "outfit",
  "confidence": 0.88,
  "query_attributes": {
    "silhouette": {"value": "tailored", "confidence": 0.7},
    "formality": 4,
    "season": {"value": "fall", "confidence": 0.6},
    "dominant_colors": ["navy", "white"],
    "garments_present": ["blazer", "trousers"],
    "occasion": {"value": "office", "confidence": 0.66}
  },
  "results": [
    {
      "product_id": "5",
      "title": "Black Leather Oxford",
      "handle": "black-leather-oxford",
      "price": "200.00",
      "image_url": "https://cdn.shopify.com/....jpg",
      "variant_id": "88",
      "score": 0.9,
      "rationale": "A sleek black oxford grounds the tailored navy suit for the office."
    }
  ]
}
```

**Ambiguous detection** (`mode: "both"`) adds two extra fields so the storefront
can render tabs:

```json
{
  "mode": "both",
  "confidence": 0.04,
  "query_attributes": { "...outfit attributes..." },
  "results": [ "...outfit_results (default tab)..." ],
  "shoe_results": [ "...Mode B grid..." ],
  "outfit_results": [ "...Mode A grid with rationales..." ]
}
```

Guards: 5 MB upload cap (`413`), non-image content types rejected (`415`), empty
upload (`400`), and `/recommend` is rate limited (30/min per IP). CORS is locked
to `ALLOWED_ORIGINS`.

### `POST /webhooks/products-update` (HMAC-verified)

Re-indexes a single product when it changes in Shopify. The body must be the raw
Shopify payload; the `X-Shopify-Hmac-Sha256` header is verified against
`SHOPIFY_WEBHOOK_SECRET`. Out-of-stock / deleted / image-less products are
dropped from the index.

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "vector_count": 5231,
  "product_count": 2000,
  "last_index_time": 1721145600.0,
  "extractor": "clip",
  "ranker_model": "anthropic/claude-haiku-4.5"
}
```

---

## Deploying to Railway

The storefront widget calls this API from the shopper's browser, so it has to be
publicly reachable over HTTPS — `localhost` won't do once the theme block is
live. Railway gives you that plus a managed Postgres.

**Three pieces:** the API (this Dockerfile), Postgres, and Qdrant.

### 1. Vectors: Qdrant Cloud

Take the [Qdrant Cloud](https://cloud.qdrant.io) free 1 GB cluster rather than
running Qdrant on Railway. Railway's private networking requires services to
bind on IPv6, which means extra Qdrant config to get right; the free tier holds
far more than this catalog needs and sidesteps it. Copy the cluster URL and API
key into `QDRANT_URL` / `QDRANT_API_KEY`.

### 2. API service

Point a new Railway service at this repo — it builds the Dockerfile as-is. The
container reads Railway's assigned `$PORT`, so no start-command override.

Add the Postgres plugin in the same project. Railway injects `DATABASE_URL` in
`postgresql://` form; `app/config.py` rewrites it to the `psycopg` 3 dialect
this image actually ships, so leave the injected value alone.

Then set:

| Variable | Value |
|---|---|
| `SHOPIFY_SHOP` | `your-store.myshopify.com` |
| `SHOPIFY_CLIENT_ID` / `SHOPIFY_CLIENT_SECRET` | From the Dev Dashboard app |
| `SHOPIFY_API_VERSION` | e.g. `2026-07` |
| `SHOPIFY_WEBHOOK_SECRET` | The app's API secret key |
| `OPENROUTER_API_KEY` | `sk-or-…` |
| `QDRANT_URL` / `QDRANT_API_KEY` | From Qdrant Cloud |
| `ALLOWED_ORIGINS` | Your storefront origin, e.g. `https://your-store.myshopify.com` |

`ALLOWED_ORIGINS` is the one people forget: get it wrong and the widget fails
with an opaque CORS error in the browser console, not a server-side log.

### 3. Index the catalog

Run the indexer *inside* the deployed container, which already has torch and the
CLIP weights:

```bash
railway ssh -s api -- python indexer.py full
```

Not `railway run` — that executes on your own machine with Railway's variables
injected, so it needs the whole ML stack installed locally. `railway ssh` is the
one that runs in the container.

Then confirm the service is live, and point the theme block's **khriss API base
URL** at the same host:

```bash
curl -s https://your-app.up.railway.app/health | jq
```

### Cost and sizing

Budget roughly **$10–20/month**. The driver is RAM, not traffic.

**Give the service at least 2 GB.** Measured against the real image, startup
alone (CPU torch + the OpenCLIP weights, before serving anything):

| Container limit | Result |
|---|---|
| 1000 MB | OOM killed |
| 1280 MB | OOM killed |
| 1536 MB | boots |
| 2048 MB | boots |

So ~1.5 GB is the hard floor just to reach `Uvicorn running`; 2 GB leaves room
for inference on top. Below it the container is killed part-way through loading
the model and the only evidence is a bare `Killed` in the logs, right after the
timm import warning, with no traceback — nothing names memory as the cause.

`OMP_NUM_THREADS=1` does **not** rescue a 1 GB container; it was tested and
still OOMs. Memory limits on Railway come from the plan attached to the
*workspace* that owns the project, so upgrading a personal account does nothing
for a project living in someone else's workspace.

Expect a multi-minute first build. Don't put this on anything that scales to
zero — every cold start reloads the model, and a shopper waits out that reload.

---

## Deploying the theme app extension

The storefront widget lives in `theme-extension/` as a Shopify **theme app
extension** (an app block).

The app project that deploys it is `khriss-app/`, scaffolded with
`npm init @shopify/app@latest` (extension-only template). It does **not** hold a
copy of the widget — `khriss-app/shopify.app.toml` points back out at this
directory:

```toml
extension_directories = [ "extensions/*", "../theme-extension" ]
```

So `theme-extension/` stays the single source of truth; edit it in place.

1. Build to run theme-check locally before pushing anything:
   ```bash
   cd khriss-app && npx shopify app build
   ```
2. Deploy it:
   ```bash
   cd khriss-app && npx shopify app deploy
   ```
3. In the Shopify admin, open **Online Store → Themes → Customize**.
4. On the page where you want the widget, **Add block → Apps → khriss
   recommender**.
5. In the block settings, set **khriss API base URL** to your deployed API
   (e.g. `https://recommender.yourdomain.com`, no trailing slash). Optionally
   edit the heading/subheading/button label.
6. Save. The widget renders an upload button with a client-side preview, then a
   results grid: Mode A shows rationales under each shoe, Mode B shows a plain
   similarity grid, and ambiguous detection renders both as tabs. "Add to cart"
   posts the `variant_id` to Shopify's `/cart/add.js`.

Make sure your storefront domain is in `ALLOWED_ORIGINS` or the browser will
block the request.

---

## Registering the `products/update` webhook

Point Shopify at your deployed API so single-product edits re-index automatically.

First mint a token (they last 24 hours, so do this immediately before the call):

```bash
SHOPIFY_ADMIN_TOKEN=$(curl -s -X POST "https://$SHOPIFY_SHOP/admin/oauth/access_token" -d "grant_type=client_credentials" -d "client_id=$SHOPIFY_CLIENT_ID" -d "client_secret=$SHOPIFY_CLIENT_SECRET" | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
```

Then register the webhook:

```bash
curl -X POST \
  "https://$SHOPIFY_SHOP/admin/api/$SHOPIFY_API_VERSION/webhooks.json" \
  -H "X-Shopify-Access-Token: $SHOPIFY_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "webhook": {
      "topic": "products/update",
      "address": "https://recommender.yourdomain.com/webhooks/products-update",
      "format": "json"
    }
  }'
```

The `SHOPIFY_WEBHOOK_SECRET` in your `.env` must match the store's webhook
signing secret (Shopify **Settings → Notifications → Webhooks** shows it). The
endpoint verifies every payload's HMAC and rejects mismatches with `401`.

---

## Running cost

Mode B is **free** — pure vector search, no LLM call. LLM cost is billed through
your OpenRouter account (OpenRouter passes through each model's list price) and
comes from two places:

| Lever | Knob | Cost |
|---|---|---|
| **Attribute extraction at index time** | `EXTRACTOR` | `clip` = $0 (runs locally). `llm` = ~$0.001/image via OpenRouter (Claude Haiku 4.5 ≈ $1 / MTok in, $5 / MTok out). |
| **Mode A ranking (per outfit request)** | `RANKER_MODEL` | One OpenRouter call over ~30 slim candidate records → typically well under $0.001/request. Point `RANKER_MODEL` at a larger slug for higher-quality rationales at higher cost. |

To minimise cost: keep `EXTRACTOR=clip` (extraction is free and re-run on every
re-index), keep `RANKER_MODEL=anthropic/claude-haiku-4.5`, and note that Mode B
traffic costs nothing regardless. To raise quality: set `EXTRACTOR=llm` for
sharper tags, and/or point `RANKER_MODEL` at a more capable OpenRouter model.

---

## Tuning

| What | Where | Effect |
|---|---|---|
| **Label sets** | `app/labels.py` (`SHOE_SCHEMA`, `SHOE_MULTI`, `OUTFIT_SCHEMA`, `OUTFIT_MULTI`, `*_TEMPLATES`) | The vocabulary the CLIP extractor scores against. Add types/materials/tags your catalog uses; adjust templates to phrasings CLIP recognises. Re-index after editing. |
| **Mode confidence threshold** | `MODE_CONFIDENCE_THRESHOLD` | Higher → more uploads treated as ambiguous (show both tabs). Lower → the router commits more often. Pair with `ROUTE_TEMPERATURE` (higher temperature = more decisive routing). |
| **Diversity threshold** | `MODE_A_DIVERSITY_THRESHOLD` | Cosine above which a Mode A candidate is dropped as a near-duplicate of one already selected. Lower → more aggressive de-duplication (avoids six colourways of one sneaker). |
| **Formality tolerance** | `MODE_A_FORMALITY_TOLERANCE` | ±N formality points a shoe may differ from the outfit. `0` = exact match only; `2` = looser. |
| **Same-type boost** | `SAME_TYPE_BOOST` | Mode B additive score bump when a candidate's shoe type matches the query's (a boot query surfacing boots first). `0` disables it. |
| **Result counts** | `MODE_B_TOP_K`, `MODE_A_CANDIDATE_POOL` | Size of the Mode B grid and the Mode A shortlist sent to the ranker. |

---

## Troubleshooting

- **Wrong mode detection.** If shoes are being read as outfits (or vice versa),
  raise `ROUTE_TEMPERATURE` to make routing more decisive, or nudge
  `MODE_CONFIDENCE_THRESHOLD`. If a whole category is consistently misrouted,
  the input crops probably include both a person and shoes — that's genuinely
  ambiguous, and the "both" tabs are the right answer.
- **Empty results.** Check `/health` — if `vector_count` is `0`, the index never
  ran (or ran against an empty catalog). Re-run `python indexer.py full`. In
  Mode A, over-tight filters starve the candidate pool: loosen
  `MODE_A_FORMALITY_TOLERANCE`, widen or drop the price band, and confirm your
  catalog actually has in-stock shoes at that formality/season.
- **JSON parse failures (ranker).** The ranker validates every reply against a
  Pydantic contract and retries once with a repair prompt; persistent failures
  return an empty Mode A result rather than garbage. If it happens often, your
  `RANKER_MODEL` may be too small — bump it. Hallucinated product IDs are always
  dropped (validated against the shortlist).
- **Stale index.** Register the `products/update` webhook (above) so edits
  re-index automatically, and re-run `python indexer.py incremental` after bulk
  catalog changes. `last_index_time` on `/health` tells you when a product was
  last re-indexed via webhook.
- **Cold start.** The first request after boot waits for OpenCLIP to load
  (~1–3 s). Weights are baked into the Docker image and the model is warmed at
  startup, so subsequent requests are ~100 ms. If `model_loaded` is `false` on
  `/health`, startup hasn't finished.

---

## Tests

Fully offline — mocked Shopify/OpenRouter clients, a fake CLIP embedder over a
toy concept vocabulary, and an in-memory vector store, so no network or GPU is
needed:

```bash
pip install -r requirements.txt   # or just the non-ML deps for the fast path
pytest
```

The suite covers the indexer (a ~20-shoe fixture catalog across types), Mode B
retrieval, mode routing, Mode A filtering/diversity, the ranker's
JSON/hallucination/repair contract, the end-to-end pipeline, and the HTTP
surface (upload guards + webhook HMAC).
