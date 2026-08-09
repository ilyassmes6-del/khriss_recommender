"""FastAPI service.

Endpoints:
  POST /recommend               multipart image upload -> recommendations
  POST /webhooks/products-update HMAC-verified single-product re-index
  GET  /health                  index size, model loaded, last index time

CORS is locked to ALLOWED_ORIGINS; /recommend is rate limited and capped at 5MB.

Deliberately no `from __future__ import annotations` here: slowapi's @limiter
wrapper carries its own module globals, so postponed (string) annotations leave
FastAPI unable to resolve 'UploadFile' and the app dies at import. Python 3.11
evaluates the annotations in this file natively, so the import buys us nothing.
"""
import json
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app import db, pipeline
from app.config import settings
from app.qdrant_store import get_store
from app.shopify_client import ShopifyClient
from app.webhooks import verify_hmac

_ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
_state = {"model_loaded": False, "last_index_time": None}

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    pipeline.warm_up()  # load CLIP once, keep warm
    _state["model_loaded"] = True
    yield


app = FastAPI(title="khriss", version="0.1.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    lambda request, exc: JSONResponse(status_code=429, content={"detail": "rate limit exceeded"}),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    store = get_store()
    with db.get_session() as session:
        product_count = db.count_products(session)
    return {
        "status": "ok",
        "model_loaded": _state["model_loaded"],
        "vector_count": store.count(),
        "product_count": product_count,
        "last_index_time": _state["last_index_time"],
        "extractor": settings.extractor,
        "ranker_model": settings.ranker_model,
    }


@app.post("/recommend")
@limiter.limit("30/minute")
async def recommend(request: Request, image: UploadFile = File(...)):
    if image.content_type not in _ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="unsupported media type")

    data = await image.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="image exceeds 5MB cap")
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")

    try:
        result = pipeline.recommend(data)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"recommendation failed: {exc}")
    return result.model_dump(exclude_none=True)


@app.post("/webhooks/products-update")
async def products_update(
    request: Request,
    background: BackgroundTasks,
    x_shopify_hmac_sha256: Optional[str] = Header(default=None),
):
    raw = await request.body()
    if not verify_hmac(raw, x_shopify_hmac_sha256):
        raise HTTPException(status_code=401, detail="invalid hmac")

    payload = json.loads(raw)
    gid = payload.get("admin_graphql_api_id")  # gid://shopify/Product/123
    if not gid:
        pid = payload.get("id")
        gid = f"gid://shopify/Product/{pid}" if pid else None
    if not gid:
        raise HTTPException(status_code=400, detail="no product id in payload")

    background.add_task(_reindex_one, gid)
    return {"status": "accepted"}


def _reindex_one(product_gid: str) -> None:
    from app.indexing import Indexer  # local import: heavy deps

    client = ShopifyClient()
    product = client.get_product(product_gid)
    indexer = Indexer()
    pid = product_gid.rsplit("/", 1)[-1]
    if product is None or not product.get("in_stock") or not product.get("images"):
        # Product went out of stock / deleted / lost its image -> drop it.
        indexer.delete_product(pid)
    else:
        indexer.index_product(product)
    _state["last_index_time"] = time.time()
