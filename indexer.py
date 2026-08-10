#!/usr/bin/env python3
"""khriss catalog indexer CLI.

Usage
-----
  python indexer.py full          # index the whole catalog (resume-safe)
  python indexer.py incremental   # only products not already in the checkpoint
  python indexer.py dump --limit 10   # extract + print attributes for N products,
                                       # write nothing (eyeball the extractor)
  python indexer.py reset         # clear the resume checkpoint

`full` clears the checkpoint first; `incremental` keeps it and skips done IDs.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

from tqdm import tqdm

from app import db
from app.config import settings
from app.indexing import Indexer, clear_checkpoint
from app.shopify_client import ShopifyClient


def _run_index(resume: bool) -> None:
    db.init_db()
    shop = ShopifyClient()
    indexer = Indexer()
    indexer.store.ensure_collection(dim=indexer.embedder.dim)

    bar = tqdm(desc="indexing", unit="product")

    def progress(_product: dict, ok: bool) -> None:
        bar.update(1)

    start = time.time()
    # Only a full pass sees the whole feed, so only a full pass may delete.
    stats = indexer.run(
        shop.iter_products(), resume=resume, progress=progress, prune=not resume
    )
    bar.close()
    elapsed = time.time() - start
    print(
        f"done in {elapsed:.1f}s | indexed={stats.indexed} "
        f"skipped={stats.skipped_no_image} failed={stats.failed} "
        f"pruned={stats.pruned}"
    )


def _dump(limit: int) -> None:
    """Extract attributes for the first N products and print them. No writes."""
    from app.extractor import get_extractor

    shop = ShopifyClient()
    extractor = get_extractor()
    import httpx

    http = httpx.Client(timeout=30.0, follow_redirects=True)

    shown = 0
    for product in itertools.islice(
        (p for p in shop.iter_products() if p.get("images")), limit
    ):
        img = http.get(product["images"][0]).content
        attrs = extractor.extract_shoe(img)
        print(
            json.dumps(
                {
                    "product_id": product["product_id"],
                    "title": product["title"],
                    "in_stock": product["in_stock"],
                    "attributes": attrs.model_dump(mode="json"),
                },
                indent=2,
            )
        )
        shown += 1
    print(f"\ndumped {shown} products (extractor={settings.extractor})")


def main() -> None:
    parser = argparse.ArgumentParser(description="khriss catalog indexer")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("full", help="index the whole catalog (clears checkpoint)")
    sub.add_parser("incremental", help="index only products not yet done")
    d = sub.add_parser("dump", help="print extracted attributes, write nothing")
    d.add_argument("--limit", type=int, default=10)
    sub.add_parser("reset", help="clear the resume checkpoint")

    args = parser.parse_args()
    if args.cmd == "full":
        clear_checkpoint()
        _run_index(resume=False)
    elif args.cmd == "incremental":
        _run_index(resume=True)
    elif args.cmd == "dump":
        _dump(args.limit)
    elif args.cmd == "reset":
        clear_checkpoint()
        print("checkpoint cleared")


if __name__ == "__main__":
    main()
