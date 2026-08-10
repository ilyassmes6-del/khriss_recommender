"""Product categories, derived from Shopify's product_type.

Category is the one attribute we do *not* ask CLIP for. `product_type` is
merchant-entered ground truth -- a ring is a ring -- whereas zero-shot scoring
a handbag against a shoe vocabulary produces confident nonsense. Everything
downstream (which label schema to extract with, which slice of the index to
search, what "complete the look" is allowed to return) keys off this.

A store operator adapts the app to their catalog by editing PRODUCT_TYPE_MAP.
Unmapped types resolve to None and are skipped at index time rather than
guessed at: a mis-categorised product pollutes results in both directions.
"""
from __future__ import annotations

from typing import Optional

SHOES = "shoes"
BAGS = "bags"
JEWELRY = "jewelry"

CATEGORIES = (SHOES, BAGS, JEWELRY)

# Shopify product_type -> category. Keys are matched case-insensitively after
# trimming, so "SAC" and "Sac" both land on bags.
PRODUCT_TYPE_MAP: dict[str, str] = {
    # Chaussures
    "talons": SHOES,
    "plates": SHOES,
    "mocassin": SHOES,
    "mocassins": SHOES,
    "bottines": SHOES,
    "bottes": SHOES,
    "basket": SHOES,
    "baskets": SHOES,
    "sandales": SHOES,
    "mules": SHOES,
    "escarpins": SHOES,
    # Sacs
    "sac": BAGS,
    "sacs": BAGS,
    "sac a main": BAGS,
    "pochette": BAGS,
    # Bijoux
    "bagues": JEWELRY,
    "bague": JEWELRY,
    "bracelet": JEWELRY,
    "bracelets": JEWELRY,
    "collier": JEWELRY,
    "colliers": JEWELRY,
    "boucles d'oreilles": JEWELRY,
    "boucle d'oreille": JEWELRY,
}

# What the storefront calls each category, for stylist rationales and headings.
DISPLAY_FR: dict[str, str] = {
    SHOES: "chaussures",
    BAGS: "sac",
    JEWELRY: "bijou",
}


def from_product_type(product_type: Optional[str]) -> Optional[str]:
    """Map a Shopify product_type to a category, or None if unrecognised."""
    if not product_type:
        return None
    return PRODUCT_TYPE_MAP.get(product_type.strip().lower())
