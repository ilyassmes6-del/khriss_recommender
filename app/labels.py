"""Editable label schemas for zero-shot attribute extraction.

This is the single place a store operator tunes what the CLIP extractor looks
for. Each axis is a list of candidate labels; the extractor embeds a templated
prompt per label and keeps the highest-cosine label per axis (plus confidence).

Design notes
------------
* Axes are grouped into two schemas: SHOE (for products and Mode B queries) and
  OUTFIT (for Mode A queries).
* `formality` and `season` deliberately overlap between the two schemas so Mode A
  filtering can compare an outfit's formality/season against a shoe's directly.
* Some axes are "multi-label" (dominant_colors, style_tags, garments_present):
  we keep the top-N above a floor instead of a single winner.
* The templates turn a bare label into a natural sentence; CLIP was trained on
  captions, so "a photo of a leather shoe" scores far more reliably than
  "leather".
"""
from __future__ import annotations

# Formality is scored on a 1..5 integer scale via descriptive anchors so the
# same axis is comparable across shoes and outfits.
FORMALITY_ANCHORS: dict[int, str] = {
    1: "very casual, athletic or beachwear",
    2: "casual everyday wear",
    3: "smart casual, business casual",
    4: "formal, business or dressy",
    5: "black tie, very formal eveningwear",
}

SEASONS = ["spring", "summer", "fall", "winter", "all-season"]

COLORS = [
    "black", "white", "grey", "brown", "tan", "beige", "navy", "blue",
    "red", "burgundy", "green", "olive", "pink", "purple", "yellow",
    "orange", "gold", "silver", "multicolor",
]

# ---------------------------------------------------------------------------
# SHOE schema
# ---------------------------------------------------------------------------
SHOE_SCHEMA: dict[str, list[str]] = {
    "type": [
        "sneaker", "running shoe", "boot", "ankle boot", "loafer", "heel",
        "sandal", "oxford", "derby", "mule", "flat", "espadrille", "clog",
        "slipper",
    ],
    "material": [
        "leather", "suede", "canvas", "mesh", "knit", "synthetic",
        "patent leather", "nubuck", "rubber", "denim",
    ],
    "sole_type": [
        "rubber sole", "leather sole", "platform sole", "lug sole",
        "foam sole", "cork sole", "espadrille sole",
    ],
    "heel_height": ["flat", "low heel", "mid heel", "high heel"],
    "toe_shape": ["round toe", "pointed toe", "almond toe", "square toe", "open toe"],
    "pattern": [
        "solid color", "animal print", "floral print", "striped",
        "camouflage", "metallic", "colorblock",
    ],
    "season": SEASONS,
}

# Multi-label axes for shoes: keep several winners above the floor.
SHOE_MULTI: dict[str, list[str]] = {
    "dominant_colors": COLORS,
    "style_tags": [
        "minimalist", "chunky", "retro", "sporty", "elegant", "streetwear",
        "classic", "trendy", "rugged", "preppy", "bohemian", "edgy",
    ],
}

# ---------------------------------------------------------------------------
# OUTFIT schema
# ---------------------------------------------------------------------------
OUTFIT_SCHEMA: dict[str, list[str]] = {
    "silhouette": [
        "oversized", "fitted", "tailored", "relaxed", "flowy", "structured",
    ],
    "pattern": [
        "solid color", "animal print", "floral print", "striped", "plaid",
        "camouflage", "colorblock",
    ],
    "season": SEASONS,
    "occasion": [
        "everyday casual", "office", "date night", "formal event", "workout",
        "beach or vacation", "party", "outdoor",
    ],
}

OUTFIT_MULTI: dict[str, list[str]] = {
    "dominant_colors": COLORS,
    "garments_present": [
        "t-shirt", "shirt", "blouse", "sweater", "hoodie", "jacket", "blazer",
        "coat", "dress", "skirt", "jeans", "trousers", "shorts", "leggings",
        "suit", "activewear",
    ],
    "style_tags": [
        "minimalist", "streetwear", "elegant", "sporty", "classic", "trendy",
        "bohemian", "preppy", "edgy", "business",
    ],
}

# ---------------------------------------------------------------------------
# Prompt templates. `{label}` is filled with each candidate.
# ---------------------------------------------------------------------------
SHOE_TEMPLATES = [
    "a photo of a {label} shoe",
    "a close-up product photo of {label} footwear",
    "{label}",
]

OUTFIT_TEMPLATES = [
    "a photo of a person wearing {label}",
    "a photo of {label} clothing",
    "{label}",
]

# Confidence floor for multi-label axes (cosine on L2-normalised embeddings).
MULTI_LABEL_FLOOR = 0.20
MULTI_LABEL_MAX = 3


# ---------------------------------------------------------------------------
# BAG schema
# ---------------------------------------------------------------------------
# Scoring a handbag against the shoe vocabulary produced confident nonsense
# ("this tote is a mule"), which then drove retrieval. Each category gets the
# axes that actually describe it.
BAG_SCHEMA: dict[str, list[str]] = {
    "type": [
        "tote bag", "shoulder bag", "crossbody bag", "clutch", "handbag",
        "backpack", "bucket bag", "baguette bag", "satchel", "hobo bag",
    ],
    "material": [
        "leather", "suede", "patent leather", "canvas", "raffia", "straw",
        "nylon", "velvet", "denim", "synthetic",
    ],
    "hardware": ["gold hardware", "silver hardware", "no visible hardware"],
    "pattern": [
        "solid", "animal print", "quilted", "woven", "logo print", "striped",
        "embellished",
    ],
    "season": SEASONS,
}

BAG_MULTI: dict[str, list[str]] = {
    "dominant_colors": COLORS,
    "style_tags": [
        "minimalist", "elegant", "classic", "trendy", "bohemian", "edgy",
        "business", "evening",
    ],
}

BAG_TEMPLATES = [
    "a photo of a {label} bag",
    "a close-up product photo of a {label} handbag",
    "{label}",
]


# ---------------------------------------------------------------------------
# JEWELRY schema
# ---------------------------------------------------------------------------
JEWELRY_SCHEMA: dict[str, list[str]] = {
    "type": [
        "ring", "bracelet", "bangle", "necklace", "pendant", "earrings",
        "hoop earrings", "stud earrings", "anklet", "brooch",
    ],
    "material": [
        "gold", "silver", "rose gold", "pearl", "stainless steel", "resin",
        "beaded", "gemstone", "enamel", "crystal",
    ],
    "finish": ["polished", "matte", "hammered", "engraved"],
    "pattern": ["solid", "chain link", "twisted", "embellished", "geometric"],
    "season": SEASONS,
}

JEWELRY_MULTI: dict[str, list[str]] = {
    "dominant_colors": COLORS,
    "style_tags": [
        "minimalist", "elegant", "statement", "delicate", "classic", "trendy",
        "bohemian", "vintage",
    ],
}

JEWELRY_TEMPLATES = [
    "a photo of a {label}",
    "a close-up product photo of {label} jewellery",
    "{label}",
]


# Dispatch tables so callers pick a schema by category rather than branching.
SCHEMA_BY_CATEGORY: dict[str, dict[str, list[str]]] = {
    "shoes": SHOE_SCHEMA,
    "bags": BAG_SCHEMA,
    "jewelry": JEWELRY_SCHEMA,
}
MULTI_BY_CATEGORY: dict[str, dict[str, list[str]]] = {
    "shoes": SHOE_MULTI,
    "bags": BAG_MULTI,
    "jewelry": JEWELRY_MULTI,
}
TEMPLATES_BY_CATEGORY: dict[str, list[str]] = {
    "shoes": SHOE_TEMPLATES,
    "bags": BAG_TEMPLATES,
    "jewelry": JEWELRY_TEMPLATES,
}
