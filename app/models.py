"""Pydantic contracts.

These types are the boundary between (a) internal pipeline stages, (b) the HTTP
API surface, and (c) every LLM JSON response. Anything the ranker LLM returns is
parsed through `RankerResponse` and repaired on failure — see app/ranker.py.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Attribute extraction outputs
# ---------------------------------------------------------------------------
class LabeledAxis(BaseModel):
    """One axis result: the winning label and its confidence [0,1]."""

    value: str
    confidence: float = Field(ge=0.0, le=1.0)


class ShoeAttributes(BaseModel):
    type: Optional[LabeledAxis] = None
    material: Optional[LabeledAxis] = None
    sole_type: Optional[LabeledAxis] = None
    heel_height: Optional[LabeledAxis] = None
    toe_shape: Optional[LabeledAxis] = None
    pattern: Optional[LabeledAxis] = None
    season: Optional[LabeledAxis] = None
    formality: int = Field(ge=1, le=5, default=3)
    dominant_colors: list[str] = Field(default_factory=list)
    style_tags: list[str] = Field(default_factory=list)

    def type_value(self) -> Optional[str]:
        return self.type.value if self.type else None


class ItemAttributes(BaseModel):
    """Attributes of any single catalog item, whatever its category.

    The axes every category shares are named fields; the ones specific to one
    category (a shoe's heel_height, a bag's hardware, a jewel's finish) land in
    `extras`, so adding a category means editing app/labels.py and nothing here.
    """

    category: Optional[str] = None
    type: Optional[LabeledAxis] = None
    material: Optional[LabeledAxis] = None
    pattern: Optional[LabeledAxis] = None
    season: Optional[LabeledAxis] = None
    formality: int = Field(ge=1, le=5, default=3)
    dominant_colors: list[str] = Field(default_factory=list)
    style_tags: list[str] = Field(default_factory=list)
    extras: dict[str, LabeledAxis] = Field(default_factory=dict)

    def type_value(self) -> Optional[str]:
        return self.type.value if self.type else None


class OutfitAttributes(BaseModel):
    silhouette: Optional[LabeledAxis] = None
    pattern: Optional[LabeledAxis] = None
    season: Optional[LabeledAxis] = None
    occasion: Optional[LabeledAxis] = None
    formality: int = Field(ge=1, le=5, default=3)
    dominant_colors: list[str] = Field(default_factory=list)
    garments_present: list[str] = Field(default_factory=list)
    style_tags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Catalog / retrieval
# ---------------------------------------------------------------------------
class ProductResult(BaseModel):
    """A single shoe returned to the storefront. Common to both modes."""

    product_id: str
    title: str
    handle: str
    price: Optional[str] = None
    image_url: Optional[str] = None
    variant_id: Optional[str] = None
    # Which catalog category this piece belongs to (shoes | bags | jewelry).
    # The storefront groups outfit results into one row per category, so it
    # needs the bucket on every card rather than inferring it from the title.
    category: Optional[str] = None
    # The size `variant_id` resolves to, when the shopper picked one. None for
    # sizeless pieces (bags, jewellery), so the card can label a shoe "Taille 38"
    # without claiming a handbag has a size.
    size: Optional[str] = None
    score: float = 0.0
    # Mode A only: one-sentence shopper-facing rationale.
    rationale: Optional[str] = None


# ---------------------------------------------------------------------------
# Mode routing
# ---------------------------------------------------------------------------
Mode = Literal["shoe", "outfit", "both"]


class RouteResult(BaseModel):
    mode: Mode
    confidence: float = Field(ge=0.0, le=1.0)
    shoe_score: float
    outfit_score: float
    # Which category the single-item reading landed on (shoes | bags |
    # jewelry). None for a pure outfit. Retrieval scopes the index to it, so
    # a bag is never ranked against a bracelet.
    category: Optional[str] = None


# ---------------------------------------------------------------------------
# LLM contract for Mode A ranking
# ---------------------------------------------------------------------------
class RankedShoe(BaseModel):
    product_id: str
    rationale: str
    coherence: float = Field(ge=0.0, le=1.0)

    @field_validator("rationale")
    @classmethod
    def _trim(cls, v: str) -> str:
        return v.strip()


class RankerResponse(BaseModel):
    """Strict JSON the ranker must return. Extra keys are ignored."""

    ranked: list[RankedShoe] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level API response
# ---------------------------------------------------------------------------
class RecommendResponse(BaseModel):
    mode: Mode
    confidence: float
    # query_attributes is the extracted schema for whatever the photo was.
    # For "both", we surface the outfit attributes (the richer of the two).
    query_attributes: dict = Field(default_factory=dict)
    results: list[ProductResult] = Field(default_factory=list)
    # Echo of the size the results were filtered to, so the storefront can label
    # the grid without trusting its own local state.
    size: Optional[str] = None
    # Populated only when mode == "both": per-tab result lists.
    shoe_results: Optional[list[ProductResult]] = None
    outfit_results: Optional[list[ProductResult]] = None
