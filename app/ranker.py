"""Mode A ranking — a single LLM call via OpenRouter (Claude Haiku 4.5 default).

Input: the outfit's extracted attributes + a shortlist of candidate shoes
(id, title, shoe attributes, price — no images). Output: the top 3 shoes, each
with a one-sentence shopper-facing rationale and a coherence score.

Robustness:
* Strict JSON out, parsed through the RankerResponse Pydantic contract.
* On parse/validation failure we retry ONCE with a repair prompt that echoes the
  malformed text and the schema.
* Every returned product_id is validated against the shortlist; hallucinated IDs
  are dropped rather than surfaced to the shopper.
"""
from __future__ import annotations

import json
from typing import Optional

from pydantic import ValidationError

from app.config import settings
from app.models import OutfitAttributes, ProductResult, RankerResponse

_SYSTEM = (
    "You are a fashion stylist for a boutique selling shoes, handbags and "
    "jewellery. From a fixed candidate list you assemble one complete look for "
    "the shopper's outfit. Enforce colour harmony and formality consistency: "
    "each piece's formality must sit close to the outfit's, and its colours "
    "must harmonise with the outfit's palette. You never invent product IDs — "
    "you only choose from the list. "
    "Write every rationale in FRENCH, addressing the shopper directly, in one "
    "natural sentence. Respond with STRICT JSON and no prose."
)

# Several pieces per category rather than the single best overall: the shopper
# wants options for each slot (a few pairs of shoes, a few bags, a few jewels),
# not one of each. The per-category cap keeps the balance -- without it the
# catalog's ~320 shoes to 11 bags would swamp the look with shoes alone.
_PER_CATEGORY = 3

_SCHEMA_HINT = (
    'Return exactly this shape:\n'
    '{"ranked": [{"product_id": "<id from list>", '
    '"rationale": "<une phrase en français, adressée au client>", '
    '"coherence": <number 0..1>}]}\n'
    f"Choose UP TO {_PER_CATEGORY} items per category, and cover every category "
    "that appears in the list (shoes, bags, jewelry) -- give the shopper a few "
    "options for each. Order best first within each category. The rationale "
    "must be in French."
)


def rank_outfit(
    outfit: OutfitAttributes,
    candidates: list[dict],
    client=None,
    top_n: int = 12,
    per_category: int = _PER_CATEGORY,
) -> list[ProductResult]:
    if not candidates:
        return []
    if client is None:
        from app.llm import get_llm

        client = get_llm()

    valid_ids = {c["product_id"] for c in candidates}
    by_id = {c["product_id"]: c for c in candidates}

    prompt = _build_prompt(outfit, candidates)
    text = _call(client, prompt)
    parsed = _parse(text)
    if parsed is None:
        # One repair attempt.
        text = _call(client, _repair_prompt(text))
        parsed = _parse(text)
    if parsed is None:
        return []  # give up gracefully rather than surface garbage

    results: list[ProductResult] = []
    per_cat_count: dict[str, int] = {}
    for item in parsed.ranked:
        if item.product_id not in valid_ids:
            continue  # drop hallucinated IDs
        c = by_id[item.product_id]
        # The prompt asks for at most `per_category` pieces per category; cap it
        # here too, because a model that ignores the instruction would otherwise
        # return a pile of shoes and call it a complete look.
        cat = c.get("category")
        if cat:
            if per_cat_count.get(cat, 0) >= per_category:
                continue
            per_cat_count[cat] = per_cat_count.get(cat, 0) + 1
        results.append(
            ProductResult(
                product_id=item.product_id,
                title=c["title"],
                handle=c["handle"],
                price=c.get("price"),
                image_url=c.get("image_url"),
                variant_id=c.get("variant_id"),
                category=cat,
                score=round(item.coherence, 4),
                rationale=item.rationale,
            )
        )
        if len(results) >= top_n:
            break
    return results


def _build_prompt(outfit: OutfitAttributes, candidates: list[dict]) -> str:
    outfit_json = outfit.model_dump(mode="json")
    lines = []
    for c in candidates:
        attrs = c.get("attributes", {})
        lines.append(
            json.dumps(
                {
                    "product_id": c["product_id"],
                    "title": c["title"],
                    "price": c.get("price"),
                    # The model needs the category to honour one-per-category.
                    "category": c.get("category"),
                    "attributes": _slim_attrs(attrs),
                }
            )
        )
    return (
        f"Outfit attributes:\n{json.dumps(outfit_json)}\n\n"
        f"Candidate pieces, with their category "
        f"(choose from these product_ids only):\n"
        + "\n".join(lines)
        + f"\n\n{_SCHEMA_HINT}"
    )


def _slim_attrs(attrs: dict) -> dict:
    """Keep only the styling-relevant fields to save tokens."""
    keep = ["type", "material", "formality", "season", "dominant_colors", "style_tags"]
    out = {}
    for k in keep:
        v = attrs.get(k)
        if isinstance(v, dict):  # LabeledAxis dump
            v = v.get("value")
        if v:
            out[k] = v
    return out


def _call(client, prompt: str) -> str:
    return client.chat(
        model=settings.ranker_model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        # Room for up to ~9 French rationales plus JSON scaffolding. Too low and
        # the reply truncates mid-JSON -> parse fails -> repair truncates again
        # -> the shopper silently gets zero results.
        max_tokens=2000,
    )


def _repair_prompt(bad_text: str) -> str:
    return (
        "Your previous reply was not valid JSON for the required schema.\n"
        f"Previous reply:\n{bad_text}\n\n"
        f"Reply again with STRICT JSON only.\n{_SCHEMA_HINT}"
    )


def _parse(text: str) -> Optional[RankerResponse]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    try:
        return RankerResponse.model_validate(data)
    except ValidationError:
        return None
