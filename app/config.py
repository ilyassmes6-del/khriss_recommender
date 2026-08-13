"""Central configuration, loaded once from the environment.

Everything tunable at deploy time lives here so the rest of the code never
reads os.environ directly. Import `settings` and read attributes.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Shopify ---
    shopify_shop: str = "example.myshopify.com"
    # Dev Dashboard apps don't expose a permanent token: the client id/secret
    # are exchanged for a 24h one (see shopify_client.AccessTokenProvider).
    shopify_client_id: str = ""
    shopify_client_secret: str = ""
    # Optional override. Set this only for a legacy custom app's shpat_ token,
    # which never expires and skips the exchange entirely.
    shopify_admin_token: str = ""
    shopify_api_version: str = "2024-04"
    shopify_webhook_secret: str = ""

    # --- OpenRouter (LLM provider: Mode A ranking + optional vision extractor) ---
    # OpenRouter serves an OpenAI-compatible /chat/completions endpoint. Models
    # are namespaced slugs; the default routes to Claude Haiku 4.5.
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_referer: str = ""  # optional OpenRouter attribution header
    openrouter_title: str = "khriss"  # optional OpenRouter attribution header
    ranker_model: str = "anthropic/claude-haiku-4.5"
    # Model for the optional vision extractor; falls back to ranker_model if blank.
    vision_model: str = ""

    # --- Extractor selection ---
    # "clip" (free, zero-shot) or "llm" (OpenRouter vision, higher fidelity).
    extractor: str = "clip"

    # --- Datastores ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    database_url: str = "postgresql+psycopg://khriss:khriss@localhost:5432/khriss"

    # --- Serving / behaviour ---
    allowed_origins: str = "*"
    mode_confidence_threshold: float = 0.10
    # Softmax temperature for mapping the two routing cosines to probabilities.
    # Higher = more decisive. Tuned so a ~0.02 cosine gap sits near the default
    # ambiguity threshold; raise it to make routing commit more readily.
    route_temperature: float = 10.0

    # --- Retrieval knobs (documented in README > tuning) ---
    mode_b_top_k: int = 12
    mode_a_candidate_pool: int = 36  # ~12 per category before the ranker's cut
    mode_a_diversity_threshold: float = 0.92  # drop near-duplicate colorways
    mode_a_formality_tolerance: int = 1  # ±N formality points
    same_type_boost: float = 0.05  # Mode B soft boost for matching shoe type

    # --- CLIP model ---
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"

    # --- Qdrant collection ---
    qdrant_collection: str = "shoes"

    # --- Upload guard ---
    max_upload_bytes: int = 5 * 1024 * 1024  # 5 MB

    # --- Sizes ---
    # Which Shopify option names carry a shoe size. This catalog uses "Taille"
    # (values 36-41); the others are here so a store that labels the axis
    # differently works without a code change. Matched case-insensitively, and
    # anything else (notably Shopify's "Title"/"Default Title" for a product
    # with no options) is treated as sizeless.
    size_option_names: str = "taille,size,pointure"

    @property
    def size_option_names_set(self) -> set[str]:
        return {n.strip().lower() for n in self.size_option_names.split(",") if n.strip()}

    @field_validator("database_url")
    @classmethod
    def _pin_psycopg3_driver(cls, v: str) -> str:
        """Force the psycopg 3 dialect, whatever the host hands us.

        Managed Postgres add-ons (Railway, Heroku, Fly) inject a bare
        postgres:// or postgresql:// URL. SQLAlchemy reads those as psycopg2,
        which we don't install -- so an untouched URL fails at engine creation.
        """
        # An unresolvable ${{Service.DATABASE_URL}} reference is substituted
        # with "" rather than left unset, which overrides the default above and
        # surfaces as an opaque SQLAlchemy parse error at startup. Name it.
        if not v.strip():
            raise ValueError(
                "DATABASE_URL is empty. On Railway this means the "
                "${{Postgres.DATABASE_URL}} reference did not resolve -- check "
                "that the Postgres service is named exactly as referenced, or "
                "re-add it with the 'Add Reference' picker."
            )

        for prefix in ("postgres://", "postgresql://"):
            if v.startswith(prefix):
                return "postgresql+psycopg://" + v[len(prefix):]
        return v

    @property
    def allowed_origins_list(self) -> list[str]:
        raw = self.allowed_origins.strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
