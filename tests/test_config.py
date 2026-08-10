"""Settings normalisation.

Covers the deploy-time footgun: managed Postgres add-ons hand out a URL whose
driver we don't ship, and the failure lands at engine creation on the host
rather than anywhere a local test run would notice.
"""
from __future__ import annotations

import pytest

from app.config import Settings


@pytest.mark.parametrize(
    "given",
    [
        "postgres://u:p@host:5432/db",  # Heroku-style, still emitted by some add-ons
        "postgresql://u:p@host:5432/db",  # Railway / Fly
        "postgresql+psycopg://u:p@host:5432/db",  # already correct
    ],
)
def test_database_url_always_lands_on_psycopg3(given):
    assert Settings(database_url=given).database_url == (
        "postgresql+psycopg://u:p@host:5432/db"
    )


def test_sqlite_url_is_left_alone():
    """The test harness swaps in SQLite; the validator must not touch it."""
    url = "sqlite:///tmp/khriss.db"
    assert Settings(database_url=url).database_url == url


def test_empty_database_url_says_what_actually_went_wrong():
    """Railway blanks an unresolvable reference; the default can't save us."""
    with pytest.raises(ValueError, match="did not resolve"):
        Settings(database_url="")


def test_rewritten_url_resolves_a_driver_we_ship():
    """The string being right is not the point -- the dialect must import.

    create_engine() resolves and imports the DBAPI without opening a socket, so
    this reproduces the boot failure (psycopg2 missing) with no live Postgres.
    """
    from sqlalchemy import create_engine

    settings = Settings(database_url="postgresql://u:p@host:5432/db")
    engine = create_engine(settings.database_url)
    assert engine.dialect.driver == "psycopg"
