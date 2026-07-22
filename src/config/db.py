"""
src/config/db.py
─────────────────
Thread-safe PostgreSQL connection pool using psycopg3 (psycopg).

Every request handler calls `get_conn()` as a context manager.
The pool is initialised once at FastAPI startup and torn down at shutdown.
"""
import os
import logging
from contextlib import contextmanager
from typing import Generator

import psycopg
from psycopg_pool import ConnectionPool
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── singleton pool (created in lifespan) ──────────────────────────────────────
_pool: ConnectionPool | None = None


def init_pool() -> None:
    """Call once at application startup."""
    global _pool
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL environment variable is not set.")

    _pool = ConnectionPool(
        conninfo=dsn,
        min_size=2,
        max_size=10,
        open=True,
    )
    logger.info("Database connection pool initialised (min=2, max=10).")


def close_pool() -> None:
    """Call once at application shutdown."""
    global _pool
    if _pool:
        _pool.close()
        logger.info("Database connection pool closed.")


@contextmanager
def get_conn() -> Generator[psycopg.Connection, None, None]:
    """
    Yield a connection from the pool.

    Usage:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(...)
    """
    if _pool is None:
        raise RuntimeError("Pool not initialised — call init_pool() first.")

    with _pool.connection() as conn:
        yield conn


# ── simple one-shot helper (kept for notebook / script usage) ─────────────────
def get_connection() -> psycopg.Connection:
    """Return a standalone connection (no pool). Notebooks use this."""
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL environment variable is not set.")
    return psycopg.connect(dsn)