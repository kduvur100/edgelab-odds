"""
edgelab_odds.db
===============
Thin wrapper around DuckDB for the edgelab-odds pipeline.

All heavy data work (ingestion, feature building, backtesting) talks to
the database through this module so the connection + schema lifecycle
stays in one place.

Usage:
    from edgelab_odds.db import get_conn, init_db

    conn = get_conn()           # returns an open DuckDB connection
    init_db(conn)               # creates tables if they do not exist
"""

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import duckdb
import pandas as pd

from edgelab_odds.config import settings

log = logging.getLogger(__name__)


# ── Connection ────────────────────────────────────────────────────────────────

def get_conn(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Return an open DuckDB connection to the project database.

    The database file is created automatically if it does not exist.
    Callers are responsible for closing the connection when done.
    """
    settings.ensure_dirs()
    path = str(settings.db_path)
    log.debug("Opening DuckDB connection: %s (read_only=%s)", path, read_only)
    return duckdb.connect(path, read_only=read_only)


@contextmanager
def conn_ctx(read_only: bool = False) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Context manager that opens and auto-closes a DuckDB connection.

    Example::

        with conn_ctx() as conn:
            conn.execute("SELECT count(*) FROM fights")
    """
    conn = get_conn(read_only=read_only)
    try:
        yield conn
    finally:
        conn.close()


# ── Schema bootstrap ──────────────────────────────────────────────────────────

def init_db(conn: duckdb.DuckDBPyConnection | None = None) -> None:
    """Create all tables (if they do not already exist).

    Reads ``sql/schema.sql`` from the project root and executes it.
    Safe to call multiple times — uses CREATE TABLE IF NOT EXISTS.
    """
    schema_path = settings.project_root / "sql" / "schema.sql"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    sql = schema_path.read_text()

    _conn = conn or get_conn()
    try:
        # DuckDB supports executing multi-statement SQL strings directly
        _conn.execute(sql)
        log.info("Database schema initialised from %s", schema_path)
    finally:
        if conn is None:
            _conn.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def table_exists(table: str, conn: duckdb.DuckDBPyConnection) -> bool:
    """Return True if *table* exists in the connected database."""
    result = conn.execute(
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_name = ?",
        [table],
    ).fetchone()
    return bool(result and result[0] > 0)


def row_count(table: str, conn: duckdb.DuckDBPyConnection) -> int:
    """Return the number of rows in *table*."""
    return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]  # type: ignore[index]


def query_df(sql: str, conn: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """Execute *sql* and return the result as a pandas DataFrame.

    Opens (and closes) its own connection if *conn* is not provided.
    """
    _conn = conn or get_conn(read_only=True)
    try:
        return _conn.execute(sql).df()
    finally:
        if conn is None:
            _conn.close()


def upsert_df(
    df: pd.DataFrame,
    table: str,
    conn: duckdb.DuckDBPyConnection,
    if_exists: str = "append",
) -> int:
    """Write *df* into *table* using DuckDB's fast bulk-insert.

    Parameters
    ----------
    df:
        DataFrame to write.
    table:
        Target table name.
    conn:
        Open DuckDB connection.
    if_exists:
        ``"append"`` (default) adds rows; ``"replace"`` drops and recreates
        the table first.

    Returns
    -------
    int
        Number of rows written.
    """
    if if_exists == "replace":
        conn.execute(f"DROP TABLE IF EXISTS {table}")

    # Register the DataFrame as a temporary view so DuckDB can INSERT from it
    conn.register("_staging", df)
    conn.execute(f"INSERT INTO {table} SELECT * FROM _staging")
    conn.unregister("_staging")

    written = len(df)
    log.debug("Wrote %d rows to %s", written, table)
    return written
