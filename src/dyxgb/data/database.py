"""Database loaders for SQLite, DuckDB, and PostgreSQL."""

from abc import abstractmethod
from pathlib import Path
from urllib.parse import urlparse

import polars as pl

from dyxgb.data.base import BaseLoader, DataSource


class DatabaseLoader(BaseLoader):
    """Base class for database loaders."""

    def __init__(self, uri: str, query: str | None = None, table: str | None = None):
        self.uri = uri
        self._query = query
        self._table = table

        if not query and not table:
            raise ValueError("Either 'query' or 'table' must be provided")

    @property
    def query(self) -> str:
        """Get the SQL query to execute."""
        if self._query:
            return self._query
        return f"SELECT * FROM {self._table}"

    @abstractmethod
    def load(self) -> pl.DataFrame:
        """Load data from database."""
        pass


class SQLiteLoader(DatabaseLoader):
    """Load data from SQLite database."""

    def load(self) -> pl.DataFrame:
        """Load data using Polars' native SQLite support."""
        # Extract path from URI (sqlite:///path/to/db.sqlite)
        parsed = urlparse(self.uri)
        db_path = parsed.path
        if db_path.startswith("/") and len(db_path) > 1:
            # Handle sqlite:///absolute/path vs sqlite:///./relative/path
            if parsed.netloc:
                db_path = parsed.netloc + db_path
            # For Windows: sqlite:///C:/path -> C:/path
            elif db_path.startswith("/") and len(db_path) > 2 and db_path[2] == ":":
                db_path = db_path[1:]

        if not Path(db_path).exists():
            raise FileNotFoundError(f"SQLite database not found: {db_path}")

        import sqlite3

        conn = sqlite3.connect(db_path)
        try:
            # Use Polars to read from SQLite connection
            df = pl.read_database(self.query, conn)
        finally:
            conn.close()

        return df


class DuckDBLoader(DatabaseLoader):
    """Load data from DuckDB database."""

    def load(self) -> pl.DataFrame:
        """Load data using DuckDB."""
        try:
            import duckdb
        except ImportError:
            raise ImportError(
                "DuckDB is required for DuckDB sources. Install with: uv add duckdb"
            )

        # Extract path from URI (duckdb:///path/to/db.duckdb or :memory:)
        parsed = urlparse(self.uri)
        db_path = parsed.path.lstrip("/") if parsed.path else ":memory:"

        if db_path != ":memory:" and not Path(db_path).exists():
            raise FileNotFoundError(f"DuckDB database not found: {db_path}")

        conn = duckdb.connect(db_path, read_only=True)
        try:
            result = conn.execute(self.query)
            df = result.pl()
        finally:
            conn.close()

        return df


class PostgresLoader(DatabaseLoader):
    """Load data from PostgreSQL database."""

    def load(self) -> pl.DataFrame:
        """Load data using connectorx for best performance with Polars."""
        try:
            import connectorx as cx
        except ImportError:
            raise ImportError(
                "connectorx is required for PostgreSQL sources. "
                "Install with: uv add connectorx"
            )

        # connectorx expects postgresql:// not postgres://
        uri = self.uri
        if uri.startswith("postgres://"):
            uri = "postgresql://" + uri[len("postgres://") :]

        df = cx.read_sql(uri, self.query, return_type="polars")
        return df


def get_database_loader(
    source_type: DataSource | str,
    uri: str,
    query: str | None = None,
    table: str | None = None,
) -> DatabaseLoader:
    """Factory function to get the appropriate database loader."""
    if isinstance(source_type, str):
        source_type = DataSource(source_type)

    loaders = {
        DataSource.SQLITE: SQLiteLoader,
        DataSource.DUCKDB: DuckDBLoader,
        DataSource.POSTGRES: PostgresLoader,
    }

    loader_cls = loaders.get(source_type)
    if not loader_cls:
        raise ValueError(f"Unsupported database type: {source_type}")

    return loader_cls(uri, query=query, table=table)


def load_from_uri(
    uri: str, query: str | None = None, table: str | None = None
) -> pl.DataFrame:
    """Auto-detect database type from URI and load data."""
    parsed = urlparse(uri)
    scheme = parsed.scheme.lower()

    scheme_to_source = {
        "sqlite": DataSource.SQLITE,
        "duckdb": DataSource.DUCKDB,
        "postgres": DataSource.POSTGRES,
        "postgresql": DataSource.POSTGRES,
    }

    source_type = scheme_to_source.get(scheme)
    if not source_type:
        raise ValueError(
            f"Unsupported URI scheme: {scheme}. "
            f"Supported: {', '.join(scheme_to_source.keys())}"
        )

    loader = get_database_loader(source_type, uri, query=query, table=table)
    return loader.load()
