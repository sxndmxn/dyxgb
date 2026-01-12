"""File-based data loaders for CSV, Parquet, and JSON."""

from pathlib import Path

import polars as pl

from dyxgb.data.base import BaseLoader


class FileLoader(BaseLoader):
    """Load data from files (CSV, Parquet, JSON, NDJSON)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Data file not found: {self.path}")

    def load(self) -> pl.DataFrame:
        """Load data based on file extension."""
        suffix = self.path.suffix.lower()

        loaders = {
            ".csv": self._load_csv,
            ".parquet": self._load_parquet,
            ".pq": self._load_parquet,
            ".json": self._load_json,
            ".ndjson": self._load_ndjson,
            ".jsonl": self._load_ndjson,
        }

        loader = loaders.get(suffix)
        if not loader:
            supported = ", ".join(loaders.keys())
            raise ValueError(f"Unsupported file format: {suffix}. Supported: {supported}")

        return loader()

    def _load_csv(self) -> pl.DataFrame:
        return pl.read_csv(self.path)

    def _load_parquet(self) -> pl.DataFrame:
        return pl.read_parquet(self.path)

    def _load_json(self) -> pl.DataFrame:
        return pl.read_json(self.path)

    def _load_ndjson(self) -> pl.DataFrame:
        return pl.read_ndjson(self.path)


def load_file(path: str | Path) -> pl.DataFrame:
    """Convenience function to load a file."""
    return FileLoader(path).load()
