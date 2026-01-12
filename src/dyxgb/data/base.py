"""Base classes and protocols for data loading."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import polars as pl


class DataSource(str, Enum):
    """Supported data source types."""

    FILE = "file"
    SQLITE = "sqlite"
    DUCKDB = "duckdb"
    POSTGRES = "postgres"


@runtime_checkable
class DataLoader(Protocol):
    """Protocol for data loaders."""

    def load(self) -> pl.DataFrame:
        """Load data and return as Polars DataFrame."""
        ...


@dataclass
class DataConfig:
    """Configuration for a data source."""

    source_type: DataSource
    # For file sources
    path: str | None = None
    # For database sources
    uri: str | None = None
    query: str | None = None
    table: str | None = None

    def __post_init__(self) -> None:
        if self.source_type == DataSource.FILE:
            if not self.path:
                raise ValueError("File source requires 'path'")
        else:
            if not self.uri:
                raise ValueError(f"{self.source_type.value} source requires 'uri'")
            if not self.query and not self.table:
                raise ValueError(
                    f"{self.source_type.value} source requires 'query' or 'table'"
                )


class BaseLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load(self) -> pl.DataFrame:
        """Load data and return as Polars DataFrame."""
        pass

    def validate_columns(self, df: pl.DataFrame, required: list[str]) -> None:
        """Validate that required columns exist in the DataFrame."""
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
