"""Base classes for data transforms."""

from abc import ABC, abstractmethod
from typing import Any

import polars as pl


class BaseTransform(ABC):
    """Abstract base class for all transforms.

    Transforms follow the fit/transform pattern:
    - fit(): Learn parameters from training data (e.g., mean, std, label mappings)
    - transform(): Apply the transform to data
    - fit_transform(): Convenience method to do both

    Stateful transforms (e.g., ScaleTransform, EncodeTransform) must be fitted
    on training data before being applied to prediction data.
    """

    name: str = "base"
    is_fitted: bool = False

    @abstractmethod
    def fit(self, df: pl.DataFrame, target_column: str | None = None) -> "BaseTransform":
        """Fit transform on training data.

        Args:
            df: Training DataFrame
            target_column: Optional target column name (excluded from some transforms)

        Returns:
            self for method chaining
        """
        pass

    @abstractmethod
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply transform to data.

        Args:
            df: DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        pass

    def fit_transform(self, df: pl.DataFrame, target_column: str | None = None) -> pl.DataFrame:
        """Fit and transform in one step.

        Args:
            df: Training DataFrame
            target_column: Optional target column name

        Returns:
            Transformed DataFrame
        """
        return self.fit(df, target_column).transform(df)

    def get_params(self) -> dict[str, Any]:
        """Get fitted parameters for serialization."""
        return {}

    def set_params(self, params: dict[str, Any]) -> None:
        """Set parameters from deserialization."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fitted={self.is_fitted})"


class StatelessTransform(BaseTransform):
    """Transform that doesn't need fitting.

    Used for transforms like renaming, casting, and feature engineering
    that don't learn parameters from data.
    """

    is_fitted: bool = True  # Always considered "fitted"

    def fit(self, df: pl.DataFrame, target_column: str | None = None) -> "StatelessTransform":
        """No-op for stateless transforms."""
        return self
