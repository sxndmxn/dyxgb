"""Categorical encoding transform using LabelEncoder."""

from typing import Any

import polars as pl

from dyxgb.transforms.base import BaseTransform


class EncodeTransform(BaseTransform):
    """Encode categorical columns as integers using LabelEncoder.

    Maps each unique category to an integer. The mapping is learned from
    training data and applied consistently to prediction data.

    Example config:
        encode:
            columns:
                - category
                - region
                - status
            unknown_value: -1  # Value for categories not seen in training

    Behavior:
        - During fit: Learns unique values from training data
        - During transform: Maps values to integers
        - Unknown values (not in training): Mapped to unknown_value (-1 by default)

    This is a STATEFUL transform - must be fitted on training data first.
    """

    name = "encode"

    def __init__(
        self,
        columns: list[str],
        unknown_value: int = -1,
    ) -> None:
        """Initialize encode transform.

        Args:
            columns: List of column names to encode
            unknown_value: Integer value for unseen categories (default: -1)
        """
        self.columns = columns
        self.unknown_value = unknown_value

        # Learned encodings: column -> {value: code}
        self.encoders: dict[str, dict[Any, int]] = {}
        # Reverse mappings for decoding: column -> {code: value}
        self.decoders: dict[str, dict[int, Any]] = {}
        self.is_fitted = False

    def fit(self, df: pl.DataFrame, target_column: str | None = None) -> "EncodeTransform":
        """Learn category encodings from training data.

        Args:
            df: Training DataFrame
            target_column: Ignored for this transform

        Returns:
            self
        """
        self.encoders = {}
        self.decoders = {}

        for col in self.columns:
            if col not in df.columns:
                continue

            # Get unique non-null values
            unique_vals = df[col].drop_nulls().unique().sort().to_list()

            # Create encoding mapping (0, 1, 2, ...)
            encoder = {val: i for i, val in enumerate(unique_vals)}
            decoder = {i: val for val, i in encoder.items()}

            self.encoders[col] = encoder
            self.decoders[col] = decoder

        self.is_fitted = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply label encoding to categorical columns.

        Args:
            df: DataFrame to transform

        Returns:
            DataFrame with encoded columns (as Int64)
        """
        if not self.is_fitted:
            raise RuntimeError("EncodeTransform must be fitted before transform")

        for col in self.columns:
            if col not in df.columns:
                continue
            if col not in self.encoders:
                continue

            encoder = self.encoders[col]

            # Use Polars replace with default for unknown values
            # First, create mapping as a struct for replace
            df = df.with_columns(
                pl.col(col).replace(encoder, default=self.unknown_value).cast(pl.Int64).alias(col)
            )

        return df

    def inverse_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Decode integer columns back to original categories.

        Args:
            df: DataFrame with encoded columns

        Returns:
            DataFrame with decoded columns
        """
        if not self.is_fitted:
            raise RuntimeError("EncodeTransform must be fitted before inverse_transform")

        for col in self.columns:
            if col not in df.columns:
                continue
            if col not in self.decoders:
                continue

            decoder = self.decoders[col]

            df = df.with_columns(pl.col(col).replace(decoder, default=None).alias(col))

        return df

    def get_classes(self, column: str) -> list[Any]:
        """Get the list of classes for a column (in order of their codes).

        Args:
            column: Column name

        Returns:
            List of class values in code order
        """
        if column not in self.decoders:
            raise ValueError(f"Column '{column}' not fitted")

        decoder = self.decoders[column]
        return [decoder[i] for i in range(len(decoder))]

    def get_params(self) -> dict[str, Any]:
        """Get parameters for serialization."""
        return {
            "columns": self.columns,
            "unknown_value": self.unknown_value,
            "encoders": self.encoders,
            "decoders": self.decoders,
        }

    def set_params(self, params: dict[str, Any]) -> None:
        """Set parameters from deserialization."""
        self.columns = params.get("columns", [])
        self.unknown_value = params.get("unknown_value", -1)
        self.encoders = params.get("encoders", {})
        self.decoders = params.get("decoders", {})
        self.is_fitted = bool(self.encoders)

    def __repr__(self) -> str:
        n_cols = len(self.columns)
        n_fitted = len(self.encoders)
        return f"EncodeTransform({n_cols} columns, {n_fitted} fitted)"
