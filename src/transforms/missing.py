"""Missing value imputation transform."""

from typing import Any

import polars as pl

from dyxgb.transforms.base import BaseTransform


class MissingTransform(BaseTransform):
    """Handle missing values with various imputation strategies.

    Strategies:
        - mean: Fill with column mean (numeric only)
        - median: Fill with column median (numeric only)
        - mode: Fill with most frequent value (works for all types)
        - constant: Fill with a specified constant value
        - drop: Drop rows with missing values

    Example config:
        missing:
            # Global strategy for all columns
            strategy: median

            # Or per-column strategies
            columns:
                amount: median
                age: mean
                category: mode
                status: constant
            constant_value: "unknown"  # Used when strategy is 'constant'

    This transform is STATEFUL - it learns fill values from training data
    and applies them consistently to prediction data.
    """

    name = "missing"

    def __init__(
        self,
        strategy: str = "median",
        columns: dict[str, str] | None = None,
        constant_value: Any = None,
    ) -> None:
        """Initialize missing value transform.

        Args:
            strategy: Default strategy for all columns
            columns: Optional per-column strategy overrides
            constant_value: Value to use for 'constant' strategy
        """
        self.strategy = strategy
        self.columns = columns or {}
        self.constant_value = constant_value

        # Learned fill values (fitted on training data)
        self.fill_values: dict[str, Any] = {}
        self.is_fitted = False

        # Validate strategies
        valid_strategies = {"mean", "median", "mode", "constant", "drop"}
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy '{strategy}'. Valid: {valid_strategies}")
        for col, strat in self.columns.items():
            if strat not in valid_strategies:
                raise ValueError(f"Invalid strategy '{strat}' for column '{col}'")

    def fit(self, df: pl.DataFrame, target_column: str | None = None) -> "MissingTransform":
        """Learn fill values from training data.

        Args:
            df: Training DataFrame
            target_column: Target column to exclude from imputation

        Returns:
            self
        """
        self.fill_values = {}

        # Determine which columns to process
        cols_to_process = set(df.columns)
        if target_column and target_column in cols_to_process:
            cols_to_process.remove(target_column)

        for col in cols_to_process:
            # Get strategy for this column
            col_strategy = self.columns.get(col, self.strategy)

            if col_strategy == "drop":
                # No fill value needed for drop strategy
                continue
            elif col_strategy == "constant":
                self.fill_values[col] = self.constant_value
            elif col_strategy == "mean":
                if df[col].dtype.is_numeric():
                    self.fill_values[col] = df[col].mean()
            elif col_strategy == "median":
                if df[col].dtype.is_numeric():
                    self.fill_values[col] = df[col].median()
            elif col_strategy == "mode":
                # Mode works for any type
                mode_result = df[col].drop_nulls().mode()
                if len(mode_result) > 0:
                    self.fill_values[col] = mode_result[0]

        self.is_fitted = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply imputation to data.

        Args:
            df: DataFrame to transform

        Returns:
            DataFrame with missing values handled
        """
        if not self.is_fitted:
            raise RuntimeError("MissingTransform must be fitted before transform")

        # Check if we need to drop rows
        cols_to_drop = [col for col in df.columns if self.columns.get(col, self.strategy) == "drop"]

        if cols_to_drop:
            df = df.drop_nulls(subset=cols_to_drop)

        # Apply fill values
        fill_exprs = []
        for col, fill_value in self.fill_values.items():
            if col in df.columns and fill_value is not None:
                fill_exprs.append(pl.col(col).fill_null(fill_value).alias(col))

        if fill_exprs:
            return df.with_columns(fill_exprs)
        return df

    def get_params(self) -> dict[str, Any]:
        """Get parameters for serialization."""
        return {
            "strategy": self.strategy,
            "columns": self.columns,
            "constant_value": self.constant_value,
            "fill_values": self.fill_values,
        }

    def set_params(self, params: dict[str, Any]) -> None:
        """Set parameters from deserialization."""
        self.strategy = params.get("strategy", "median")
        self.columns = params.get("columns", {})
        self.constant_value = params.get("constant_value")
        self.fill_values = params.get("fill_values", {})
        self.is_fitted = bool(self.fill_values)

    def __repr__(self) -> str:
        return f"MissingTransform(strategy='{self.strategy}', fitted={self.is_fitted})"
