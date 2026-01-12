"""Scaling/normalization transform using StandardScaler."""

from typing import Any

import polars as pl

from dyxgb.transforms.base import BaseTransform


class ScaleTransform(BaseTransform):
    """Scale numeric columns using StandardScaler or MinMaxScaler.

    Methods:
        - standard: (x - mean) / std  (StandardScaler)
        - minmax: (x - min) / (max - min)  (MinMaxScaler)

    Example config:
        scale:
            method: standard
            columns:  # Optional - if omitted, scales all numeric columns
                - amount
                - age
                - score

    This is a STATEFUL transform - learns mean/std (or min/max) from training
    data and applies the same transformation to prediction data.
    """

    name = "scale"

    def __init__(
        self,
        method: str = "standard",
        columns: list[str] | None = None,
    ) -> None:
        """Initialize scale transform.

        Args:
            method: Scaling method - 'standard' or 'minmax'
            columns: List of columns to scale. If None, scales all numeric columns.
        """
        if method not in ("standard", "minmax"):
            raise ValueError(f"Invalid method '{method}'. Use 'standard' or 'minmax'")

        self.method = method
        self.columns = columns

        # Learned parameters: column -> {mean, std} or {min, max}
        self.params_: dict[str, dict[str, float]] = {}
        self.is_fitted = False

    def fit(self, df: pl.DataFrame, target_column: str | None = None) -> "ScaleTransform":
        """Learn scaling parameters from training data.

        Args:
            df: Training DataFrame
            target_column: Target column to exclude from scaling

        Returns:
            self
        """
        self.params_ = {}

        # Determine columns to scale
        if self.columns:
            cols_to_scale = [c for c in self.columns if c in df.columns]
        else:
            # Auto-detect numeric columns
            cols_to_scale = [
                c for c in df.columns if df[c].dtype.is_numeric() and c != target_column
            ]

        for col in cols_to_scale:
            if not df[col].dtype.is_numeric():
                continue

            if self.method == "standard":
                mean_val = df[col].mean()
                std_val = df[col].std()

                # Handle zero std (constant column)
                if std_val == 0 or std_val is None:
                    std_val = 1.0

                self.params_[col] = {
                    "mean": float(mean_val) if mean_val is not None else 0.0,
                    "std": float(std_val),
                }
            else:  # minmax
                min_val = df[col].min()
                max_val = df[col].max()

                # Handle case where min == max (constant column)
                range_val = max_val - min_val if max_val != min_val else 1.0

                self.params_[col] = {
                    "min": float(min_val) if min_val is not None else 0.0,
                    "max": float(max_val) if max_val is not None else 1.0,
                    "range": float(range_val),
                }

        self.is_fitted = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply scaling to numeric columns.

        Args:
            df: DataFrame to transform

        Returns:
            DataFrame with scaled columns
        """
        if not self.is_fitted:
            raise RuntimeError("ScaleTransform must be fitted before transform")

        scale_exprs = []

        for col, params in self.params_.items():
            if col not in df.columns:
                continue

            if self.method == "standard":
                # (x - mean) / std
                mean = params["mean"]
                std = params["std"]
                scale_exprs.append(((pl.col(col) - mean) / std).alias(col))
            else:  # minmax
                # (x - min) / (max - min)
                min_val = params["min"]
                range_val = params["range"]
                scale_exprs.append(((pl.col(col) - min_val) / range_val).alias(col))

        if scale_exprs:
            return df.with_columns(scale_exprs)
        return df

    def inverse_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Reverse the scaling transformation.

        Args:
            df: DataFrame with scaled columns

        Returns:
            DataFrame with original scale
        """
        if not self.is_fitted:
            raise RuntimeError("ScaleTransform must be fitted before inverse_transform")

        inverse_exprs = []

        for col, params in self.params_.items():
            if col not in df.columns:
                continue

            if self.method == "standard":
                # x * std + mean
                mean = params["mean"]
                std = params["std"]
                inverse_exprs.append((pl.col(col) * std + mean).alias(col))
            else:  # minmax
                # x * (max - min) + min
                min_val = params["min"]
                range_val = params["range"]
                inverse_exprs.append((pl.col(col) * range_val + min_val).alias(col))

        if inverse_exprs:
            return df.with_columns(inverse_exprs)
        return df

    def get_params(self) -> dict[str, Any]:
        """Get parameters for serialization."""
        return {
            "method": self.method,
            "columns": self.columns,
            "params_": self.params_,
        }

    def set_params(self, params: dict[str, Any]) -> None:
        """Set parameters from deserialization."""
        self.method = params.get("method", "standard")
        self.columns = params.get("columns")
        self.params_ = params.get("params_", {})
        self.is_fitted = bool(self.params_)

    def __repr__(self) -> str:
        n_fitted = len(self.params_)
        return f"ScaleTransform(method='{self.method}', {n_fitted} columns fitted)"
