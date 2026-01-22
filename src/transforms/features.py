"""Feature engineering transform using Polars expressions."""

from typing import Any

import polars as pl

from dyxgb.transforms.base import StatelessTransform
from dyxgb.transforms.registry import build_expression


class FeatureTransform(StatelessTransform):
    """Create new features using human-readable functions or Polars expressions.

    Supports two syntax options:

    1. Human-readable function syntax (recommended):
        features:
            - name: amount_log
              function: log
              column: amount

            - name: age_squared
              function: square
              column: age

            - name: amount_per_age
              function: ratio
              columns: [amount, age]

            - name: is_high_value
              function: threshold
              column: amount
              value: 1000

    2. Raw Polars expression syntax (advanced users):
        features:
            - name: amount_log
              expr: "pl.col('amount').log1p()"
            - name: age_squared
              expr: "pl.col('age') ** 2"

    Available functions:
        Math: log, square, sqrt, abs, clip, ratio, difference, product, threshold, bin
        String: length, lower, upper, contains
        Date: dayofweek, month, year, days_since
        Null: fillna, is_null

    Use `dyxgb functions` to see all available functions with descriptions.
    """

    name = "features"

    def __init__(self, features: list[dict[str, Any]]) -> None:
        """Initialize feature transform.

        Args:
            features: List of feature definitions with 'name' and either
                     'function' + 'column'/'columns' or 'expr' keys
        """
        self.features = features

        # Validate feature definitions
        for i, feat in enumerate(features):
            if "name" not in feat:
                raise ValueError(f"Feature {i} missing 'name' key")
            if "expr" not in feat and "function" not in feat:
                raise ValueError(
                    f"Feature '{feat.get('name', i)}' requires either 'expr' or 'function' key"
                )

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply feature expressions to create new columns.

        Args:
            df: DataFrame to transform

        Returns:
            DataFrame with new feature columns added
        """
        for feat in self.features:
            name = feat["name"]

            try:
                if "function" in feat:
                    # New human-readable function syntax
                    expr = build_expression(feat)
                else:
                    # Legacy Polars expression syntax
                    expr_str = feat["expr"]
                    # Evaluate the Polars expression
                    # We provide 'pl' in the eval namespace
                    expr = eval(expr_str, {"pl": pl, "__builtins__": {}})

                # Apply the expression and alias with the feature name
                if isinstance(expr, pl.Expr):
                    df = df.with_columns(expr.alias(name))
                else:
                    raise ValueError(
                        f"Expression for '{name}' did not return a Polars expression. "
                        f"Got: {type(expr)}"
                    )
            except Exception as e:
                if "function" in feat:
                    raise ValueError(
                        f"Error building feature '{name}' with function "
                        f"'{feat['function']}': {e}"
                    ) from e
                else:
                    raise ValueError(
                        f"Error evaluating expression for feature '{name}': "
                        f"{feat.get('expr')}\nError: {e}"
                    ) from e

        return df

    def get_params(self) -> dict[str, list[dict[str, Any]]]:
        """Get features for serialization."""
        return {"features": self.features}

    def set_params(self, params: dict[str, list[dict[str, Any]]]) -> None:
        """Set features from deserialization."""
        self.features = params.get("features", [])

    def __repr__(self) -> str:
        feature_names = [f["name"] for f in self.features]
        return f"FeatureTransform({feature_names})"
