"""Feature engineering transform using Polars expressions."""

from typing import Any

import polars as pl

from dyxgb.transforms.base import StatelessTransform


class FeatureTransform(StatelessTransform):
    """Create new features using Polars expressions.

    Allows defining computed columns using the full power of Polars
    expression syntax.

    Example config:
        features:
            - name: amount_log
              expr: "pl.col('amount').log1p()"
            - name: age_squared
              expr: "pl.col('age') ** 2"
            - name: amount_per_age
              expr: "pl.col('amount') / (pl.col('age') + 1)"
            - name: is_high_value
              expr: "pl.col('amount') > 1000"
            - name: category_amount
              expr: "pl.col('category') + '_' + pl.col('amount').cast(pl.Utf8)"

    Expression Syntax:
        - Use pl.col('column_name') to reference columns
        - Use standard Python operators: +, -, *, /, **, //, %
        - Use Polars methods: .log(), .log1p(), .sqrt(), .abs(), etc.
        - Use conditionals: pl.when(...).then(...).otherwise(...)
        - Use string methods: .str.to_lowercase(), .str.contains(), etc.

    Security Note:
        Expressions are evaluated using eval(). Only use with trusted config files.
    """

    name = "features"

    def __init__(self, features: list[dict[str, str]]) -> None:
        """Initialize feature transform.

        Args:
            features: List of feature definitions, each with 'name' and 'expr' keys
        """
        self.features = features

        # Validate feature definitions
        for i, feat in enumerate(features):
            if "name" not in feat:
                raise ValueError(f"Feature {i} missing 'name' key")
            if "expr" not in feat:
                raise ValueError(f"Feature '{feat.get('name', i)}' missing 'expr' key")

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply feature expressions to create new columns.

        Args:
            df: DataFrame to transform

        Returns:
            DataFrame with new feature columns added
        """
        for feat in self.features:
            name = feat["name"]
            expr_str = feat["expr"]

            try:
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
                raise ValueError(
                    f"Error evaluating expression for feature '{name}': {expr_str}\nError: {e}"
                ) from e

        return df

    def get_params(self) -> dict[str, list[dict[str, str]]]:
        """Get features for serialization."""
        return {"features": self.features}

    def set_params(self, params: dict[str, list[dict[str, str]]]) -> None:
        """Set features from deserialization."""
        self.features = params.get("features", [])

    def __repr__(self) -> str:
        feature_names = [f["name"] for f in self.features]
        return f"FeatureTransform({feature_names})"
