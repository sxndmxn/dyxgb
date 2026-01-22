"""Function registry for human-readable feature engineering.

This module provides a registry of common data transformation functions
that can be used without knowing Polars syntax.
"""

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

import polars as pl


@dataclass
class FunctionSpec:
    """Specification for a registered function."""

    name: str
    description: str
    category: str
    builder: Callable[..., pl.Expr]
    params: list[str]  # Required parameters beyond column(s)
    example: str


def _log(column: str, **kwargs: Any) -> pl.Expr:
    """Natural log plus 1 (log1p) to handle zeros."""
    return pl.col(column).log1p()


def _square(column: str, **kwargs: Any) -> pl.Expr:
    """Square the values."""
    return pl.col(column).pow(2)


def _sqrt(column: str, **kwargs: Any) -> pl.Expr:
    """Square root of values."""
    return pl.col(column).sqrt()


def _abs(column: str, **kwargs: Any) -> pl.Expr:
    """Absolute value."""
    return pl.col(column).abs()


def _clip(column: str, min_val: float | None = None, max_val: float | None = None, **kwargs: Any) -> pl.Expr:
    """Clip values to a range."""
    expr = pl.col(column)
    if min_val is not None:
        expr = expr.clip(lower_bound=min_val)
    if max_val is not None:
        expr = expr.clip(upper_bound=max_val)
    return expr


def _ratio(columns: list[str], **kwargs: Any) -> pl.Expr:
    """Ratio of first column to second (col1 / col2)."""
    if len(columns) != 2:
        raise ValueError("ratio requires exactly 2 columns")
    return pl.col(columns[0]) / (pl.col(columns[1]) + 1e-10)  # Avoid division by zero


def _difference(columns: list[str], **kwargs: Any) -> pl.Expr:
    """Difference between columns (col1 - col2)."""
    if len(columns) != 2:
        raise ValueError("difference requires exactly 2 columns")
    return pl.col(columns[0]) - pl.col(columns[1])


def _product(columns: list[str], **kwargs: Any) -> pl.Expr:
    """Product of columns (col1 * col2)."""
    if len(columns) != 2:
        raise ValueError("product requires exactly 2 columns")
    return pl.col(columns[0]) * pl.col(columns[1])


def _threshold(column: str, value: float, **kwargs: Any) -> pl.Expr:
    """Boolean: is value greater than threshold."""
    return pl.col(column) > value


def _bin(column: str, bins: list[float], labels: list[str] | None = None, **kwargs: Any) -> pl.Expr:
    """Bin values into discrete buckets."""
    # Create when/then/otherwise chain for binning
    expr = pl.col(column)
    n_bins = len(bins) + 1

    # Generate default labels if not provided
    if labels is None:
        labels = []
        labels.append(f"<{bins[0]}")
        for i in range(len(bins) - 1):
            labels.append(f"{bins[i]}-{bins[i+1]}")
        labels.append(f">={bins[-1]}")

    if len(labels) != n_bins:
        raise ValueError(f"Expected {n_bins} labels for {len(bins)} bin edges, got {len(labels)}")

    # Build the expression using cut
    return expr.cut(bins, labels=labels)


def _length(column: str, **kwargs: Any) -> pl.Expr:
    """Length of string."""
    return pl.col(column).str.len_chars()


def _lower(column: str, **kwargs: Any) -> pl.Expr:
    """Convert string to lowercase."""
    return pl.col(column).str.to_lowercase()


def _upper(column: str, **kwargs: Any) -> pl.Expr:
    """Convert string to uppercase."""
    return pl.col(column).str.to_uppercase()


def _contains(column: str, pattern: str, **kwargs: Any) -> pl.Expr:
    """Check if string contains pattern."""
    return pl.col(column).str.contains(pattern)


def _dayofweek(column: str, **kwargs: Any) -> pl.Expr:
    """Day of week (Monday=1, Sunday=7)."""
    return pl.col(column).dt.weekday()


def _month(column: str, **kwargs: Any) -> pl.Expr:
    """Month of year (1-12)."""
    return pl.col(column).dt.month()


def _year(column: str, **kwargs: Any) -> pl.Expr:
    """Year."""
    return pl.col(column).dt.year()


def _days_since(column: str, reference_date: str | None = None, **kwargs: Any) -> pl.Expr:
    """Days since a reference date (default: today)."""
    if reference_date:
        ref = pl.lit(reference_date).str.to_date()
    else:
        ref = pl.lit(date.today())
    return (ref - pl.col(column)).dt.total_days()


def _fillna(column: str, value: Any, **kwargs: Any) -> pl.Expr:
    """Fill null values with a constant."""
    return pl.col(column).fill_null(value)


def _is_null(column: str, **kwargs: Any) -> pl.Expr:
    """Boolean: is value null."""
    return pl.col(column).is_null()


# Registry mapping function names to their specifications
FUNCTION_REGISTRY: dict[str, FunctionSpec] = {
    # Math functions
    "log": FunctionSpec(
        name="log",
        description="Natural log plus 1 (log1p) - safe for zeros",
        category="math",
        builder=_log,
        params=[],
        example="function: log\ncolumn: amount",
    ),
    "square": FunctionSpec(
        name="square",
        description="Square the values (x^2)",
        category="math",
        builder=_square,
        params=[],
        example="function: square\ncolumn: age",
    ),
    "sqrt": FunctionSpec(
        name="sqrt",
        description="Square root of values",
        category="math",
        builder=_sqrt,
        params=[],
        example="function: sqrt\ncolumn: variance",
    ),
    "abs": FunctionSpec(
        name="abs",
        description="Absolute value",
        category="math",
        builder=_abs,
        params=[],
        example="function: abs\ncolumn: difference",
    ),
    "clip": FunctionSpec(
        name="clip",
        description="Clip values to a min/max range",
        category="math",
        builder=_clip,
        params=["min_val", "max_val"],
        example="function: clip\ncolumn: score\nmin_val: 0\nmax_val: 100",
    ),
    "ratio": FunctionSpec(
        name="ratio",
        description="Ratio of two columns (col1 / col2)",
        category="math",
        builder=_ratio,
        params=[],
        example="function: ratio\ncolumns: [amount, quantity]",
    ),
    "difference": FunctionSpec(
        name="difference",
        description="Difference between two columns (col1 - col2)",
        category="math",
        builder=_difference,
        params=[],
        example="function: difference\ncolumns: [end_date, start_date]",
    ),
    "product": FunctionSpec(
        name="product",
        description="Product of two columns (col1 * col2)",
        category="math",
        builder=_product,
        params=[],
        example="function: product\ncolumns: [price, quantity]",
    ),
    "threshold": FunctionSpec(
        name="threshold",
        description="Boolean: is value greater than threshold",
        category="math",
        builder=_threshold,
        params=["value"],
        example="function: threshold\ncolumn: amount\nvalue: 1000",
    ),
    "bin": FunctionSpec(
        name="bin",
        description="Bin values into discrete buckets",
        category="math",
        builder=_bin,
        params=["bins"],
        example="function: bin\ncolumn: age\nbins: [18, 35, 55, 70]\nlabels: [young, adult, middle, senior, elderly]",
    ),
    # String functions
    "length": FunctionSpec(
        name="length",
        description="Length of string (character count)",
        category="string",
        builder=_length,
        params=[],
        example="function: length\ncolumn: name",
    ),
    "lower": FunctionSpec(
        name="lower",
        description="Convert string to lowercase",
        category="string",
        builder=_lower,
        params=[],
        example="function: lower\ncolumn: category",
    ),
    "upper": FunctionSpec(
        name="upper",
        description="Convert string to uppercase",
        category="string",
        builder=_upper,
        params=[],
        example="function: upper\ncolumn: code",
    ),
    "contains": FunctionSpec(
        name="contains",
        description="Boolean: does string contain pattern",
        category="string",
        builder=_contains,
        params=["pattern"],
        example="function: contains\ncolumn: email\npattern: '@gmail'",
    ),
    # Date functions
    "dayofweek": FunctionSpec(
        name="dayofweek",
        description="Day of week (Monday=1, Sunday=7)",
        category="date",
        builder=_dayofweek,
        params=[],
        example="function: dayofweek\ncolumn: transaction_date",
    ),
    "month": FunctionSpec(
        name="month",
        description="Month of year (1-12)",
        category="date",
        builder=_month,
        params=[],
        example="function: month\ncolumn: created_at",
    ),
    "year": FunctionSpec(
        name="year",
        description="Extract year from date",
        category="date",
        builder=_year,
        params=[],
        example="function: year\ncolumn: birth_date",
    ),
    "days_since": FunctionSpec(
        name="days_since",
        description="Days since reference date (default: today)",
        category="date",
        builder=_days_since,
        params=["reference_date"],
        example="function: days_since\ncolumn: last_purchase\nreference_date: '2024-01-01'",
    ),
    # Null handling functions
    "fillna": FunctionSpec(
        name="fillna",
        description="Fill null values with a constant",
        category="null",
        builder=_fillna,
        params=["value"],
        example="function: fillna\ncolumn: score\nvalue: 0",
    ),
    "is_null": FunctionSpec(
        name="is_null",
        description="Boolean: is value null",
        category="null",
        builder=_is_null,
        params=[],
        example="function: is_null\ncolumn: optional_field",
    ),
}


def get_function(name: str) -> FunctionSpec:
    """Get a function specification by name.

    Args:
        name: Function name

    Returns:
        FunctionSpec for the function

    Raises:
        ValueError: If function not found
    """
    if name not in FUNCTION_REGISTRY:
        available = ", ".join(sorted(FUNCTION_REGISTRY.keys()))
        raise ValueError(f"Unknown function '{name}'. Available functions: {available}")
    return FUNCTION_REGISTRY[name]


def build_expression(feature_config: dict[str, Any]) -> pl.Expr:
    """Build a Polars expression from a feature config dict.

    Args:
        feature_config: Dict with 'function', 'column'/'columns', and optional params

    Returns:
        Polars expression

    Raises:
        ValueError: If function not found or required params missing
    """
    func_name = feature_config.get("function")
    if not func_name:
        raise ValueError("Feature config missing 'function' key")

    spec = get_function(func_name)

    # Get column(s)
    column = feature_config.get("column")
    columns = feature_config.get("columns")

    # Build kwargs for the function
    kwargs: dict[str, Any] = {}

    # Add any extra parameters from config
    for key, value in feature_config.items():
        if key not in ("name", "function", "column", "columns"):
            kwargs[key] = value

    # Call the builder with column or columns
    if columns:
        return spec.builder(columns=columns, **kwargs)
    elif column:
        return spec.builder(column=column, **kwargs)
    else:
        raise ValueError(f"Feature config for '{func_name}' requires 'column' or 'columns'")


def list_functions(category: str | None = None) -> list[FunctionSpec]:
    """List all available functions, optionally filtered by category.

    Args:
        category: Filter by category (math, string, date, null)

    Returns:
        List of FunctionSpec objects
    """
    funcs = list(FUNCTION_REGISTRY.values())
    if category:
        funcs = [f for f in funcs if f.category == category]
    return sorted(funcs, key=lambda f: (f.category, f.name))


def get_categories() -> list[str]:
    """Get list of all function categories."""
    return sorted(set(f.category for f in FUNCTION_REGISTRY.values()))
