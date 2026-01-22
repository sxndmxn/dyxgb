"""Type casting transform."""

import polars as pl

from dyxgb.transforms.base import StatelessTransform

# Mapping from config strings to Polars dtypes
DTYPE_MAP = {
    "int": pl.Int64,
    "int8": pl.Int8,
    "int16": pl.Int16,
    "int32": pl.Int32,
    "int64": pl.Int64,
    "uint8": pl.UInt8,
    "uint16": pl.UInt16,
    "uint32": pl.UInt32,
    "uint64": pl.UInt64,
    "float": pl.Float64,
    "float32": pl.Float32,
    "float64": pl.Float64,
    "str": pl.Utf8,
    "string": pl.Utf8,
    "bool": pl.Boolean,
    "boolean": pl.Boolean,
    "date": pl.Date,
    "datetime": pl.Datetime,
    "time": pl.Time,
}


class CastTransform(StatelessTransform):
    """Cast columns to specified types.

    Example config:
        cast:
            amount: float
            age: int
            date: datetime
            is_active: bool

    Supported types:
        - int, int8, int16, int32, int64
        - uint8, uint16, uint32, uint64
        - float, float32, float64
        - str, string
        - bool, boolean
        - date, datetime, time
    """

    name = "cast"

    def __init__(self, types: dict[str, str]) -> None:
        """Initialize cast transform.

        Args:
            types: Dictionary mapping column names to type strings
        """
        self.types = types

        # Validate types at init time
        for col, type_str in types.items():
            if type_str.lower() not in DTYPE_MAP:
                valid = ", ".join(sorted(DTYPE_MAP.keys()))
                raise ValueError(
                    f"Unknown type '{type_str}' for column '{col}'. Valid types: {valid}"
                )

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Cast columns to specified types.

        Args:
            df: DataFrame to transform

        Returns:
            DataFrame with cast columns
        """
        existing_columns = set(df.columns)

        cast_exprs = []
        for col, type_str in self.types.items():
            if col not in existing_columns:
                continue

            dtype = DTYPE_MAP[type_str.lower()]

            # Handle datetime specially - may need parsing
            if dtype == pl.Datetime and df[col].dtype == pl.Utf8:
                cast_exprs.append(pl.col(col).str.to_datetime(strict=False).alias(col))
            elif dtype == pl.Date and df[col].dtype == pl.Utf8:
                cast_exprs.append(pl.col(col).str.to_date(strict=False).alias(col))
            else:
                cast_exprs.append(pl.col(col).cast(dtype).alias(col))

        if cast_exprs:
            return df.with_columns(cast_exprs)
        return df

    def get_params(self) -> dict[str, dict[str, str]]:
        """Get types for serialization."""
        return {"types": self.types}

    def set_params(self, params: dict[str, dict[str, str]]) -> None:
        """Set types from deserialization."""
        self.types = params.get("types", {})

    def __repr__(self) -> str:
        return f"CastTransform({len(self.types)} columns)"
