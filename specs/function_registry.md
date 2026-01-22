# Function Registry Specification

## Overview
A human-readable function dictionary for feature engineering that allows users to create features without knowing Polars syntax.

## Config Syntax

### New Function-Based Syntax
```yaml
features:
  - name: amount_log
    function: log
    column: amount

  - name: ratio_feature
    function: ratio
    columns: [amount, age]

  - name: high_value_flag
    function: threshold
    column: amount
    value: 1000
```

### Legacy Expression Syntax (still supported)
```yaml
features:
  - name: custom_feature
    expr: "pl.col('amount').rolling_mean(window_size=7)"
```

## Function Definitions

### Math Functions

| Function | Description | Parameters | Polars Expression |
|----------|-------------|------------|-------------------|
| `log` | Natural logarithm (log1p for safety) | column | `pl.col(column).log1p()` |
| `square` | Square the value | column | `pl.col(column) ** 2` |
| `sqrt` | Square root | column | `pl.col(column).sqrt()` |
| `abs` | Absolute value | column | `pl.col(column).abs()` |
| `clip` | Cap values at min/max | column, min, max | `pl.col(column).clip(min, max)` |
| `ratio` | Divide two columns | columns[2] | `pl.col(columns[0]) / pl.col(columns[1])` |
| `difference` | Subtract two columns | columns[2] | `pl.col(columns[0]) - pl.col(columns[1])` |
| `product` | Multiply two columns | columns[2] | `pl.col(columns[0]) * pl.col(columns[1])` |
| `threshold` | Binary flag (1 if > value) | column, value | `pl.when(pl.col(column) > value).then(1).otherwise(0)` |
| `bin` | Bucket into ranges | column, bins | `pl.col(column).cut(bins)` |

### String Functions

| Function | Description | Parameters | Polars Expression |
|----------|-------------|------------|-------------------|
| `length` | String length | column | `pl.col(column).str.len_chars()` |
| `lower` | Lowercase string | column | `pl.col(column).str.to_lowercase()` |
| `upper` | Uppercase string | column | `pl.col(column).str.to_uppercase()` |
| `contains` | Contains pattern (binary) | column, pattern | `pl.col(column).str.contains(pattern).cast(pl.Int8)` |

### Date Functions

| Function | Description | Parameters | Polars Expression |
|----------|-------------|------------|-------------------|
| `dayofweek` | Day of week (0-6) | column | `pl.col(column).dt.weekday()` |
| `month` | Month (1-12) | column | `pl.col(column).dt.month()` |
| `year` | Year | column | `pl.col(column).dt.year()` |
| `days_since` | Days since date | column | `(pl.lit(datetime.now()) - pl.col(column)).dt.total_days()` |

### Null Functions

| Function | Description | Parameters | Polars Expression |
|----------|-------------|------------|-------------------|
| `fillna` | Replace nulls with value | column, value | `pl.col(column).fill_null(value)` |
| `is_null` | Null indicator (binary) | column | `pl.col(column).is_null().cast(pl.Int8)` |

## Implementation Details

### Registry Structure (registry.py)
```python
FUNCTION_REGISTRY = {
    "log": {
        "description": "Natural logarithm (log1p for safety)",
        "params": ["column"],
        "builder": lambda column: pl.col(column).log1p(),
    },
    # ... more functions
}
```

### Config Dataclass Updates (config.py)
```python
@dataclass
class FeatureConfig:
    name: str
    # New fields for function-based syntax
    function: str | None = None
    column: str | None = None
    columns: list[str] | None = None
    value: float | None = None
    min: float | None = None
    max: float | None = None
    pattern: str | None = None
    bins: list[float] | None = None
    # Legacy field
    expr: str | None = None
```

### CLI Command (cli.py)
```bash
$ dyxgb functions

Available Feature Engineering Functions:

Math Functions:
  log         Natural logarithm (log1p for safety)      column
  square      Square the value                          column
  ...

String Functions:
  length      String length                             column
  ...
```

## Validation Rules
1. Must have either `function` or `expr`, not both
2. If `function` is set, required params must be present
3. `columns` must have exactly 2 items for ratio/difference/product
4. `bins` must be a sorted list for bin function
