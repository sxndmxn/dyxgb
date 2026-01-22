# dyxgb - Dynamic XGBoost

A flexible CLI tool for XGBoost training and prediction with support for multiple data sources, hyperparameter tuning, and both classification and regression tasks.

**Unix Philosophy**: dyxgb is designed for composability. Commands work as Unix filters with clean stdin/stdout contracts, predictable exit codes, and pipe-friendly defaults.

## Features

- **Unix-Friendly**: stdin/stdout support, composable commands, clean I/O contracts
- **Multiple Data Sources**: Load data from CSV, Parquet, JSON files or databases (SQLite, DuckDB, PostgreSQL)
- **Classification & Regression**: Support for both task types with appropriate metrics
- **Hyperparameter Tuning**: Integrated Optuna for automated hyperparameter optimization
- **Interactive Mode**: InquirerPy-powered prompts for exploratory workflows
- **Single Artifact**: Model, encoder, and pipeline bundled in one `.dyxgb` file
- **Feature Importance**: Extract and export feature importance scores
- **Evaluation Metrics**: Comprehensive metrics for model evaluation

## Installation

```bash
# Clone the repository
git clone https://github.com/sxndmxn/dyxgb.git
cd dyxgb

# Install with uv (recommended)
uv sync

# With optional dependencies
uv sync --all-extras

# Or with pip
pip install -e .
pip install -e ".[all]"  # with all extras
```

### Optional Dependencies

```bash
# Interactive mode (InquirerPy prompts)
pip install -e ".[interactive]"

# Hyperparameter tuning (Optuna)
pip install -e ".[tuning]"

# PostgreSQL support
pip install -e ".[postgres]"

# Everything
pip install -e ".[all]"
```

## CLI Contract

dyxgb follows Unix philosophy conventions:

### I/O Behavior

| Stream | Content |
|--------|---------|
| **stdout** | Data only (CSV, JSONL, JSON) - never logs or progress |
| **stderr** | Logs, progress, human-readable output |

### Exit Codes

| Code | Meaning |
|------|---------|
| **0** | Success |
| **1** | Runtime error (I/O failure, model load failure, etc.) |
| **2** | Usage error (invalid args, unsupported formats, etc.) |

### Default Behavior

| Command | stdin | stdout | Default Format |
|---------|-------|--------|----------------|
| `train` | Rejected (exit 2) | N/A | `.dyxgb` bundle file |
| `predict` | CSV/JSONL | Predictions | CSV |
| `evaluate` | N/A | Metrics | JSON object |
| `importance` | N/A | Feature importance | CSV |

## Unix Pipelines

dyxgb commands are designed for composition:

```bash
# Predict as a filter (stdin -> stdout)
dyxgb predict --model model.dyxgb < new.csv > preds.csv

# JSONL input/output
cat data.jsonl | dyxgb predict -m model.dyxgb --input-format jsonl --output-format jsonl > preds.jsonl

# Evaluate and parse with jq
dyxgb evaluate -s test.csv -m model.dyxgb --target y | jq '.accuracy'

# Get top 5 features as CSV
dyxgb importance -m model.dyxgb --top 5 > top_features.csv

# Chain with other tools
dyxgb importance -m model.dyxgb | head -6 | column -t -s,

# Combine train + evaluate
dyxgb train -s train.csv --target y -o model.dyxgb && \
  dyxgb evaluate -s test.csv -m model.dyxgb --target y > metrics.json
```

## Quick Start

### Train a Model

```bash
# Basic training (outputs model.dyxgb bundle)
dyxgb train \
  --source data/train.csv \
  --target label \
  --output model.dyxgb

# With hyperparameter tuning
dyxgb train \
  --source data/train.parquet \
  --target price \
  --task regression \
  --tune \
  --tune-trials 100
```

### Make Predictions

```bash
# From file
dyxgb predict \
  --source data/new_data.csv \
  --model model.dyxgb \
  --output predictions.csv

# From stdin (pipe-friendly default)
dyxgb predict --model model.dyxgb < new_data.csv > preds.csv

# JSONL format
dyxgb predict -m model.dyxgb \
  --input-format jsonl \
  --output-format jsonl < data.jsonl
```

### Evaluate Model

```bash
# Get metrics as JSON (default: stdout)
dyxgb evaluate \
  --source test.csv \
  --model model.dyxgb \
  --target label

# Save to file
dyxgb evaluate -s test.csv -m model.dyxgb --target y --output metrics.json
```

### Feature Importance

```bash
# CSV to stdout (default)
dyxgb importance --model model.dyxgb

# Top N features
dyxgb importance -m model.dyxgb --top 10

# JSONL format
dyxgb importance -m model.dyxgb --output-format jsonl

# Save to file
dyxgb importance -m model.dyxgb --output importance.parquet
```

### Interactive Mode

```bash
# Run interactive wizard (requires interactive extra)
dyxgb interactive
```

## Database Sources

```bash
# PostgreSQL
dyxgb train \
  --source "postgres://user:pass@localhost:5432/mydb" \
  --query "SELECT * FROM training_data" \
  --target label

# DuckDB
dyxgb train \
  --source "duckdb:///data/analytics.duckdb" \
  --table "features" \
  --target churn
```

## Configuration File

Create a `config.yaml` file for reproducible workflows:

```yaml
data:
  train:
    type: file
    path: "data/train.parquet"

transforms:
  rename:
    txn_amt: amount
  cast:
    amount: float
    age: int
  missing:
    strategy: median
  features:
    - name: amount_log
      function: log
      column: amount
    - name: amount_per_age
      function: ratio
      columns: [amount, age]
  encode:
    columns:
      - category
  scale:
    method: standard
    columns:
      - amount
      - amount_log

model:
  task: classification
  target: label
  features:
    - amount
    - age
    - amount_log
    - amount_per_age
    - category
  hyperparameters:
    n_estimators: 300
    max_depth: 6
    learning_rate: 0.1
```

```bash
# Train with config
dyxgb train --config config.yaml

# Override config options
dyxgb train --config config.yaml --tune --tune-trials 200
```

See `config.yaml` or `config.example.yaml` for a complete example.

### Feature Engineering

Feature engineering lives under `transforms` and runs in a fixed order:
rename -> cast -> missing -> features -> encode -> scale. The pipeline is
fitted on training data and stored in the `.dyxgb` bundle, so predict and
evaluate reuse the exact same transforms.

Each entry in `transforms.features` needs a `name` and either:
- `function` plus `column` or `columns` for the built-in registry (list with `dyxgb functions`)
- `expr` for a raw Polars expression string (advanced; `pl` is in scope)

Features run after missing-value handling and before encoding/scaling, so add
any derived categoricals to `encode.columns` and derived numerics to
`scale.columns` (or omit `scale.columns` to auto-scale). `model.features`
should reference the post-transform column names.

## Output Schemas

### Predictions

**Classification** (CSV/JSONL):
```
predicted_label,confidence,prob_ClassA,prob_ClassB
A,0.85,0.85,0.15
B,0.92,0.08,0.92
```

**Regression** (CSV/JSONL):
```
predicted_value
123.45
67.89
```

### Evaluation Metrics (JSON)

**Classification**:
```json
{
  "accuracy": 0.92,
  "precision": 0.91,
  "recall": 0.93,
  "f1": 0.92,
  "roc_auc": 0.96
}
```

**Regression**:
```json
{
  "mse": 0.0234,
  "rmse": 0.153,
  "mae": 0.12,
  "r2": 0.94
}
```

### Feature Importance (CSV)

```
feature,importance
feature_3,0.4521
feature_1,0.3234
feature_2,0.2245
```

## Supported Data Sources

| Source | URI Format | Example |
|--------|------------|---------|
| CSV | File path | `data/train.csv` |
| Parquet | File path | `data/train.parquet` |
| JSON | File path | `data/train.json` |
| JSONL/NDJSON | File path | `data/train.jsonl` |
| SQLite | `sqlite:///path` | `sqlite:///data/db.sqlite` |
| DuckDB | `duckdb:///path` | `duckdb:///data/analytics.duckdb` |
| PostgreSQL | `postgres://...` | `postgres://user:pass@host:5432/db` |

## Legacy Compatibility

The new bundle format (`.dyxgb`) is preferred, but legacy separate files are still supported:

```bash
# Legacy format (model.json + encoder.joblib)
dyxgb predict \
  --source data.csv \
  --model model.json \
  --encoder encoder.joblib \
  --output predictions.csv
```

## CLI Reference

```bash
# Show help
dyxgb --help

# Command-specific help
dyxgb train --help
dyxgb predict --help
dyxgb evaluate --help
dyxgb importance --help
dyxgb interactive --help
```

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest

# Run contract tests specifically
uv run pytest tests/test_cli_contracts.py -v

# Type checking
uv run mypy src/dyxgb

# Linting
uv run ruff check src/dyxgb
```

## License

MIT
