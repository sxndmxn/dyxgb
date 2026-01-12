# dyxgb - Dynamic XGBoost

A flexible CLI tool for XGBoost training and prediction with support for multiple data sources, hyperparameter tuning, and both classification and regression tasks.

## Features

- **Multiple Data Sources**: Load data from CSV, Parquet, JSON files or databases (SQLite, DuckDB, PostgreSQL)
- **Classification & Regression**: Support for both task types with appropriate metrics
- **Hyperparameter Tuning**: Integrated Optuna for automated hyperparameter optimization
- **Interactive Mode**: InquirerPy-powered prompts for exploratory workflows
- **Batch Mode**: Config file or CLI arguments for reproducible pipelines
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

## Quick Start

### Interactive Mode

```bash
# Run interactive wizard
uv run dyxgb interactive
```

### Train from File

```bash
# Basic training
uv run dyxgb train \
  --source data/train.csv \
  --target label \
  --features "feature_1,feature_2,feature_3" \
  --output model.json

# With hyperparameter tuning
uv run dyxgb train \
  --source data/train.parquet \
  --target price \
  --task regression \
  --tune \
  --tune-trials 100 \
  --output model.json
```

### Train from Database

```bash
# From PostgreSQL
uv run dyxgb train \
  --source "postgres://user:pass@localhost:5432/mydb" \
  --query "SELECT * FROM training_data" \
  --target label \
  --output model.json

# From DuckDB
uv run dyxgb train \
  --source "duckdb:///data/analytics.duckdb" \
  --table "features" \
  --target churn \
  --output model.json
```

### Train with Config File

```bash
# Using YAML config
uv run dyxgb train --config config.yaml

# Override config options
uv run dyxgb train --config config.yaml --tune --tune-trials 200
```

### Make Predictions

```bash
# Basic prediction
uv run dyxgb predict \
  --source data/new_data.csv \
  --model model.json \
  --encoder label_encoder.joblib \
  --output predictions.parquet

# From database
uv run dyxgb predict \
  --source "postgres://..." \
  --query "SELECT * FROM new_customers" \
  --model model.json \
  --output predictions.csv
```

### Evaluate Model

```bash
uv run dyxgb evaluate \
  --source data/test.csv \
  --model model.json \
  --target label \
  --task classification
```

### Feature Importance

```bash
# Display top features
uv run dyxgb importance --model model.json --top 20

# Export to file
uv run dyxgb importance --model model.json --output importance.json
```

## Configuration File

Create a `config.yaml` file for reproducible workflows:

```yaml
data:
  train:
    type: file
    path: "data/train.parquet"
  predict:
    type: postgres
    uri: "postgres://user:pass@localhost/db"
    query: "SELECT * FROM new_data"

model:
  task: classification
  target: label
  features:
    - feature_1
    - feature_2
  hyperparameters:
    n_estimators: 300
    max_depth: 6
    learning_rate: 0.1

tuning:
  enabled: true
  n_trials: 50
  metric: f1_weighted

output:
  model_path: "models/model.json"
  encoder_path: "models/encoder.joblib"
  predictions_path: "output/predictions.parquet"
```

See `config.example.yaml` for a complete example.

## Supported Data Sources

| Source | URI Format | Example |
|--------|------------|---------|
| CSV | File path | `data/train.csv` |
| Parquet | File path | `data/train.parquet` |
| JSON | File path | `data/train.json` |
| SQLite | `sqlite:///path` | `sqlite:///data/db.sqlite` |
| DuckDB | `duckdb:///path` | `duckdb:///data/analytics.duckdb` |
| PostgreSQL | `postgres://...` | `postgres://user:pass@host:5432/db` |

## CLI Reference

```bash
# Show help
uv run dyxgb --help

# Command-specific help
uv run dyxgb train --help
uv run dyxgb predict --help
uv run dyxgb evaluate --help
uv run dyxgb importance --help
uv run dyxgb interactive --help
```

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest

# Type checking
uv run mypy src/dyxgb

# Linting
uv run ruff check src/dyxgb
```

## License

MIT
