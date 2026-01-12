# dyxgb Test Data & Stress Testing

This directory contains comprehensive test data and stress testing tools for dyxgb.

## 📁 Directory Contents

### Test Datasets

All datasets are available in three formats: CSV, Parquet, and JSON (NDJSON)

#### Classification Datasets
- **classification_small** (100 rows) - Quick testing, simple features
- **classification_medium** (1,000 rows) - Standard testing, moderate complexity
- **classification_large** (10,000 rows) - Performance testing, complex features
- **classification_xlarge** (50,000 rows) - Heavy stress testing, complex features
- **classification_train/test** (800/200 rows) - Pre-split for evaluation testing
- **multiclass_classification** (2,000 rows) - 5-class classification with imbalanced classes

#### Regression Datasets
- **regression_small** (100 rows) - Quick testing
- **regression_medium** (1,000 rows) - Standard testing
- **regression_large** (10,000 rows) - Performance testing
- **regression_train/test** (800/200 rows) - Pre-split for evaluation testing

#### Special Datasets
- **edge_cases** (500 rows) - Extreme edge cases:
  - Very high percentage of missing values (30-35%)
  - Extreme value ranges (tiny: 1e-10, huge: 1e10)
  - Zero/near-zero variance features
  - Highly correlated features
  - Sparse categorical features
  - Infinity values

#### Database Sources
- **test_data.sqlite** - SQLite database with train_data and test_data tables
- **test_data.duckdb** - DuckDB database with train_data and test_data tables

### Configuration Files

Pre-configured YAML files for different scenarios:

- **config_classification_basic.yaml** - Simple classification without transforms
- **config_classification_advanced.yaml** - Advanced with all transform types
- **config_regression.yaml** - Regression with feature engineering
- **config_tuning.yaml** - Hyperparameter tuning with Optuna
- **config_database.yaml** - Database source example
- **config_multiclass.yaml** - Multi-class classification

### Scripts

- **generate_test_data.py** - Generates all test datasets
- **stress_test.py** - Comprehensive stress testing suite

## 🎯 Edge Cases Covered

The test data covers all these edge cases:

### Data Quality Issues
- ✅ Missing values (5-35% depending on dataset)
- ✅ Outliers and extreme values
- ✅ Infinity values in numeric columns
- ✅ Zero/near-zero variance features
- ✅ High cardinality categorical features
- ✅ Sparse categorical values (rare categories)

### Data Distributions
- ✅ Imbalanced classes (70-30, 80-20 splits)
- ✅ Multi-class imbalance (40%, 25%, 20%, 10%, 5%)
- ✅ Log-normal distributions
- ✅ Exponential distributions
- ✅ Normal distributions
- ✅ Uniform distributions

### Data Types
- ✅ Numeric (integer, float)
- ✅ Categorical (low, medium, high cardinality)
- ✅ Boolean
- ✅ Mixed data types in single dataset

### Feature Characteristics
- ✅ Highly correlated features
- ✅ Features with different scales (0-1, 1000-10000, etc.)
- ✅ Constant features
- ✅ Near-constant features
- ✅ Derived/engineered features

### Dataset Sizes
- ✅ Small (100 rows) - Quick iteration
- ✅ Medium (1,000 rows) - Standard testing
- ✅ Large (10,000 rows) - Performance testing
- ✅ Extra large (50,000 rows) - Stress testing

## 🚀 Quick Start

### 1. Generate Test Data

```bash
cd test_data
python3 generate_test_data.py
```

This creates all datasets in CSV, Parquet, and JSON formats, plus SQLite and DuckDB databases.

### 2. Run Basic Test

```bash
cd /home/runner/work/dyxgb/dyxgb

# Classification
python3 -m dyxgb.cli train \
  --source test_data/classification_small.csv \
  --target label \
  --task classification \
  --output test_data/models/my_model.json

# Regression
python3 -m dyxgb.cli train \
  --source test_data/regression_small.csv \
  --target price \
  --task regression \
  --output test_data/models/regression_model.json
```

### 3. Run with Config File

```bash
python3 -m dyxgb.cli train --config test_data/config_classification_basic.yaml
```

### 4. Run Full Stress Test

```bash
cd test_data
python3 stress_test.py
```

The stress test runs 15+ comprehensive tests covering:
- Different file formats (CSV, Parquet, JSON)
- Different data sources (files, SQLite, DuckDB)
- Different task types (binary classification, multi-class, regression)
- Different dataset sizes
- Config file workflows
- Transform pipelines
- All CLI commands (train, predict, evaluate, importance)
- Edge cases

## 📊 Test Scenarios

### Scenario 1: Basic Classification
```bash
python3 -m dyxgb.cli train \
  --config test_data/config_classification_basic.yaml
```

### Scenario 2: Advanced with Transforms
```bash
python3 -m dyxgb.cli train \
  --config test_data/config_classification_advanced.yaml
```

Tests:
- Missing value imputation (median, mode)
- Feature engineering (ratios, log transforms, conditionals)
- Categorical encoding
- Feature scaling

### Scenario 3: Regression
```bash
python3 -m dyxgb.cli train \
  --config test_data/config_regression.yaml
```

### Scenario 4: Hyperparameter Tuning
```bash
# Requires optuna: pip install optuna
python3 -m dyxgb.cli train \
  --config test_data/config_tuning.yaml
```

### Scenario 5: Database Sources
```bash
# SQLite
python3 -m dyxgb.cli train \
  --source "sqlite:///test_data/test_data.sqlite" \
  --table train_data \
  --target label \
  --output test_data/models/db_model.json

# DuckDB
python3 -m dyxgb.cli train \
  --source "duckdb:///test_data/test_data.duckdb" \
  --table train_data \
  --target label \
  --output test_data/models/duckdb_model.json
```

### Scenario 6: Complete Workflow
```bash
# Train
python3 -m dyxgb.cli train \
  --source test_data/classification_train.csv \
  --target label \
  --output test_data/models/workflow_model.json \
  --encoder-output test_data/models/workflow_encoder.joblib

# Predict
python3 -m dyxgb.cli predict \
  --source test_data/classification_test.csv \
  --model test_data/models/workflow_model.json \
  --encoder test_data/models/workflow_encoder.joblib \
  --output test_data/output/predictions.csv

# Evaluate
python3 -m dyxgb.cli evaluate \
  --source test_data/classification_test.csv \
  --model test_data/models/workflow_model.json \
  --encoder test_data/models/workflow_encoder.joblib \
  --target label

# Feature Importance
python3 -m dyxgb.cli importance \
  --model test_data/models/workflow_model.json \
  --output test_data/output/importance.json
```

## 🧪 Stress Test Details

The `stress_test.py` script runs comprehensive tests:

| Test # | Description | Dataset | Features Tested |
|--------|-------------|---------|-----------------|
| 1 | Basic CSV Classification | Small (100) | CSV format, basic workflow |
| 2 | Parquet Medium | Medium (1k) | Parquet format, larger dataset |
| 3 | JSON Format | Small (100) | JSON/NDJSON format |
| 4 | Regression Task | Medium (1k) | Regression vs classification |
| 5 | Large Dataset | Large (10k) | Performance, memory usage |
| 6 | Extra Large Dataset | XLarge (50k) | Heavy stress test |
| 7 | Multi-class | Medium (2k) | 5-class classification |
| 8 | Config Basic | Medium (1k) | Config file workflow |
| 9 | Config Advanced | Medium (1k) | All transform types |
| 10 | Config Regression | Medium (1k) | Regression config |
| 11 | Prediction Command | Test split | predict CLI command |
| 12 | Evaluation Command | Test split | evaluate CLI command |
| 13 | Importance Command | Test split | importance CLI command |
| 14 | Edge Cases | Edge (500) | Extreme values, missing data |
| 15 | SQLite Database | Medium (1k) | Database loading |
| 16 | DuckDB Database | Medium (1k) | DuckDB integration |

### Running Individual Tests

You can modify `stress_test.py` to run specific tests by commenting out tests in the `tests` list.

### Interpreting Results

The stress test outputs:
1. Real-time progress with color-coded status
2. Execution time for each test
3. Summary statistics (total, passed, failed, time)
4. JSON results file: `stress_test_results.json`

Example output:
```
============================================================
                    TEST SUMMARY
============================================================

Total Tests: 15
Passed: 14
Failed: 1
Total Time: 245.67s (4.09 minutes)
```

## 📈 Performance Benchmarks

Expected execution times (approximate, varies by hardware):

| Dataset Size | Training Time | Memory Usage |
|--------------|---------------|--------------|
| Small (100) | < 2s | < 100 MB |
| Medium (1k) | 2-5s | ~200 MB |
| Large (10k) | 10-30s | ~500 MB |
| XLarge (50k) | 60-180s | ~2 GB |

## 🔍 Dataset Schema Details

### Classification Datasets

Features:
- `age` (int): 18-80
- `income` (float): Log-normal distribution, ~10% missing
- `credit_score` (float): Normal distribution, ~5% missing
- `account_balance` (float): Uniform distribution
- `region` (categorical): 3 categories, ~3% missing
- `product_type` (categorical): 10 categories
- `customer_segment` (categorical): 4 categories
- `has_subscription` (binary): 0/1
- `is_active` (boolean): True/False
- `small_value` (float): 0-1 range
- `large_value` (float): 1M-10M range
- Additional features in medium/large/xlarge datasets

Target:
- `label` (int): 0/1 (imbalanced: ~70-30 or 80-20)

### Regression Datasets

Features:
- `square_feet` (float): 500-5000
- `bedrooms` (int): 1-5
- `bathrooms` (float): 1-4
- `year_built` (int): 1950-2024, ~5% missing
- `lot_size` (float): Log-normal, ~8% missing
- `neighborhood` (categorical): 4 categories
- `property_type` (categorical): 4 categories
- `condition` (categorical): 4 categories, ~4% missing
- `has_garage` (binary): 0/1
- `has_pool` (binary): 0/1
- `renovated` (boolean): True/False
- Additional features in larger datasets

Target:
- `price` (float): 50k-5M+ (includes outliers)

## 🐛 Troubleshooting

### Missing Dependencies

```bash
pip install numpy polars pandas pyarrow duckdb
```

### Database Connection Issues

Ensure database files exist:
```bash
ls -lh test_data/test_data.sqlite
ls -lh test_data/test_data.duckdb
```

Regenerate if needed:
```bash
python3 test_data/generate_test_data.py
```

### Memory Issues with Large Datasets

For xlarge datasets (50k rows), ensure sufficient RAM (4GB+). Skip xlarge tests if needed by commenting out in `stress_test.py`.

## 📝 Adding Custom Tests

To add your own test scenarios:

1. Create custom data:
```python
import polars as pl
df = pl.DataFrame({
    'feature1': [...],
    'feature2': [...],
    'target': [...]
})
df.write_csv('test_data/my_custom_data.csv')
```

2. Create a config file:
```yaml
data:
  train:
    type: file
    path: "test_data/my_custom_data.csv"
model:
  task: classification
  target: target
```

3. Run:
```bash
python3 -m dyxgb.cli train --config test_data/my_config.yaml
```

## 📄 License

This test data and scripts are part of the dyxgb project and follow the same MIT license.
