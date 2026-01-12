# Quick Start Guide - dyxgb Test Data

## 🚀 Getting Started in 3 Steps

### 1. Generate Test Data (if not already done)
```bash
cd test_data
python3 generate_test_data.py
```

### 2. Run Your First Test
```bash
# Simple numeric dataset (works out of the box)
python3 -m dyxgb.cli train \
  --source test_data/simple_train.csv \
  --target label \
  --output test_data/models/my_model.json
```

### 3. Try With Configuration
```bash
# Advanced test with transforms
python3 -m dyxgb.cli train \
  --config test_data/config_classification_advanced.yaml
```

## 📋 Common Commands

### Training

```bash
# Basic CSV training
python3 -m dyxgb.cli train \
  --source test_data/classification_train.csv \
  --target label \
  --output models/model.json \
  --encoder-output models/encoder.joblib

# Regression
python3 -m dyxgb.cli train \
  --source test_data/regression_train.csv \
  --target price \
  --task regression \
  --output models/regression_model.json

# With hyperparameter tuning
python3 -m dyxgb.cli train \
  --config test_data/config_tuning.yaml

# From database
python3 -m dyxgb.cli train \
  --source "sqlite:///test_data/test_data.sqlite" \
  --table train_data \
  --target label \
  --output models/db_model.json
```

### Prediction

```bash
python3 -m dyxgb.cli predict \
  --source test_data/classification_test.csv \
  --model models/model.json \
  --encoder models/encoder.joblib \
  --output predictions.csv
```

### Evaluation

```bash
python3 -m dyxgb.cli evaluate \
  --source test_data/classification_test.csv \
  --model models/model.json \
  --encoder models/encoder.joblib \
  --target label
```

### Feature Importance

```bash
python3 -m dyxgb.cli importance \
  --model models/model.json \
  --top 10
```

## 🧪 Run Stress Tests

```bash
cd test_data
python3 stress_test.py
```

This runs 16 comprehensive tests covering:
- Different formats (CSV, Parquet, JSON)
- Different sources (files, SQLite, DuckDB)
- Different tasks (classification, regression, multi-class)
- Different sizes (small to xlarge)
- All CLI commands
- Edge cases

## 📊 Available Datasets

| Dataset | Rows | Use Case |
|---------|------|----------|
| `simple_train.csv` | 160 | Quick numeric-only test ✅ Start here! |
| `classification_small` | 100 | Fast iteration |
| `classification_medium` | 1k | Standard testing |
| `classification_large` | 10k | Performance test |
| `classification_xlarge` | 50k | Stress test |
| `regression_medium` | 1k | Regression task |
| `multiclass_classification` | 2k | 5-class problem |
| `edge_cases` | 500 | Extreme values |

All datasets available in CSV, Parquet, and JSON formats!

## ⚙️ Available Configs

| Config | Description |
|--------|-------------|
| `config_classification_basic.yaml` | Simple classification |
| `config_classification_advanced.yaml` | With all transforms |
| `config_regression.yaml` | Regression task |
| `config_tuning.yaml` | Hyperparameter optimization |
| `config_database.yaml` | Database source |
| `config_multiclass.yaml` | Multi-class classification |

## 💡 Tips

1. **Start simple**: Use `simple_train.csv` for initial testing
2. **Use configs for categorical data**: They include proper encoding
3. **Check the README**: `test_data/README.md` has detailed examples
4. **Run stress tests**: Validates all dyxgb features at once
5. **Different formats**: Test with CSV, Parquet, or JSON
6. **Database testing**: Use SQLite/DuckDB configs

## 🔍 Edge Cases Covered

✅ Missing values (5-35%)
✅ Imbalanced classes
✅ Outliers & extreme values
✅ Categorical features
✅ High cardinality
✅ Correlated features
✅ Different scales
✅ Sparse data

## 📚 More Information

- **Full Documentation**: `test_data/README.md`
- **Summary**: `test_data/SUMMARY.md`
- **Generate Script**: `test_data/generate_test_data.py`
- **Stress Test**: `test_data/stress_test.py`

## 🆘 Troubleshooting

**Error: "DataFrame.dtypes for data must be int, float, bool or category"**
→ Use a config file with `encode` transform for categorical columns
→ Or use `simple_train.csv` which is numeric-only

**Database not found**
→ Run `python3 test_data/generate_test_data.py` first

**Out of memory with xlarge dataset**
→ Use smaller datasets or increase available RAM

## ✨ Example Workflow

```bash
# 1. Generate data
cd /home/runner/work/dyxgb/dyxgb
python3 test_data/generate_test_data.py

# 2. Train a model
python3 -m dyxgb.cli train \
  --config test_data/config_classification_advanced.yaml

# 3. Make predictions  
python3 -m dyxgb.cli predict \
  --source test_data/classification_test.parquet \
  --model test_data/models/classification_advanced_model.json \
  --encoder test_data/models/classification_advanced_encoder.joblib \
  --pipeline test_data/models/classification_advanced_pipeline.joblib \
  --output predictions.csv

# 4. Evaluate
python3 -m dyxgb.cli evaluate \
  --source test_data/classification_test.parquet \
  --model test_data/models/classification_advanced_model.json \
  --encoder test_data/models/classification_advanced_encoder.joblib \
  --target label

# 5. Check importance
python3 -m dyxgb.cli importance \
  --model test_data/models/classification_advanced_model.json
```

---

**Ready to stress test dyxgb? Start with `simple_train.csv` and work your way up!** 🚀
