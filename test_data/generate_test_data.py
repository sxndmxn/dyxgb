#!/usr/bin/env python3
"""
Generate comprehensive test data for dyxgb with all edge cases.

This script creates datasets covering:
- Classification and regression tasks
- Multiple file formats (CSV, Parquet, JSON)
- Edge cases: missing values, imbalanced classes, outliers, categorical features
- Different dataset sizes (small, medium, large)
- Database sources (SQLite, DuckDB)
"""

import numpy as np
import polars as pl
from pathlib import Path
import duckdb
import sqlite3

# Set random seed for reproducibility
np.random.seed(42)


def generate_classification_data(n_rows: int, complexity: str = "standard") -> pl.DataFrame:
    """
    Generate classification dataset with edge cases.
    
    Args:
        n_rows: Number of rows to generate
        complexity: 'simple', 'standard', or 'complex'
    
    Returns:
        Polars DataFrame with classification data
    """
    
    # Categorical features with varying cardinality
    categories_low = ['A', 'B', 'C']
    categories_medium = ['cat_' + str(i) for i in range(10)]
    categories_high = ['item_' + str(i) for i in range(50)]
    
    data = {
        # Numeric features with different distributions
        'age': np.random.randint(18, 80, n_rows),
        'income': np.random.lognormal(10, 1, n_rows),  # Log-normal distribution
        'credit_score': np.random.normal(650, 100, n_rows),
        'account_balance': np.random.uniform(-5000, 50000, n_rows),
        
        # Categorical features
        'region': np.random.choice(categories_low, n_rows),
        'product_type': np.random.choice(categories_medium, n_rows),
        'customer_segment': np.random.choice(['Premium', 'Standard', 'Basic', 'New'], n_rows),
        
        # Binary features
        'has_subscription': np.random.choice([0, 1], n_rows),
        'is_active': np.random.choice([True, False], n_rows),
        
        # Features with different scales
        'small_value': np.random.uniform(0, 1, n_rows),
        'large_value': np.random.uniform(1000000, 10000000, n_rows),
    }
    
    if complexity in ['standard', 'complex']:
        # Add more features for standard/complex
        data.update({
            'feature_1': np.random.randn(n_rows),
            'feature_2': np.random.randn(n_rows) * 10,
            'feature_3': np.random.exponential(2, n_rows),
            'high_cardinality_cat': np.random.choice(categories_high, n_rows),
        })
    
    if complexity == 'complex':
        # Add correlated features
        data['feature_4'] = data['age'] * 0.5 + np.random.randn(n_rows) * 5
        data['feature_5'] = data['income'] * 0.0001 + np.random.randn(n_rows)
        
    df = pl.DataFrame(data)
    
    # Create target based on features (with some noise)
    # This creates a realistic classification scenario
    target_score = (
        (df['age'] > 40).cast(pl.Int32) * 0.3 +
        (df['income'] > df['income'].median()).cast(pl.Int32) * 0.3 +
        (df['credit_score'] > 650).cast(pl.Int32) * 0.2 +
        (df['has_subscription'] == 1).cast(pl.Int32) * 0.2
    )
    
    # Add noise and create binary target
    noise = pl.Series(np.random.randn(n_rows) * 0.2)
    target_prob = (target_score + noise).clip(0, 1)
    
    # Create imbalanced classes (70-30 split by default, more extreme for complex)
    if complexity == 'complex':
        threshold = 0.8  # Very imbalanced: ~80-20
    else:
        threshold = 0.6  # Moderately imbalanced: ~70-30
        
    df = df.with_columns(
        pl.when(target_prob > threshold)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias('label')
    )
    
    # Introduce missing values (edge case)
    missing_mask_income = np.random.random(n_rows) < 0.1  # 10% missing
    missing_mask_credit = np.random.random(n_rows) < 0.05  # 5% missing
    missing_mask_region = np.random.random(n_rows) < 0.03  # 3% missing
    
    df = df.with_columns([
        pl.when(pl.Series(missing_mask_income)).then(None).otherwise(pl.col('income')).alias('income'),
        pl.when(pl.Series(missing_mask_credit)).then(None).otherwise(pl.col('credit_score')).alias('credit_score'),
        pl.when(pl.Series(missing_mask_region)).then(None).otherwise(pl.col('region')).alias('region'),
    ])
    
    # Add some extreme outliers
    outlier_indices = np.random.choice(n_rows, size=max(1, n_rows // 100), replace=False)
    outlier_mask = np.zeros(n_rows, dtype=bool)
    outlier_mask[outlier_indices] = True
    
    df = df.with_columns([
        pl.when(pl.Series(outlier_mask))
        .then(pl.col('income') * 10)  # 10x income for outliers
        .otherwise(pl.col('income'))
        .alias('income')
    ])
    
    return df


def generate_regression_data(n_rows: int, complexity: str = "standard") -> pl.DataFrame:
    """
    Generate regression dataset with edge cases.
    
    Args:
        n_rows: Number of rows to generate
        complexity: 'simple', 'standard', or 'complex'
    
    Returns:
        Polars DataFrame with regression data
    """
    
    data = {
        # Numeric predictors
        'square_feet': np.random.uniform(500, 5000, n_rows),
        'bedrooms': np.random.randint(1, 6, n_rows),
        'bathrooms': np.random.uniform(1, 4, n_rows),
        'year_built': np.random.randint(1950, 2024, n_rows),
        'lot_size': np.random.lognormal(8, 0.5, n_rows),
        
        # Categorical predictors
        'neighborhood': np.random.choice(['Downtown', 'Suburbs', 'Rural', 'Urban'], n_rows),
        'property_type': np.random.choice(['House', 'Condo', 'Townhouse', 'Apartment'], n_rows),
        'condition': np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], n_rows),
        
        # Binary features
        'has_garage': np.random.choice([0, 1], n_rows, p=[0.3, 0.7]),
        'has_pool': np.random.choice([0, 1], n_rows, p=[0.8, 0.2]),
        'renovated': np.random.choice([True, False], n_rows, p=[0.4, 0.6]),
    }
    
    if complexity in ['standard', 'complex']:
        data.update({
            'distance_to_city': np.random.uniform(0, 50, n_rows),
            'crime_rate': np.random.gamma(2, 2, n_rows),
            'school_rating': np.random.uniform(1, 10, n_rows),
        })
    
    df = pl.DataFrame(data)
    
    # Create realistic price target based on features
    base_price = (
        df['square_feet'] * 200 +  # $200 per sqft
        df['bedrooms'] * 20000 +   # $20k per bedroom
        df['lot_size'] * 50 +       # $50 per sqft of lot
        (2024 - df['year_built']) * -500  # Depreciation
    )
    
    # Add categorical effects using when-then logic
    neighborhood_bonus = (
        pl.when(pl.col('neighborhood') == 'Downtown').then(pl.lit(100000))
        .when(pl.col('neighborhood') == 'Urban').then(pl.lit(50000))
        .when(pl.col('neighborhood') == 'Suburbs').then(pl.lit(0))
        .otherwise(pl.lit(-50000))  # Rural
    )
    
    condition_bonus = (
        pl.when(pl.col('condition') == 'Excellent').then(pl.lit(50000))
        .when(pl.col('condition') == 'Good').then(pl.lit(20000))
        .when(pl.col('condition') == 'Fair').then(pl.lit(0))
        .otherwise(pl.lit(-30000))  # Poor
    )
    
    # Combine effects with noise
    noise = np.random.normal(0, 50000, n_rows)
    
    df = df.with_columns(
        (base_price + neighborhood_bonus + condition_bonus + pl.Series(noise))
        .clip(50000, None)
        .alias('price')
    )
    
    # Introduce missing values
    missing_mask_lot = np.random.random(n_rows) < 0.08
    missing_mask_year = np.random.random(n_rows) < 0.05
    missing_mask_condition = np.random.random(n_rows) < 0.04
    
    df = df.with_columns([
        pl.when(pl.Series(missing_mask_lot)).then(None).otherwise(pl.col('lot_size')).alias('lot_size'),
        pl.when(pl.Series(missing_mask_year)).then(None).otherwise(pl.col('year_built')).alias('year_built'),
        pl.when(pl.Series(missing_mask_condition)).then(None).otherwise(pl.col('condition')).alias('condition'),
    ])
    
    # Add extreme outliers (luxury properties)
    outlier_indices = np.random.choice(n_rows, size=max(1, n_rows // 50), replace=False)
    outlier_mask = np.zeros(n_rows, dtype=bool)
    outlier_mask[outlier_indices] = True
    
    df = df.with_columns([
        pl.when(pl.Series(outlier_mask))
        .then(pl.col('price') * 5)  # Luxury properties 5x price
        .otherwise(pl.col('price'))
        .alias('price')
    ])
    
    if complexity == 'complex':
        # Add polynomial features
        df = df.with_columns([
            (pl.col('square_feet') ** 2).alias('square_feet_squared'),
            (pl.col('year_built') * pl.col('square_feet')).alias('year_sqft_interaction'),
        ])
    
    return df


def generate_multiclass_classification_data(n_rows: int) -> pl.DataFrame:
    """Generate multi-class classification dataset (5 classes)."""
    
    data = {
        'feature_1': np.random.randn(n_rows),
        'feature_2': np.random.randn(n_rows),
        'feature_3': np.random.exponential(1, n_rows),
        'feature_4': np.random.uniform(-10, 10, n_rows),
        'category_a': np.random.choice(['X', 'Y', 'Z'], n_rows),
        'category_b': np.random.choice(['Group1', 'Group2', 'Group3', 'Group4'], n_rows),
        'binary_flag': np.random.choice([0, 1], n_rows),
    }
    
    df = pl.DataFrame(data)
    
    # Create 5-class target with imbalance
    # Class distribution: 40%, 25%, 20%, 10%, 5%
    class_probs = [0.40, 0.25, 0.20, 0.10, 0.05]
    target = np.random.choice(range(5), size=n_rows, p=class_probs)
    
    # Add some logic to make it learnable
    target = np.where(
        (df['feature_1'].to_numpy() > 1) & (df['feature_2'].to_numpy() > 0),
        0,  # Class 0 for high feature values
        target
    )
    target = np.where(
        (df['feature_3'].to_numpy() > 2) & (df['binary_flag'].to_numpy() == 1),
        1,  # Class 1 for specific conditions
        target
    )
    
    df = df.with_columns(pl.Series('label', target))
    
    # Add missing values
    missing_mask = np.random.random(n_rows) < 0.07
    df = df.with_columns([
        pl.when(pl.Series(missing_mask)).then(None).otherwise(pl.col('feature_3')).alias('feature_3')
    ])
    
    return df


def generate_edge_case_dataset(n_rows: int = 100) -> pl.DataFrame:
    """
    Generate dataset with extreme edge cases:
    - All data types
    - High percentage of missing values
    - Extreme outliers
    - Zero variance features
    - Highly correlated features
    - Near-constant categorical features
    """
    
    data = {
        # Numeric with extreme ranges
        'tiny_values': np.random.uniform(1e-10, 1e-9, n_rows),
        'huge_values': np.random.uniform(1e9, 1e10, n_rows),
        'mixed_scale': np.concatenate([
            np.random.uniform(0, 1, n_rows // 2),
            np.random.uniform(1000, 10000, n_rows // 2)
        ]),
        
        # Zero/near-zero variance
        'constant_feature': np.ones(n_rows) * 42,
        'near_constant': np.concatenate([
            np.ones(n_rows - 2) * 100,
            np.array([101, 99])
        ]),
        
        # Highly correlated features
        'corr_base': np.random.randn(n_rows),
    }
    
    data['corr_feature_1'] = data['corr_base'] + np.random.randn(n_rows) * 0.01
    data['corr_feature_2'] = data['corr_base'] * 2 + np.random.randn(n_rows) * 0.01
    
    # Categorical with extreme distributions
    data['rare_category'] = np.concatenate([
        np.array(['Common'] * (n_rows - 5)),
        np.array(['Rare1', 'Rare2', 'Rare3', 'Rare4', 'Rare5'])
    ])
    
    data['binary_sparse'] = np.random.choice([0, 1], n_rows, p=[0.99, 0.01])
    
    df = pl.DataFrame(data)
    
    # Create target
    df = df.with_columns(
        (pl.col('corr_base') > 0).cast(pl.Int32).alias('label')
    )
    
    # Add LOTS of missing values (30-40%)
    for col in ['tiny_values', 'huge_values', 'mixed_scale', 'near_constant']:
        missing_mask = np.random.random(n_rows) < 0.35
        df = df.with_columns([
            pl.when(pl.Series(missing_mask)).then(None).otherwise(pl.col(col)).alias(col)
        ])
    
    # Add some NaN and Inf values (edge case)
    special_mask = np.random.random(n_rows) < 0.05
    df = df.with_columns([
        pl.when(pl.Series(special_mask[:n_rows]))
        .then(pl.lit(float('inf')))
        .otherwise(pl.col('huge_values'))
        .alias('huge_values')
    ])
    
    return df


def save_dataset_multiple_formats(df: pl.DataFrame, base_path: Path, name: str) -> None:
    """Save dataset in CSV, Parquet, and JSON formats."""
    
    # CSV
    csv_path = base_path / f"{name}.csv"
    df.write_csv(csv_path)
    print(f"✓ Created {csv_path}")
    
    # Parquet
    parquet_path = base_path / f"{name}.parquet"
    df.write_parquet(parquet_path)
    print(f"✓ Created {parquet_path}")
    
    # JSON (newline-delimited)
    json_path = base_path / f"{name}.json"
    df.write_ndjson(json_path)
    print(f"✓ Created {json_path}")


def create_database_sources(df_train: pl.DataFrame, df_test: pl.DataFrame, base_path: Path) -> None:
    """Create SQLite and DuckDB database sources."""
    
    # SQLite
    sqlite_path = base_path / "test_data.sqlite"
    sqlite_conn = sqlite3.connect(sqlite_path)
    
    df_train.to_pandas().to_sql('train_data', sqlite_conn, if_exists='replace', index=False)
    df_test.to_pandas().to_sql('test_data', sqlite_conn, if_exists='replace', index=False)
    
    sqlite_conn.close()
    print(f"✓ Created SQLite database: {sqlite_path}")
    
    # DuckDB
    duckdb_path = base_path / "test_data.duckdb"
    duckdb_conn = duckdb.connect(str(duckdb_path))
    
    duckdb_conn.execute("CREATE TABLE train_data AS SELECT * FROM df_train")
    duckdb_conn.execute("CREATE TABLE test_data AS SELECT * FROM df_test")
    
    duckdb_conn.close()
    print(f"✓ Created DuckDB database: {duckdb_path}")


def main():
    """Generate all test datasets."""
    
    base_path = Path(__file__).parent
    print(f"\n{'='*60}")
    print("Generating comprehensive test data for dyxgb")
    print(f"{'='*60}\n")
    
    # 1. Classification datasets - different sizes
    print("📊 Generating classification datasets...")
    
    # Small dataset (100 rows) - for quick testing
    df_class_small = generate_classification_data(100, complexity='simple')
    save_dataset_multiple_formats(df_class_small, base_path, "classification_small")
    
    # Medium dataset (1000 rows) - standard testing
    df_class_medium = generate_classification_data(1000, complexity='standard')
    save_dataset_multiple_formats(df_class_medium, base_path, "classification_medium")
    
    # Large dataset (10000 rows) - stress testing
    df_class_large = generate_classification_data(10000, complexity='complex')
    save_dataset_multiple_formats(df_class_large, base_path, "classification_large")
    
    # Very large dataset (50000 rows) - heavy stress testing
    print("\n📊 Generating very large classification dataset (this may take a moment)...")
    df_class_xlarge = generate_classification_data(50000, complexity='complex')
    save_dataset_multiple_formats(df_class_xlarge, base_path, "classification_xlarge")
    
    # 2. Regression datasets
    print("\n📊 Generating regression datasets...")
    
    df_reg_small = generate_regression_data(100, complexity='simple')
    save_dataset_multiple_formats(df_reg_small, base_path, "regression_small")
    
    df_reg_medium = generate_regression_data(1000, complexity='standard')
    save_dataset_multiple_formats(df_reg_medium, base_path, "regression_medium")
    
    df_reg_large = generate_regression_data(10000, complexity='complex')
    save_dataset_multiple_formats(df_reg_large, base_path, "regression_large")
    
    # 3. Multi-class classification
    print("\n📊 Generating multi-class classification dataset...")
    df_multiclass = generate_multiclass_classification_data(2000)
    save_dataset_multiple_formats(df_multiclass, base_path, "multiclass_classification")
    
    # 4. Edge case dataset
    print("\n📊 Generating edge case dataset...")
    df_edge = generate_edge_case_dataset(500)
    save_dataset_multiple_formats(df_edge, base_path, "edge_cases")
    
    # 5. Train/test splits for evaluation
    print("\n📊 Creating train/test splits...")
    
    # Classification train/test
    n_train = int(len(df_class_medium) * 0.8)
    df_class_train = df_class_medium[:n_train]
    df_class_test = df_class_medium[n_train:]
    
    save_dataset_multiple_formats(df_class_train, base_path, "classification_train")
    save_dataset_multiple_formats(df_class_test, base_path, "classification_test")
    
    # Regression train/test
    n_train_reg = int(len(df_reg_medium) * 0.8)
    df_reg_train = df_reg_medium[:n_train_reg]
    df_reg_test = df_reg_medium[n_train_reg:]
    
    save_dataset_multiple_formats(df_reg_train, base_path, "regression_train")
    save_dataset_multiple_formats(df_reg_test, base_path, "regression_test")
    
    # 6. Database sources
    print("\n📊 Creating database sources...")
    create_database_sources(df_class_train, df_class_test, base_path)
    
    # 7. Summary
    print(f"\n{'='*60}")
    print("✅ Test data generation complete!")
    print(f"{'='*60}")
    print("\nDatasets created:")
    print("  • Classification: small (100), medium (1k), large (10k), xlarge (50k)")
    print("  • Regression: small (100), medium (1k), large (10k)")
    print("  • Multi-class: 5 classes (2k rows)")
    print("  • Edge cases: extreme values, missing data (500)")
    print("  • Train/test splits: classification, regression")
    print("  • Database sources: SQLite, DuckDB")
    print("\nFormats: CSV, Parquet, JSON (for each dataset)")
    print("\nEdge cases covered:")
    print("  ✓ Missing values (5-35% depending on dataset)")
    print("  ✓ Imbalanced classes")
    print("  ✓ Outliers and extreme values")
    print("  ✓ Multiple data types (numeric, categorical, boolean)")
    print("  ✓ High cardinality categorical features")
    print("  ✓ Correlated features")
    print("  ✓ Different scales and distributions")
    print("  ✓ Zero/near-zero variance features")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
