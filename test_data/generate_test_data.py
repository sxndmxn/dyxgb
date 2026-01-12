#!/usr/bin/env python3
"""
Generate comprehensive test data for dyxgb with all edge cases using Faker.

This script creates datasets covering:
- Classification and regression tasks
- Multiple file formats (CSV, Parquet, JSON)
- Edge cases: missing values, imbalanced classes, outliers, categorical features
- Different dataset sizes (small, medium, large)
- Database sources (SQLite, DuckDB)
- Realistic fake data using Faker library for robust testing
"""

import numpy as np
import polars as pl
from pathlib import Path
import duckdb
import sqlite3
from faker import Faker

# Set random seed for reproducibility
np.random.seed(42)
fake = Faker()
Faker.seed(42)


def generate_classification_data(n_rows: int, complexity: str = "standard") -> pl.DataFrame:
    """
    Generate classification dataset with edge cases using Faker for realistic data.
    
    Args:
        n_rows: Number of rows to generate
        complexity: 'simple', 'standard', or 'complex'
    
    Returns:
        Polars DataFrame with classification data
    """
    
    # Use Faker to generate realistic data
    data = {
        # Realistic demographic data from Faker
        'age': [fake.random_int(min=18, max=95) for _ in range(n_rows)],
        'name': [fake.name() for _ in range(n_rows)],
        'email': [fake.email() for _ in range(n_rows)],
        'phone': [fake.phone_number() for _ in range(n_rows)],
        'job_title': [fake.job() for _ in range(n_rows)],
        'company': [fake.company() for _ in range(n_rows)],
        
        # Realistic financial data with edge cases
        'income': [max(0, fake.random_int(min=-10000, max=500000)) for _ in range(n_rows)],  # Some negative (edge case)
        'credit_score': [fake.random_int(min=300, max=850) for _ in range(n_rows)],
        'account_balance': [fake.random_int(min=-50000, max=1000000) for _ in range(n_rows)],
        
        # Geographic data
        'city': [fake.city() for _ in range(n_rows)],
        'state': [fake.state() for _ in range(n_rows)],
        'country': [fake.country() for _ in range(n_rows)],
        'zipcode': [fake.zipcode() for _ in range(n_rows)],
        
        # Internet/tech data
        'ip_address': [fake.ipv4() for _ in range(n_rows)],
        'user_agent': [fake.user_agent() for _ in range(n_rows)],
        
        # Dates with potential edge cases
        'signup_date': [fake.date_between(start_date='-10y', end_date='today') for _ in range(n_rows)],
        'last_login': [fake.date_time_between(start_date='-1y', end_date='now') for _ in range(n_rows)],
        
        # Binary/categorical features
        'has_subscription': np.random.choice([0, 1], n_rows, p=[0.7, 0.3]),
        'is_active': np.random.choice([True, False], n_rows, p=[0.6, 0.4]),
        'subscription_tier': [fake.random_element(elements=('Free', 'Basic', 'Premium', 'Enterprise')) for _ in range(n_rows)],
    }
    
    if complexity in ['standard', 'complex']:
        # Add more realistic features for standard/complex
        data.update({
            'ssn': [fake.ssn() for _ in range(n_rows)],
            'credit_card': [fake.credit_card_number() for _ in range(n_rows)],
            'license_plate': [fake.license_plate() for _ in range(n_rows)],
            'mac_address': [fake.mac_address() for _ in range(n_rows)],
            'browser': [fake.random_element(elements=('Chrome', 'Firefox', 'Safari', 'Edge', 'Opera', 'IE', 'Unknown')) for _ in range(n_rows)],
            'device_type': [fake.random_element(elements=('Desktop', 'Mobile', 'Tablet', 'IoT', 'Unknown')) for _ in range(n_rows)],
        })
    
    if complexity == 'complex':
        # Add more edge case features
        data.update({
            'latitude': [float(fake.latitude()) for _ in range(n_rows)],
            'longitude': [float(fake.longitude()) for _ in range(n_rows)],
            'bio': [fake.text(max_nb_chars=200) for _ in range(n_rows)],  # Text data
            'website': [fake.url() for _ in range(n_rows)],
            'color_preference': [fake.color_name() for _ in range(n_rows)],
            'currency': [fake.currency_code() for _ in range(n_rows)],
        })
        
    df = pl.DataFrame(data)
    
    # Create target based on features with some realistic logic
    # Higher income, better credit score, premium subscription -> more likely to be 1
    income_normalized = (df['income'] - df['income'].mean()) / (df['income'].std() + 1e-8)
    credit_normalized = (df['credit_score'] - df['credit_score'].mean()) / (df['credit_score'].std() + 1e-8)
    
    target_score = (
        (df['age'] > 50).cast(pl.Int32) * 0.2 +
        (income_normalized > 0).cast(pl.Int32) * 0.3 +
        (credit_normalized > 0).cast(pl.Int32) * 0.2 +
        (df['has_subscription'] == 1).cast(pl.Int32) * 0.3
    )
    
    # Add noise and create binary target
    noise = pl.Series(np.random.randn(n_rows) * 0.3)
    target_prob = (target_score + noise).clip(0, 1)
    
    # Create imbalanced classes
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
    
    # Introduce missing values (edge case) - more realistic patterns
    missing_mask_income = np.random.random(n_rows) < 0.12  # 12% missing
    missing_mask_credit = np.random.random(n_rows) < 0.08  # 8% missing
    missing_mask_email = np.random.random(n_rows) < 0.05  # 5% missing
    missing_mask_phone = np.random.random(n_rows) < 0.15  # 15% missing (common)
    
    df = df.with_columns([
        pl.when(pl.Series(missing_mask_income)).then(None).otherwise(pl.col('income')).alias('income'),
        pl.when(pl.Series(missing_mask_credit)).then(None).otherwise(pl.col('credit_score')).alias('credit_score'),
        pl.when(pl.Series(missing_mask_email)).then(None).otherwise(pl.col('email')).alias('email'),
        pl.when(pl.Series(missing_mask_phone)).then(None).otherwise(pl.col('phone')).alias('phone'),
    ])
    
    # Add some extreme outliers and corrupted data (fucked up data for robustness testing)
    outlier_indices = np.random.choice(n_rows, size=max(1, n_rows // 100), replace=False)
    outlier_mask = np.zeros(n_rows, dtype=bool)
    outlier_mask[outlier_indices] = True
    
    # Introduce unrealistic values, corruptions
    df = df.with_columns([
        pl.when(pl.Series(outlier_mask))
        .then(pl.col('income') * 100)  # 100x income for outliers (corrupt data)
        .otherwise(pl.col('income'))
        .alias('income'),
        pl.when(pl.Series(outlier_mask))
        .then(pl.lit(999))  # Invalid credit score
        .otherwise(pl.col('credit_score'))
        .alias('credit_score')
    ])
    
    return df


def generate_regression_data(n_rows: int, complexity: str = "standard") -> pl.DataFrame:
    """
    Generate regression dataset with edge cases using Faker for realistic property data.
    
    Args:
        n_rows: Number of rows to generate
        complexity: 'simple', 'standard', or 'complex'
    
    Returns:
        Polars DataFrame with regression data
    """
    
    data = {
        # Use Faker for realistic property addresses
        'address': [fake.street_address() for _ in range(n_rows)],
        'city': [fake.city() for _ in range(n_rows)],
        'state': [fake.state() for _ in range(n_rows)],
        'zipcode': [fake.zipcode() for _ in range(n_rows)],
        
        # Property features with realistic ranges
        'square_feet': [fake.random_int(min=400, max=8000) for _ in range(n_rows)],
        'bedrooms': [fake.random_int(min=1, max=7) for _ in range(n_rows)],
        'bathrooms': [fake.random_int(min=1, max=6) for _ in range(n_rows)],
        'year_built': [fake.random_int(min=1880, max=2024) for _ in range(n_rows)],
        'lot_size': [max(0, fake.random_int(min=-1000, max=50000)) for _ in range(n_rows)],  # Some negative (edge case)
        
        # Realistic categorical data
        'neighborhood': [fake.random_element(elements=('Downtown', 'Suburbs', 'Rural', 'Urban', 'Waterfront', 'Historic')) for _ in range(n_rows)],
        'property_type': [fake.random_element(elements=('House', 'Condo', 'Townhouse', 'Apartment', 'Villa', 'Duplex')) for _ in range(n_rows)],
        'condition': [fake.random_element(elements=('Excellent', 'Good', 'Fair', 'Poor', 'New Construction', 'Fixer-upper')) for _ in range(n_rows)],
        
        # Binary features
        'has_garage': np.random.choice([0, 1, 2, 3], n_rows, p=[0.2, 0.5, 0.2, 0.1]),  # Number of garage spaces
        'has_pool': np.random.choice([0, 1], n_rows, p=[0.75, 0.25]),
        'renovated': np.random.choice([True, False], n_rows, p=[0.35, 0.65]),
        'has_basement': np.random.choice([True, False], n_rows, p=[0.6, 0.4]),
        
        # Realistic listing data
        'listing_agent': [fake.name() for _ in range(n_rows)],
        'listing_date': [fake.date_between(start_date='-2y', end_date='today') for _ in range(n_rows)],
    }
    
    if complexity in ['standard', 'complex']:
        data.update({
            'distance_to_city': [fake.random_int(min=0, max=100) for _ in range(n_rows)],
            'crime_rate': [max(0, fake.pyfloat(min_value=0, max_value=15, right_digits=2)) for _ in range(n_rows)],
            'school_rating': [fake.random_int(min=1, max=10) for _ in range(n_rows)],
            'hoa_fee': [fake.random_int(min=0, max=1000) for _ in range(n_rows)],
            'property_tax': [fake.random_int(min=1000, max=30000) for _ in range(n_rows)],
        })
    
    if complexity == 'complex':
        data.update({
            'latitude': [float(fake.latitude()) for _ in range(n_rows)],
            'longitude': [float(fake.longitude()) for _ in range(n_rows)],
            'description': [fake.text(max_nb_chars=150) for _ in range(n_rows)],
            'mls_number': [fake.bothify(text='MLS-########') for _ in range(n_rows)],
        })
    
    df = pl.DataFrame(data)
    
    # Create realistic price target based on features with more realistic pricing
    base_price = (
        df['square_feet'] * 250 +  # $250 per sqft (more realistic)
        df['bedrooms'] * 30000 +   # $30k per bedroom
        df['lot_size'] * 10 +       # $10 per sqft of lot
        (2024 - df['year_built']) * -800  # Depreciation
    )
    
    # Add categorical effects using when-then logic
    neighborhood_bonus = (
        pl.when(pl.col('neighborhood') == 'Waterfront').then(pl.lit(250000))
        .when(pl.col('neighborhood') == 'Downtown').then(pl.lit(150000))
        .when(pl.col('neighborhood') == 'Historic').then(pl.lit(100000))
        .when(pl.col('neighborhood') == 'Urban').then(pl.lit(50000))
        .when(pl.col('neighborhood') == 'Suburbs').then(pl.lit(0))
        .otherwise(pl.lit(-75000))  # Rural
    )
    
    condition_bonus = (
        pl.when(pl.col('condition') == 'New Construction').then(pl.lit(100000))
        .when(pl.col('condition') == 'Excellent').then(pl.lit(75000))
        .when(pl.col('condition') == 'Good').then(pl.lit(30000))
        .when(pl.col('condition') == 'Fair').then(pl.lit(0))
        .otherwise(pl.lit(-50000))  # Poor or Fixer-upper
    )
    
    # Combine effects with realistic noise
    noise = np.random.normal(0, 75000, n_rows)
    
    df = df.with_columns(
        (base_price + neighborhood_bonus + condition_bonus + pl.Series(noise))
        .clip(25000, None)  # Some very cheap properties (edge case)
        .alias('price')
    )
    
    # Introduce missing values (more aggressive)
    missing_mask_lot = np.random.random(n_rows) < 0.12  # 12% missing
    missing_mask_year = np.random.random(n_rows) < 0.08  # 8% missing
    missing_mask_condition = np.random.random(n_rows) < 0.10  # 10% missing
    missing_mask_address = np.random.random(n_rows) < 0.05  # 5% missing
    
    df = df.with_columns([
        pl.when(pl.Series(missing_mask_lot)).then(None).otherwise(pl.col('lot_size')).alias('lot_size'),
        pl.when(pl.Series(missing_mask_year)).then(None).otherwise(pl.col('year_built')).alias('year_built'),
        pl.when(pl.Series(missing_mask_condition)).then(None).otherwise(pl.col('condition')).alias('condition'),
        pl.when(pl.Series(missing_mask_address)).then(None).otherwise(pl.col('address')).alias('address'),
    ])
    
    # Add extreme outliers and corrupt data (fucked up data for robustness)
    outlier_indices = np.random.choice(n_rows, size=max(1, n_rows // 50), replace=False)
    outlier_mask = np.zeros(n_rows, dtype=bool)
    outlier_mask[outlier_indices] = True
    
    # Corrupt some data - unrealistic prices, invalid square footage
    df = df.with_columns([
        pl.when(pl.Series(outlier_mask))
        .then(pl.col('price') * 10)  # 10x price for extreme outliers
        .otherwise(pl.col('price'))
        .alias('price'),
        pl.when(pl.Series(outlier_mask))
        .then(pl.lit(999999))  # Invalid huge square footage
        .otherwise(pl.col('square_feet'))
        .alias('square_feet')
    ])
    
    if complexity == 'complex':
        # Add polynomial features
        df = df.with_columns([
            (pl.col('square_feet') ** 2).alias('square_feet_squared'),
            (pl.col('year_built') * pl.col('square_feet')).alias('year_sqft_interaction'),
        ])
    
    return df


def generate_multiclass_classification_data(n_rows: int) -> pl.DataFrame:
    """Generate multi-class classification dataset (5 classes) using Faker."""
    
    data = {
        # Realistic customer data
        'customer_name': [fake.name() for _ in range(n_rows)],
        'email': [fake.email() for _ in range(n_rows)],
        'account_age_days': [fake.random_int(min=1, max=3650) for _ in range(n_rows)],
        'total_purchases': [fake.random_int(min=0, max=500) for _ in range(n_rows)],
        'avg_purchase_amount': [fake.pyfloat(min_value=10, max_value=5000, right_digits=2) for _ in range(n_rows)],
        'login_frequency': [fake.random_int(min=0, max=365) for _ in range(n_rows)],
        
        # Behavioral features
        'complaints': [fake.random_int(min=0, max=20) for _ in range(n_rows)],
        'support_tickets': [fake.random_int(min=0, max=50) for _ in range(n_rows)],
        'referrals': [fake.random_int(min=0, max=10) for _ in range(n_rows)],
        
        # Categorical
        'membership_tier': [fake.random_element(elements=('Bronze', 'Silver', 'Gold', 'Platinum')) for _ in range(n_rows)],
        'region': [fake.random_element(elements=('North', 'South', 'East', 'West', 'Central')) for _ in range(n_rows)],
        'device_type': [fake.random_element(elements=('Mobile', 'Desktop', 'Tablet')) for _ in range(n_rows)],
        
        # Binary
        'has_app': np.random.choice([0, 1], n_rows, p=[0.4, 0.6]),
        'opted_in_marketing': np.random.choice([0, 1], n_rows, p=[0.6, 0.4]),
    }
    
    df = pl.DataFrame(data)
    
    # Create 5-class target with imbalance (customer segments)
    # 0: Churned, 1: At-risk, 2: Regular, 3: Engaged, 4: VIP
    # Class distribution: 40%, 25%, 20%, 10%, 5%
    class_probs = [0.40, 0.25, 0.20, 0.10, 0.05]
    target = np.random.choice(range(5), size=n_rows, p=class_probs)
    
    # Add realistic logic to make it learnable
    target = np.where(
        (df['total_purchases'].to_numpy() > 200) & (df['avg_purchase_amount'].to_numpy() > 1000),
        4,  # VIP for high-value customers
        target
    )
    target = np.where(
        (df['total_purchases'].to_numpy() > 100) & (df['login_frequency'].to_numpy() > 100),
        3,  # Engaged for active customers
        target
    )
    target = np.where(
        (df['account_age_days'].to_numpy() < 30) & (df['total_purchases'].to_numpy() == 0),
        0,  # Churned for inactive new customers
        target
    )
    
    df = df.with_columns(pl.Series('label', target))
    
    # Add missing values (realistic patterns)
    missing_mask_email = np.random.random(n_rows) < 0.10
    missing_mask_purchase = np.random.random(n_rows) < 0.15
    df = df.with_columns([
        pl.when(pl.Series(missing_mask_email)).then(None).otherwise(pl.col('email')).alias('email'),
        pl.when(pl.Series(missing_mask_purchase)).then(None).otherwise(pl.col('avg_purchase_amount')).alias('avg_purchase_amount'),
    ])
    
    return df


def generate_edge_case_dataset(n_rows: int = 100) -> pl.DataFrame:
    """
    Generate dataset with EXTREME edge cases using Faker for realistic messy data.
    This creates "fucked up" data to stress test robustness:
    - All data types
    - High percentage of missing values  
    - Extreme outliers and corruptions
    - Zero variance features
    - Highly correlated features
    - Near-constant categorical features
    - Invalid/malformed data
    """
    
    data = {
        # Realistic but corrupted personal data
        'name': [fake.name() if np.random.random() > 0.2 else fake.bothify(text='###???') for _ in range(n_rows)],  # Some gibberish
        'email': [fake.email() if np.random.random() > 0.25 else fake.word() + '@corrupted' for _ in range(n_rows)],  # Some invalid
        'ssn': [fake.ssn() if np.random.random() > 0.3 else '000-00-0000' for _ in range(n_rows)],  # Some fake SSNs
        
        # Numeric with extreme ranges and corruption
        'tiny_values': [fake.pyfloat(min_value=1e-12, max_value=1e-8, right_digits=10) for _ in range(n_rows)],
        'huge_values': [fake.pyfloat(min_value=1e8, max_value=1e11, right_digits=2) for _ in range(n_rows)],
        'negative_mess': [fake.random_int(min=-1000000, max=1000000) for _ in range(n_rows)],
        
        # Mixed scale chaos
        'mixed_scale': np.concatenate([
            np.random.uniform(0, 0.001, n_rows // 3),
            np.random.uniform(1000, 100000, n_rows // 3),
            np.random.uniform(1e-8, 1e8, n_rows - 2*(n_rows // 3))
        ]),
        
        # Zero/near-zero variance (constant)
        'constant_feature': np.ones(n_rows) * 42,
        'near_constant': np.concatenate([
            np.ones(n_rows - 3) * 100,
            np.array([101, 99, 100.001])
        ]),
        
        # Highly correlated features
        'corr_base': np.random.randn(n_rows),
    }
    
    data['corr_feature_1'] = data['corr_base'] + np.random.randn(n_rows) * 0.0001  # Nearly perfect correlation
    data['corr_feature_2'] = data['corr_base'] * 2 + np.random.randn(n_rows) * 0.0001
    data['corr_feature_3'] = data['corr_base'] * -1 + np.random.randn(n_rows) * 0.0001  # Perfect negative correlation
    
    # Categorical with extreme distributions and corruptions
    categories_normal = [fake.word() for _ in range(10)]
    data['rare_category'] = np.concatenate([
        np.array(['Common'] * (n_rows - 5)),
        np.array(['Rare1', 'Rare2', 'Rare3', 'Rare4', 'Rare5'])
    ])
    
    # Random strings that look like categories but are unique (high cardinality nightmare)
    data['chaotic_category'] = [fake.bothify(text='??-###-???') if np.random.random() > 0.5 else fake.random_element(elements=categories_normal) for _ in range(n_rows)]
    
    # Sparse binary (almost all zeros)
    data['binary_sparse'] = np.random.choice([0, 1], n_rows, p=[0.98, 0.02])
    
    # Dates with corruptions
    data['date_field'] = [fake.date_between(start_date='-50y', end_date='today') if np.random.random() > 0.15 else None for _ in range(n_rows)]
    
    # URLs and IPs (some malformed)
    data['url'] = [fake.url() if np.random.random() > 0.2 else 'http://corrupted' for _ in range(n_rows)]
    data['ip'] = [fake.ipv4() if np.random.random() > 0.2 else '999.999.999.999' for _ in range(n_rows)]
    
    df = pl.DataFrame(data)
    
    # Create target
    df = df.with_columns(
        (pl.col('corr_base') > 0).cast(pl.Int32).alias('label')
    )
    
    # Add LOTS of missing values (40-50% for some columns)
    for col in ['tiny_values', 'huge_values', 'mixed_scale', 'near_constant', 'negative_mess']:
        missing_mask = np.random.random(n_rows) < 0.45  # 45% missing
        df = df.with_columns([
            pl.when(pl.Series(missing_mask)).then(None).otherwise(pl.col(col)).alias(col)
        ])
    
    # Add extreme values: NaN, Inf, -Inf
    special_mask_inf = np.random.random(n_rows) < 0.08
    special_mask_neginf = np.random.random(n_rows) < 0.05
    
    df = df.with_columns([
        pl.when(pl.Series(special_mask_inf))
        .then(pl.lit(float('inf')))
        .when(pl.Series(special_mask_neginf))
        .then(pl.lit(float('-inf')))
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
    
    duckdb_conn.execute("DROP TABLE IF EXISTS train_data")
    duckdb_conn.execute("DROP TABLE IF EXISTS test_data")
    duckdb_conn.execute("CREATE TABLE train_data AS SELECT * FROM df_train")
    duckdb_conn.execute("CREATE TABLE test_data AS SELECT * FROM df_test")
    
    duckdb_conn.close()
    print(f"✓ Created DuckDB database: {duckdb_path}")


def main():
    """Generate all test datasets using Faker for realistic and robust test data."""
    
    base_path = Path(__file__).parent
    print(f"\n{'='*70}")
    print("Generating comprehensive test data for dyxgb using Faker")
    print("Creating realistic, messy, and 'fucked up' data for robustness testing")
    print(f"{'='*70}\n")
    
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
    print(f"\n{'='*70}")
    print("✅ Test data generation complete using Faker!")
    print(f"{'='*70}")
    print("\nDatasets created with REALISTIC DATA:")
    print("  • Classification: small (100), medium (1k), large (10k), xlarge (50k)")
    print("    - Names, emails, jobs, addresses, financial data")
    print("    - Phone numbers, IPs, user agents, dates")
    print("  • Regression: small (100), medium (1k), large (10k)")
    print("    - Property addresses, cities, states, zipcodes")
    print("    - Realistic pricing, neighborhoods, MLS numbers")
    print("  • Multi-class: 5 customer segments (2k rows)")
    print("    - Customer names, emails, behavioral data")
    print("  • Edge cases: EXTREME/CORRUPTED data (500)")
    print("    - Invalid emails, malformed SSNs, corrupted data")
    print("    - Inf/-Inf values, extreme outliers, gibberish")
    print("  • Train/test splits: classification, regression")
    print("  • Database sources: SQLite, DuckDB")
    print("\nFormats: CSV, Parquet, JSON (for each dataset)")
    print("\nEdge cases covered (ROBUST TESTING):")
    print("  ✓ Missing values (8-50% depending on dataset)")
    print("  ✓ Imbalanced classes (70-30, 80-20, 40-25-20-10-5)")
    print("  ✓ Outliers and extreme values (100x corruptions)")
    print("  ✓ REALISTIC data types (names, emails, SSNs, addresses, etc.)")
    print("  ✓ High cardinality categorical features")
    print("  ✓ Highly correlated features (>0.999)")
    print("  ✓ Different scales (1e-12 to 1e12)")
    print("  ✓ Zero/near-zero variance features")
    print("  ✓ Invalid/malformed data (corrupted emails, IPs, SSNs)")
    print("  ✓ Inf/-Inf values, NaN values")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
