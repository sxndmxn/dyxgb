"""Tests for dyxgb.transforms.registry - function registry for feature engineering."""

from datetime import date

import polars as pl
import pytest

from dyxgb.transforms.registry import (
    FUNCTION_REGISTRY,
    FunctionSpec,
    build_expression,
    get_categories,
    get_function,
    list_functions,
)


class TestFunctionRegistry:
    """Test the function registry structure."""

    def test_registry_not_empty(self):
        """Registry should have functions registered."""
        assert len(FUNCTION_REGISTRY) > 0

    def test_all_functions_have_required_fields(self):
        """Each function spec should have all required fields."""
        for name, spec in FUNCTION_REGISTRY.items():
            assert isinstance(spec, FunctionSpec)
            assert spec.name == name
            assert spec.description
            assert spec.category in ("math", "string", "date", "null")
            assert callable(spec.builder)
            assert isinstance(spec.params, list)
            assert spec.example

    def test_get_function_valid(self):
        """get_function should return spec for valid names."""
        spec = get_function("log")
        assert spec.name == "log"
        assert spec.category == "math"

    def test_get_function_invalid(self):
        """get_function should raise ValueError for invalid names."""
        with pytest.raises(ValueError, match="Unknown function 'nonexistent'"):
            get_function("nonexistent")

    def test_list_functions_all(self):
        """list_functions without filter returns all."""
        funcs = list_functions()
        assert len(funcs) == len(FUNCTION_REGISTRY)

    def test_list_functions_by_category(self):
        """list_functions with category filter works."""
        math_funcs = list_functions("math")
        assert all(f.category == "math" for f in math_funcs)
        assert len(math_funcs) > 0

    def test_get_categories(self):
        """get_categories returns all unique categories."""
        categories = get_categories()
        assert "math" in categories
        assert "string" in categories
        assert "date" in categories
        assert "null" in categories


class TestBuildExpression:
    """Test build_expression function."""

    def test_missing_function_key(self):
        """Should raise when function key missing."""
        with pytest.raises(ValueError, match="missing 'function' key"):
            build_expression({"name": "test", "column": "col"})

    def test_missing_column(self):
        """Should raise when column/columns missing."""
        with pytest.raises(ValueError, match="requires 'column' or 'columns'"):
            build_expression({"name": "test", "function": "log"})

    def test_unknown_function(self):
        """Should raise for unknown function."""
        with pytest.raises(ValueError, match="Unknown function"):
            build_expression({"name": "test", "function": "unknown", "column": "x"})


class TestMathFunctions:
    """Test math category functions."""

    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        """Sample dataframe for testing."""
        return pl.DataFrame({
            "amount": [1.0, 10.0, 100.0, 1000.0],
            "quantity": [2.0, 5.0, 10.0, 20.0],
            "negative": [-5.0, -2.0, 0.0, 3.0],
            "age": [18, 35, 55, 70],
        })

    def test_log(self, sample_df: pl.DataFrame):
        """log function applies log1p."""
        expr = build_expression({"function": "log", "column": "amount"})
        result = sample_df.select(expr.alias("result"))["result"]
        # log1p(1) = ln(2) ≈ 0.693
        assert result[0] == pytest.approx(0.693, rel=0.01)

    def test_square(self, sample_df: pl.DataFrame):
        """square function squares values."""
        expr = build_expression({"function": "square", "column": "quantity"})
        result = sample_df.select(expr.alias("result"))["result"]
        assert result.to_list() == [4.0, 25.0, 100.0, 400.0]

    def test_sqrt(self, sample_df: pl.DataFrame):
        """sqrt function takes square root."""
        expr = build_expression({"function": "sqrt", "column": "quantity"})
        result = sample_df.select(expr.alias("result"))["result"]
        assert result[0] == pytest.approx(1.414, rel=0.01)

    def test_abs(self, sample_df: pl.DataFrame):
        """abs function takes absolute value."""
        expr = build_expression({"function": "abs", "column": "negative"})
        result = sample_df.select(expr.alias("result"))["result"]
        assert result.to_list() == [5.0, 2.0, 0.0, 3.0]

    def test_clip(self, sample_df: pl.DataFrame):
        """clip function clips to range."""
        expr = build_expression({
            "function": "clip",
            "column": "amount",
            "min_val": 5.0,
            "max_val": 500.0,
        })
        result = sample_df.select(expr.alias("result"))["result"]
        assert result.to_list() == [5.0, 10.0, 100.0, 500.0]

    def test_ratio(self, sample_df: pl.DataFrame):
        """ratio function divides columns."""
        expr = build_expression({
            "function": "ratio",
            "columns": ["amount", "quantity"],
        })
        result = sample_df.select(expr.alias("result"))["result"]
        # 1/2, 10/5, 100/10, 1000/20
        assert result[1] == pytest.approx(2.0, rel=0.01)

    def test_ratio_requires_two_columns(self):
        """ratio should fail with wrong number of columns."""
        with pytest.raises(ValueError, match="requires exactly 2 columns"):
            expr = build_expression({
                "function": "ratio",
                "columns": ["a"],
            })
            # Need to evaluate to trigger the error
            pl.DataFrame({"a": [1]}).select(expr)

    def test_difference(self, sample_df: pl.DataFrame):
        """difference function subtracts columns."""
        expr = build_expression({
            "function": "difference",
            "columns": ["amount", "quantity"],
        })
        result = sample_df.select(expr.alias("result"))["result"]
        assert result.to_list() == [-1.0, 5.0, 90.0, 980.0]

    def test_product(self, sample_df: pl.DataFrame):
        """product function multiplies columns."""
        expr = build_expression({
            "function": "product",
            "columns": ["amount", "quantity"],
        })
        result = sample_df.select(expr.alias("result"))["result"]
        assert result.to_list() == [2.0, 50.0, 1000.0, 20000.0]

    def test_threshold(self, sample_df: pl.DataFrame):
        """threshold function creates boolean."""
        expr = build_expression({
            "function": "threshold",
            "column": "amount",
            "value": 50.0,
        })
        result = sample_df.select(expr.alias("result"))["result"]
        assert result.to_list() == [False, False, True, True]

    def test_bin(self, sample_df: pl.DataFrame):
        """bin function creates categorical bins."""
        expr = build_expression({
            "function": "bin",
            "column": "age",
            "bins": [30, 50, 65],
            "labels": ["young", "adult", "middle", "senior"],
        })
        result = sample_df.select(expr.alias("result"))["result"]
        assert result[0] == "young"  # 18 < 30
        assert result[1] == "adult"  # 35 in [30, 50)
        assert result[2] == "middle"  # 55 in [50, 65)
        assert result[3] == "senior"  # 70 >= 65


class TestStringFunctions:
    """Test string category functions."""

    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        """Sample dataframe for testing."""
        return pl.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "email": ["alice@gmail.com", "bob@yahoo.com", "charlie@gmail.com"],
        })

    def test_length(self, sample_df: pl.DataFrame):
        """length function returns string length."""
        expr = build_expression({"function": "length", "column": "name"})
        result = sample_df.select(expr.alias("result"))["result"]
        assert result.to_list() == [5, 3, 7]

    def test_lower(self, sample_df: pl.DataFrame):
        """lower function lowercases strings."""
        expr = build_expression({"function": "lower", "column": "name"})
        result = sample_df.select(expr.alias("result"))["result"]
        assert result.to_list() == ["alice", "bob", "charlie"]

    def test_upper(self, sample_df: pl.DataFrame):
        """upper function uppercases strings."""
        expr = build_expression({"function": "upper", "column": "name"})
        result = sample_df.select(expr.alias("result"))["result"]
        assert result.to_list() == ["ALICE", "BOB", "CHARLIE"]

    def test_contains(self, sample_df: pl.DataFrame):
        """contains function checks for pattern."""
        expr = build_expression({
            "function": "contains",
            "column": "email",
            "pattern": "@gmail",
        })
        result = sample_df.select(expr.alias("result"))["result"]
        assert result.to_list() == [True, False, True]


class TestDateFunctions:
    """Test date category functions."""

    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        """Sample dataframe with dates."""
        return pl.DataFrame({
            "event_date": [
                date(2024, 1, 15),  # Monday
                date(2024, 6, 22),  # Saturday
                date(2024, 12, 25),  # Wednesday
            ],
        })

    def test_dayofweek(self, sample_df: pl.DataFrame):
        """dayofweek returns day of week."""
        expr = build_expression({"function": "dayofweek", "column": "event_date"})
        result = sample_df.select(expr.alias("result"))["result"]
        assert result[0] == 1  # Monday
        assert result[1] == 6  # Saturday

    def test_month(self, sample_df: pl.DataFrame):
        """month returns month number."""
        expr = build_expression({"function": "month", "column": "event_date"})
        result = sample_df.select(expr.alias("result"))["result"]
        assert result.to_list() == [1, 6, 12]

    def test_year(self, sample_df: pl.DataFrame):
        """year returns year."""
        expr = build_expression({"function": "year", "column": "event_date"})
        result = sample_df.select(expr.alias("result"))["result"]
        assert result.to_list() == [2024, 2024, 2024]

    def test_days_since_with_reference(self, sample_df: pl.DataFrame):
        """days_since with reference date."""
        expr = build_expression({
            "function": "days_since",
            "column": "event_date",
            "reference_date": "2024-12-31",
        })
        result = sample_df.select(expr.alias("result"))["result"]
        # 2024-12-31 - 2024-01-15 = 351 days
        assert result[0] == 351


class TestNullFunctions:
    """Test null handling functions."""

    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        """Sample dataframe with nulls."""
        return pl.DataFrame({
            "value": [1.0, None, 3.0, None, 5.0],
        })

    def test_fillna(self, sample_df: pl.DataFrame):
        """fillna fills null values."""
        expr = build_expression({
            "function": "fillna",
            "column": "value",
            "value": 0.0,
        })
        result = sample_df.select(expr.alias("result"))["result"]
        assert result.to_list() == [1.0, 0.0, 3.0, 0.0, 5.0]

    def test_is_null(self, sample_df: pl.DataFrame):
        """is_null returns boolean for nulls."""
        expr = build_expression({"function": "is_null", "column": "value"})
        result = sample_df.select(expr.alias("result"))["result"]
        assert result.to_list() == [False, True, False, True, False]


class TestFeatureTransformIntegration:
    """Test registry integration with FeatureTransform."""

    def test_feature_transform_with_function_syntax(self):
        """FeatureTransform should work with function syntax."""
        from dyxgb.transforms.features import FeatureTransform

        df = pl.DataFrame({
            "amount": [1.0, 10.0, 100.0],
            "quantity": [2, 4, 8],
        })

        transform = FeatureTransform(features=[
            {"name": "amount_log", "function": "log", "column": "amount"},
            {"name": "amount_per_qty", "function": "ratio", "columns": ["amount", "quantity"]},
        ])

        result = transform.transform(df)

        assert "amount_log" in result.columns
        assert "amount_per_qty" in result.columns
        assert result["amount_log"][0] == pytest.approx(0.693, rel=0.01)

    def test_feature_transform_with_legacy_expr_syntax(self):
        """FeatureTransform should still work with legacy expr syntax."""
        from dyxgb.transforms.features import FeatureTransform

        df = pl.DataFrame({
            "x": [1, 2, 3],
        })

        transform = FeatureTransform(features=[
            {"name": "x_doubled", "expr": "pl.col('x') * 2"},
        ])

        result = transform.transform(df)
        assert result["x_doubled"].to_list() == [2, 4, 6]

    def test_feature_transform_mixed_syntax(self):
        """FeatureTransform should handle mixed function and expr syntax."""
        from dyxgb.transforms.features import FeatureTransform

        df = pl.DataFrame({
            "value": [4.0, 9.0, 16.0],
        })

        transform = FeatureTransform(features=[
            {"name": "value_sqrt", "function": "sqrt", "column": "value"},
            {"name": "value_cubed", "expr": "pl.col('value') ** 3"},
        ])

        result = transform.transform(df)
        assert result["value_sqrt"].to_list() == [2.0, 3.0, 4.0]
        assert result["value_cubed"].to_list() == [64.0, 729.0, 4096.0]
