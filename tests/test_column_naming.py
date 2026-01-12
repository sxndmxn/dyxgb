"""Tests for handling horrible column naming conventions.

These tests verify that dyxgb correctly handles real-world messy column names:
- Mixed case (camelCase, PascalCase, SCREAMING_SNAKE, lowercase)
- Inconsistent separators (-, _, spaces, none)
- Units in names: (lbs), (kg), [mm], etc.
- Special characters and unicode
- Duplicate-looking names with different cases
- Leading/trailing whitespace
- Numbers in various positions
"""

import io
import json
import random
import string
import tempfile
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from dyxgb.cli import app
from dyxgb.io import read_table, write_table


runner = CliRunner()


class HorribleColumnGenerator:
    """Generate datasets with intentionally awful column naming."""

    # Common patterns seen in real-world data
    CASE_STYLES = [
        lambda s: s.lower(),                          # lowercase
        lambda s: s.upper(),                          # UPPERCASE
        lambda s: s.title(),                          # Title Case
        lambda s: s.capitalize(),                     # Capitalize first
        lambda s: "".join(                            # camelCase
            w.capitalize() if i else w.lower()
            for i, w in enumerate(s.split("_"))
        ),
        lambda s: "".join(w.capitalize() for w in s.split("_")),  # PascalCase
        lambda s: "_".join(                           # rAnDoM_CaSe
            "".join(random.choice([c.upper(), c.lower()]) for c in w)
            for w in s.split("_")
        ),
    ]

    SEPARATORS = ["_", "-", " ", ".", "__", "--", ""]

    # Note: XGBoost doesn't allow [, ], or < in feature names
    UNIT_FORMATS = [
        "({})", "_{}", "-{}", " {}", "_{}_", " ({}) "
    ]

    UNITS = ["lbs", "kg", "mm", "cm", "m", "ft", "in", "sec", "ms", "pct", "%", "USD", "EUR"]

    PREFIXES = ["", "col_", "field_", "attr_", "var_", "x_", "f_", ""]
    SUFFIXES = ["", "_val", "_value", "_data", "_col", "_1", "_v2", ""]

    @classmethod
    def generate_horrible_name(cls, base_name: str, add_unit: bool = False) -> str:
        """Generate a horrible version of a column name."""
        # Apply random case style
        style = random.choice(cls.CASE_STYLES)
        name = style(base_name)

        # Random separator replacement
        sep = random.choice(cls.SEPARATORS)
        name = name.replace("_", sep)

        # Maybe add prefix/suffix
        if random.random() < 0.3:
            name = random.choice(cls.PREFIXES) + name
        if random.random() < 0.3:
            name = name + random.choice(cls.SUFFIXES)

        # Maybe add unit
        if add_unit or random.random() < 0.4:
            unit = random.choice(cls.UNITS)
            fmt = random.choice(cls.UNIT_FORMATS)
            name = name + fmt.format(unit)

        # Note: Leading/trailing whitespace is tested separately since CSV handling
        # strips it in many implementations. Keep internal spaces but not edge spaces.
        return name.strip()

    @classmethod
    def generate_dataset(
        cls,
        n_rows: int = 100,
        n_features: int = 10,
        include_label: bool = True,
        seed: int = 42,
    ) -> tuple[pl.DataFrame, dict[str, str]]:
        """Generate a dataset with horrible column names.

        Returns:
            Tuple of (DataFrame, mapping from horrible name to canonical name)
        """
        random.seed(seed)

        # Base feature names
        base_names = [
            "weight", "height", "age", "price", "quantity",
            "temperature", "pressure", "volume", "density", "speed",
            "duration", "distance", "count", "rate", "score",
            "value", "amount", "size", "length", "width",
        ]

        columns = {}
        data = {}

        # Generate horrible names for features
        used_names = set()
        for i in range(n_features):
            base = base_names[i % len(base_names)]
            if i >= len(base_names):
                base = f"{base}_{i // len(base_names)}"

            # Keep generating until we get a unique name
            for _ in range(100):
                horrible = cls.generate_horrible_name(base, add_unit=(i % 3 == 0))
                if horrible not in used_names:
                    break
            else:
                horrible = f"col_{i}"

            used_names.add(horrible)
            columns[horrible] = base
            data[horrible] = [random.random() * 100 for _ in range(n_rows)]

        # Add label column with horrible name
        if include_label:
            label_horrible = cls.generate_horrible_name("target_label")
            columns[label_horrible] = "target_label"
            data[label_horrible] = [random.choice(["A", "B", "C"]) for _ in range(n_rows)]

        df = pl.DataFrame(data)
        return df, columns


class TestHorribleColumnNames:
    """Test suite for horrible column naming conventions."""

    def test_basic_horrible_names(self, tmp_path: Path):
        """Test training and prediction with basic horrible names."""
        gen = HorribleColumnGenerator()
        df, col_map = gen.generate_dataset(n_rows=50, n_features=5, seed=123)

        # Save to CSV
        train_path = tmp_path / "train_horrible.csv"
        df.write_csv(train_path)

        # Find the label column
        label_col = [h for h, c in col_map.items() if c == "target_label"][0]
        feature_cols = [h for h, c in col_map.items() if c != "target_label"]

        # Train
        bundle_path = tmp_path / "model.dyxgb"
        result = runner.invoke(
            app,
            [
                "train",
                "--source", str(train_path),
                "--target", label_col,
                "--features", ",".join(feature_cols),
                "--output", str(bundle_path),
                "--quiet",
            ],
        )

        assert result.exit_code == 0, f"Training failed: {result.output}"
        assert bundle_path.exists()

        # Predict
        predict_data = df.drop(label_col)
        predict_path = tmp_path / "predict_horrible.csv"
        predict_data.write_csv(predict_path)

        result = runner.invoke(
            app,
            [
                "predict",
                "--source", str(predict_path),
                "--model", str(bundle_path),
                "--features", ",".join(feature_cols),
                "--output", "-",
                "--quiet",
            ],
        )

        assert result.exit_code == 0, f"Prediction failed: {result.output}"
        assert "predicted_label" in result.stdout

    def test_spaces_in_column_names(self, tmp_path: Path):
        """Test columns with spaces in names."""
        df = pl.DataFrame({
            "dog weight (lbs)": [10.0, 20.0, 30.0, 40.0, 50.0] * 4,
            "cat weight (kg)": [5.0, 10.0, 15.0, 20.0, 25.0] * 4,
            "animal age": [1, 2, 3, 4, 5] * 4,
            "target class": ["small", "medium", "large", "small", "medium"] * 4,
        })

        train_path = tmp_path / "spaces.csv"
        df.write_csv(train_path)

        bundle_path = tmp_path / "model.dyxgb"
        result = runner.invoke(
            app,
            [
                "train",
                "--source", str(train_path),
                "--target", "target class",
                "--output", str(bundle_path),
                "--quiet",
            ],
        )

        assert result.exit_code == 0, f"Training failed: {result.output}"

    def test_mixed_case_similar_names(self, tmp_path: Path):
        """Test columns that look similar but have different cases."""
        df = pl.DataFrame({
            "Weight": [10.0, 20.0, 30.0, 40.0, 50.0] * 4,
            "WEIGHT": [100.0, 200.0, 300.0, 400.0, 500.0] * 4,
            "weight": [1.0, 2.0, 3.0, 4.0, 5.0] * 4,
            "WeIgHt": [0.1, 0.2, 0.3, 0.4, 0.5] * 4,
            "Label": ["A", "B", "A", "B", "A"] * 4,
        })

        train_path = tmp_path / "mixed_case.csv"
        df.write_csv(train_path)

        bundle_path = tmp_path / "model.dyxgb"
        result = runner.invoke(
            app,
            [
                "train",
                "--source", str(train_path),
                "--target", "Label",
                "--features", "Weight,WEIGHT,weight,WeIgHt",
                "--output", str(bundle_path),
                "--quiet",
            ],
        )

        assert result.exit_code == 0, f"Training failed: {result.output}"

        # Verify all features are preserved in the bundle
        from dyxgb.bundle import load_bundle
        bundle = load_bundle(bundle_path)
        assert set(bundle.feature_columns) == {"Weight", "WEIGHT", "weight", "WeIgHt"}

    def test_special_characters_in_names(self, tmp_path: Path):
        """Test columns with special characters (avoiding XGBoost's [, ], < restrictions)."""
        df = pl.DataFrame({
            "price ($)": [10.0, 20.0, 30.0, 40.0, 50.0] * 4,
            "rate (%)": [0.1, 0.2, 0.3, 0.4, 0.5] * 4,
            "temp (°C)": [20.0, 25.0, 30.0, 35.0, 40.0] * 4,  # Use () instead of []
            "weight_kg/m²": [1.0, 2.0, 3.0, 4.0, 5.0] * 4,
            "class": ["X", "Y", "X", "Y", "X"] * 4,
        })

        train_path = tmp_path / "special.csv"
        df.write_csv(train_path)

        bundle_path = tmp_path / "model.dyxgb"
        result = runner.invoke(
            app,
            [
                "train",
                "--source", str(train_path),
                "--target", "class",
                "--output", str(bundle_path),
                "--quiet",
            ],
        )

        assert result.exit_code == 0, f"Training failed: {result.output}"

    def test_numeric_column_names(self, tmp_path: Path):
        """Test columns with numbers in various positions."""
        df = pl.DataFrame({
            "1_feature": [10.0, 20.0, 30.0, 40.0, 50.0] * 4,
            "feature_2": [0.1, 0.2, 0.3, 0.4, 0.5] * 4,
            "3feature4": [20.0, 25.0, 30.0, 35.0, 40.0] * 4,
            "123": [1.0, 2.0, 3.0, 4.0, 5.0] * 4,
            "target_0": ["A", "B", "A", "B", "A"] * 4,
        })

        train_path = tmp_path / "numeric.csv"
        df.write_csv(train_path)

        bundle_path = tmp_path / "model.dyxgb"
        result = runner.invoke(
            app,
            [
                "train",
                "--source", str(train_path),
                "--target", "target_0",
                "--output", str(bundle_path),
                "--quiet",
            ],
        )

        assert result.exit_code == 0, f"Training failed: {result.output}"

    def test_user_example_names(self, tmp_path: Path):
        """Test the exact example from user: dog,dog_memory-name,cat-weight (lbs), etc."""
        df = pl.DataFrame({
            "dog": [1.0, 2.0, 3.0, 4.0, 5.0] * 4,
            "dog_memory-name": [10.0, 20.0, 30.0, 40.0, 50.0] * 4,
            "cat-weight (lbs)": [5.0, 10.0, 15.0, 20.0, 25.0] * 4,
            "COW_WEIGHT_MAX": [100.0, 200.0, 300.0, 400.0, 500.0] * 4,
            "CoW_WEigHT_mIN (kg)": [50.0, 100.0, 150.0, 200.0, 250.0] * 4,
            "label": ["small", "medium", "large", "small", "medium"] * 4,
        })

        train_path = tmp_path / "user_example.csv"
        df.write_csv(train_path)

        bundle_path = tmp_path / "model.dyxgb"
        result = runner.invoke(
            app,
            [
                "train",
                "--source", str(train_path),
                "--target", "label",
                "--output", str(bundle_path),
                "--quiet",
            ],
        )

        assert result.exit_code == 0, f"Training failed: {result.output}"

        # Verify bundle preserves exact column names
        from dyxgb.bundle import load_bundle
        bundle = load_bundle(bundle_path)

        expected_features = {
            "dog", "dog_memory-name", "cat-weight (lbs)",
            "COW_WEIGHT_MAX", "CoW_WEigHT_mIN (kg)"
        }
        assert set(bundle.feature_columns) == expected_features

        # Test prediction with same horrible names
        predict_df = df.drop("label")
        predict_path = tmp_path / "predict_user.csv"
        predict_df.write_csv(predict_path)

        result = runner.invoke(
            app,
            [
                "predict",
                "--source", str(predict_path),
                "--model", str(bundle_path),
                "--output", "-",
                "--quiet",
            ],
        )

        assert result.exit_code == 0, f"Prediction failed: {result.output}"
        assert "predicted_label" in result.stdout

    def test_whitespace_variations(self, tmp_path: Path):
        """Test leading/trailing whitespace in column names."""
        # Note: Polars may strip whitespace, but CSV files might preserve it
        df = pl.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0] * 4,
            "feature2 ": [10.0, 20.0, 30.0, 40.0, 50.0] * 4,  # trailing space
            " feature3": [5.0, 10.0, 15.0, 20.0, 25.0] * 4,   # leading space
            " feature4 ": [100.0, 200.0, 300.0, 400.0, 500.0] * 4,  # both
            "label": ["A", "B", "A", "B", "A"] * 4,
        })

        train_path = tmp_path / "whitespace.csv"
        df.write_csv(train_path)

        bundle_path = tmp_path / "model.dyxgb"
        result = runner.invoke(
            app,
            [
                "train",
                "--source", str(train_path),
                "--target", "label",
                "--output", str(bundle_path),
                "--quiet",
            ],
        )

        assert result.exit_code == 0, f"Training failed: {result.output}"

    def test_unicode_column_names(self, tmp_path: Path):
        """Test columns with unicode characters."""
        df = pl.DataFrame({
            "température": [20.0, 25.0, 30.0, 35.0, 40.0] * 4,
            "重量": [1.0, 2.0, 3.0, 4.0, 5.0] * 4,  # Chinese for "weight"
            "größe": [10.0, 20.0, 30.0, 40.0, 50.0] * 4,  # German for "size"
            "価格": [100.0, 200.0, 300.0, 400.0, 500.0] * 4,  # Japanese for "price"
            "label": ["X", "Y", "X", "Y", "X"] * 4,
        })

        train_path = tmp_path / "unicode.csv"
        df.write_csv(train_path)

        bundle_path = tmp_path / "model.dyxgb"
        result = runner.invoke(
            app,
            [
                "train",
                "--source", str(train_path),
                "--target", "label",
                "--output", str(bundle_path),
                "--quiet",
            ],
        )

        assert result.exit_code == 0, f"Training failed: {result.output}"

    def test_empty_looking_names(self, tmp_path: Path):
        """Test columns with names that look empty or minimal."""
        df = pl.DataFrame({
            "_": [1.0, 2.0, 3.0, 4.0, 5.0] * 4,
            "__": [10.0, 20.0, 30.0, 40.0, 50.0] * 4,
            "___": [5.0, 10.0, 15.0, 20.0, 25.0] * 4,
            "a": [100.0, 200.0, 300.0, 400.0, 500.0] * 4,
            "y": ["pos", "neg", "pos", "neg", "pos"] * 4,
        })

        train_path = tmp_path / "minimal.csv"
        df.write_csv(train_path)

        bundle_path = tmp_path / "model.dyxgb"
        result = runner.invoke(
            app,
            [
                "train",
                "--source", str(train_path),
                "--target", "y",
                "--output", str(bundle_path),
                "--quiet",
            ],
        )

        assert result.exit_code == 0, f"Training failed: {result.output}"

    def test_long_column_names(self, tmp_path: Path):
        """Test very long column names."""
        df = pl.DataFrame({
            "this_is_a_very_long_column_name_that_describes_the_weight_of_the_animal_in_kilograms": [
                1.0, 2.0, 3.0, 4.0, 5.0
            ] * 4,
            "another_extremely_long_name_for_height_measurement_in_centimeters_recorded_at_noon": [
                10.0, 20.0, 30.0, 40.0, 50.0
            ] * 4,
            "short": [5.0, 10.0, 15.0, 20.0, 25.0] * 4,
            "target": ["A", "B", "A", "B", "A"] * 4,
        })

        train_path = tmp_path / "long_names.csv"
        df.write_csv(train_path)

        bundle_path = tmp_path / "model.dyxgb"
        result = runner.invoke(
            app,
            [
                "train",
                "--source", str(train_path),
                "--target", "target",
                "--output", str(bundle_path),
                "--quiet",
            ],
        )

        assert result.exit_code == 0, f"Training failed: {result.output}"

    def test_pipe_with_horrible_names(self, tmp_path: Path):
        """Test stdin/stdout piping with horrible column names."""
        # Train first
        df = pl.DataFrame({
            "dog_memory-name": [10.0, 20.0, 30.0, 40.0, 50.0] * 4,
            "cat-weight (lbs)": [5.0, 10.0, 15.0, 20.0, 25.0] * 4,
            "COW_WEIGHT_MAX": [100.0, 200.0, 300.0, 400.0, 500.0] * 4,
            "label": ["A", "B", "A", "B", "A"] * 4,
        })

        train_path = tmp_path / "train.csv"
        df.write_csv(train_path)

        bundle_path = tmp_path / "model.dyxgb"
        runner.invoke(
            app,
            [
                "train",
                "--source", str(train_path),
                "--target", "label",
                "--output", str(bundle_path),
                "--quiet",
            ],
        )

        # Predict via stdin
        predict_df = df.drop("label")
        csv_content = predict_df.write_csv()

        result = runner.invoke(
            app,
            [
                "predict",
                "--source", "-",
                "--model", str(bundle_path),
                "--output", "-",
                "--quiet",
            ],
            input=csv_content,
        )

        assert result.exit_code == 0, f"Stdin prediction failed: {result.output}"
        assert "predicted_label" in result.stdout

    def test_jsonl_with_horrible_names(self, tmp_path: Path):
        """Test JSONL format with horrible column names."""
        # Train first
        df = pl.DataFrame({
            "dog_memory-name": [10.0, 20.0, 30.0, 40.0, 50.0] * 4,
            "cat-weight (lbs)": [5.0, 10.0, 15.0, 20.0, 25.0] * 4,
            "label": ["A", "B", "A", "B", "A"] * 4,
        })

        train_path = tmp_path / "train.csv"
        df.write_csv(train_path)

        bundle_path = tmp_path / "model.dyxgb"
        runner.invoke(
            app,
            [
                "train",
                "--source", str(train_path),
                "--target", "label",
                "--output", str(bundle_path),
                "--quiet",
            ],
        )

        # Create JSONL input
        predict_df = df.drop("label")
        jsonl_lines = []
        for row in predict_df.iter_rows(named=True):
            jsonl_lines.append(json.dumps(row))
        jsonl_content = "\n".join(jsonl_lines)

        result = runner.invoke(
            app,
            [
                "predict",
                "--source", "-",
                "--input-format", "jsonl",
                "--model", str(bundle_path),
                "--output", "-",
                "--output-format", "jsonl",
                "--quiet",
            ],
            input=jsonl_content,
        )

        assert result.exit_code == 0, f"JSONL prediction failed: {result.output}"

        # Verify JSONL output
        for line in result.stdout.strip().split("\n"):
            obj = json.loads(line)
            assert "predicted_label" in obj

    def test_evaluate_with_horrible_names(self, tmp_path: Path):
        """Test evaluation with horrible column names."""
        df = pl.DataFrame({
            "dog_memory-name": [10.0, 20.0, 30.0, 40.0, 50.0] * 4,
            "cat-weight (lbs)": [5.0, 10.0, 15.0, 20.0, 25.0] * 4,
            "COW_WEIGHT_MAX": [100.0, 200.0, 300.0, 400.0, 500.0] * 4,
            "TaRgEt_LaBeL": ["A", "B", "A", "B", "A"] * 4,
        })

        train_path = tmp_path / "train.csv"
        test_path = tmp_path / "test.csv"
        df.write_csv(train_path)
        df.write_csv(test_path)

        bundle_path = tmp_path / "model.dyxgb"
        runner.invoke(
            app,
            [
                "train",
                "--source", str(train_path),
                "--target", "TaRgEt_LaBeL",
                "--output", str(bundle_path),
                "--quiet",
            ],
        )

        # Evaluate
        result = runner.invoke(
            app,
            [
                "evaluate",
                "--source", str(test_path),
                "--model", str(bundle_path),
                "--target", "TaRgEt_LaBeL",
                "--output", "-",
                "--quiet",
            ],
        )

        assert result.exit_code == 0, f"Evaluation failed: {result.output}"
        metrics = json.loads(result.stdout)
        assert "accuracy" in metrics

    def test_importance_with_horrible_names(self, tmp_path: Path):
        """Test feature importance with horrible column names."""
        df = pl.DataFrame({
            "dog_memory-name": [10.0, 20.0, 30.0, 40.0, 50.0] * 4,
            "cat-weight (lbs)": [5.0, 10.0, 15.0, 20.0, 25.0] * 4,
            "COW_WEIGHT_MAX": [100.0, 200.0, 300.0, 400.0, 500.0] * 4,
            "label": ["A", "B", "A", "B", "A"] * 4,
        })

        train_path = tmp_path / "train.csv"
        df.write_csv(train_path)

        bundle_path = tmp_path / "model.dyxgb"
        runner.invoke(
            app,
            [
                "train",
                "--source", str(train_path),
                "--target", "label",
                "--output", str(bundle_path),
                "--quiet",
            ],
        )

        # Get importance
        result = runner.invoke(
            app,
            [
                "importance",
                "--model", str(bundle_path),
                "--output", "-",
                "--quiet",
            ],
        )

        assert result.exit_code == 0, f"Importance failed: {result.output}"

        # Verify horrible names are in output
        output = result.stdout
        assert "dog_memory-name" in output or "cat-weight (lbs)" in output or "COW_WEIGHT_MAX" in output


class TestGeneratedDatasets:
    """Test with fully generated horrible datasets."""

    @pytest.mark.parametrize("seed", [1, 42, 123, 999, 2024])
    def test_random_horrible_datasets(self, tmp_path: Path, seed: int):
        """Test training/prediction with randomly generated horrible names."""
        gen = HorribleColumnGenerator()
        df, col_map = gen.generate_dataset(n_rows=50, n_features=8, seed=seed)

        train_path = tmp_path / f"random_{seed}.csv"
        df.write_csv(train_path)

        label_col = [h for h, c in col_map.items() if c == "target_label"][0]

        bundle_path = tmp_path / "model.dyxgb"
        result = runner.invoke(
            app,
            [
                "train",
                "--source", str(train_path),
                "--target", label_col,
                "--output", str(bundle_path),
                "--quiet",
            ],
        )

        assert result.exit_code == 0, f"Training failed (seed={seed}): {result.output}"

        # Predict
        predict_df = df.drop(label_col)
        predict_path = tmp_path / "predict.csv"
        predict_df.write_csv(predict_path)

        result = runner.invoke(
            app,
            [
                "predict",
                "--source", str(predict_path),
                "--model", str(bundle_path),
                "--output", "-",
                "--quiet",
            ],
        )

        assert result.exit_code == 0, f"Prediction failed (seed={seed}): {result.output}"

    def test_many_features_horrible_names(self, tmp_path: Path):
        """Test with many features all having horrible names."""
        gen = HorribleColumnGenerator()
        df, col_map = gen.generate_dataset(n_rows=100, n_features=50, seed=42)

        train_path = tmp_path / "many_features.csv"
        df.write_csv(train_path)

        label_col = [h for h, c in col_map.items() if c == "target_label"][0]

        bundle_path = tmp_path / "model.dyxgb"
        result = runner.invoke(
            app,
            [
                "train",
                "--source", str(train_path),
                "--target", label_col,
                "--output", str(bundle_path),
                "--quiet",
            ],
        )

        assert result.exit_code == 0, f"Training failed: {result.output}"

        # Verify all features preserved
        from dyxgb.bundle import load_bundle
        bundle = load_bundle(bundle_path)
        assert len(bundle.feature_columns) == 50
