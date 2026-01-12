"""Test fixtures for dyxgb."""

import os
from pathlib import Path

import pytest

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def train_csv() -> Path:
    """Path to training CSV file."""
    return TEST_DATA_DIR / "train.csv"


@pytest.fixture
def test_csv() -> Path:
    """Path to test CSV file with labels."""
    return TEST_DATA_DIR / "test.csv"


@pytest.fixture
def predict_csv() -> Path:
    """Path to prediction CSV file (no labels)."""
    return TEST_DATA_DIR / "predict.csv"


@pytest.fixture
def predict_jsonl() -> Path:
    """Path to prediction JSONL file."""
    return TEST_DATA_DIR / "predict.jsonl"


@pytest.fixture
def model_bundle(train_csv: Path, tmp_path: Path) -> Path:
    """Train a model and return path to bundle."""
    from typer.testing import CliRunner
    from dyxgb.cli import app

    runner = CliRunner()
    bundle_path = tmp_path / "model.dyxgb"

    result = runner.invoke(
        app,
        [
            "train",
            "--source", str(train_csv),
            "--target", "label",
            "--output", str(bundle_path),
            "--quiet",
        ],
    )

    if result.exit_code != 0:
        raise RuntimeError(f"Failed to train model: {result.output}")

    return bundle_path


@pytest.fixture
def legacy_model(train_csv: Path, tmp_path: Path) -> tuple[Path, Path]:
    """Train a model in legacy format (separate files)."""
    import polars as pl
    from dyxgb.model.trainer import Trainer, TaskType, save_model

    df = pl.read_csv(train_csv)
    trainer = Trainer(
        task_type=TaskType.CLASSIFICATION,
        early_stopping_rounds=10,
    )
    result = trainer.train(df, "label")

    model_path = tmp_path / "model.json"
    encoder_path = tmp_path / "encoder.joblib"
    save_model(result, model_path, encoder_path)

    return model_path, encoder_path
