"""Contract tests for CLI Unix philosophy compliance.

These tests verify:
- stdout contains only data (no logs, progress, human output)
- stderr contains logs and human-readable output
- Exit codes follow 0/1/2 convention
- stdin/stdout piping works correctly
- Format restrictions are enforced
"""

import csv
import json
import io
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dyxgb.cli import app
from dyxgb.io import EXIT_SUCCESS, EXIT_USAGE_ERROR, EXIT_RUNTIME_ERROR


runner = CliRunner()


class TestPredictContracts:
    """Contract tests for predict command."""

    def test_predict_stdout_csv_clean(self, model_bundle: Path, predict_csv: Path):
        """Verify predict outputs clean CSV to stdout."""
        result = runner.invoke(
            app,
            [
                "predict",
                "--source", str(predict_csv),
                "--model", str(model_bundle),
                "--output", "-",
                "--quiet",
            ],
        )

        assert result.exit_code == EXIT_SUCCESS

        # stdout should be valid CSV
        stdout = result.stdout
        reader = csv.DictReader(io.StringIO(stdout))
        rows = list(reader)

        assert len(rows) == 3
        assert "predicted_label" in reader.fieldnames
        assert "confidence" in reader.fieldnames

        # Verify all rows have valid data
        for row in rows:
            assert row["predicted_label"] in ("A", "B")
            confidence = float(row["confidence"])
            assert 0.0 <= confidence <= 1.0

    def test_predict_stdin_csv(self, model_bundle: Path, predict_csv: Path):
        """Verify predict can read from stdin."""
        csv_content = predict_csv.read_text()

        result = runner.invoke(
            app,
            [
                "predict",
                "--source", "-",
                "--model", str(model_bundle),
                "--output", "-",
                "--quiet",
            ],
            input=csv_content,
        )

        assert result.exit_code == EXIT_SUCCESS

        # Verify output is valid CSV
        reader = csv.DictReader(io.StringIO(result.stdout))
        rows = list(reader)
        assert len(rows) == 3

    def test_predict_stdin_jsonl(self, model_bundle: Path, predict_jsonl: Path):
        """Verify predict can read JSONL from stdin."""
        jsonl_content = predict_jsonl.read_text()

        result = runner.invoke(
            app,
            [
                "predict",
                "--source", "-",
                "--input-format", "jsonl",
                "--model", str(model_bundle),
                "--output", "-",
                "--quiet",
            ],
            input=jsonl_content,
        )

        assert result.exit_code == EXIT_SUCCESS

        # Verify output
        reader = csv.DictReader(io.StringIO(result.stdout))
        rows = list(reader)
        assert len(rows) == 3

    def test_predict_output_jsonl(self, model_bundle: Path, predict_csv: Path):
        """Verify predict can output JSONL."""
        result = runner.invoke(
            app,
            [
                "predict",
                "--source", str(predict_csv),
                "--model", str(model_bundle),
                "--output", "-",
                "--output-format", "jsonl",
                "--quiet",
            ],
        )

        assert result.exit_code == EXIT_SUCCESS

        # Verify each line is valid JSON
        lines = result.stdout.strip().split("\n")
        assert len(lines) == 3

        for line in lines:
            obj = json.loads(line)
            assert "predicted_label" in obj
            assert "confidence" in obj

    def test_predict_parquet_rejected_for_stdout(self, model_bundle: Path, predict_csv: Path):
        """Verify parquet format is rejected for stdout."""
        result = runner.invoke(
            app,
            [
                "predict",
                "--source", str(predict_csv),
                "--model", str(model_bundle),
                "--output", "-",
                "--output-format", "parquet",
                "--quiet",
            ],
        )

        assert result.exit_code == EXIT_USAGE_ERROR
        assert "parquet" in result.stderr.lower()

    def test_predict_no_logs_on_stdout(self, model_bundle: Path, predict_csv: Path):
        """Verify no log messages pollute stdout."""
        result = runner.invoke(
            app,
            [
                "predict",
                "--source", str(predict_csv),
                "--model", str(model_bundle),
                "--output", "-",
                # No --quiet flag, so logs should go to stderr
            ],
        )

        assert result.exit_code == EXIT_SUCCESS

        # stdout should start with CSV header
        first_line = result.stdout.split("\n")[0]
        assert first_line.startswith("predicted_label")

        # No "Loading" or other log messages in stdout
        assert "Loading" not in result.stdout
        assert "Error" not in result.stdout


class TestEvaluateContracts:
    """Contract tests for evaluate command."""

    def test_evaluate_stdout_json_object(self, model_bundle: Path, test_csv: Path):
        """Verify evaluate outputs valid JSON object to stdout."""
        result = runner.invoke(
            app,
            [
                "evaluate",
                "--source", str(test_csv),
                "--model", str(model_bundle),
                "--target", "label",
                "--output", "-",
                "--quiet",
            ],
        )

        assert result.exit_code == EXIT_SUCCESS

        # stdout should be valid JSON
        metrics = json.loads(result.stdout)

        # Verify expected metrics for classification
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

        # Verify values are reasonable
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0

    def test_evaluate_no_logs_on_stdout(self, model_bundle: Path, test_csv: Path):
        """Verify no log messages in evaluate stdout."""
        result = runner.invoke(
            app,
            [
                "evaluate",
                "--source", str(test_csv),
                "--model", str(model_bundle),
                "--target", "label",
                "--output", "-",
            ],
        )

        assert result.exit_code == EXIT_SUCCESS

        # stdout should be pure JSON
        stdout = result.stdout.strip()
        assert stdout.startswith("{")
        assert stdout.endswith("}")

        # Should parse as JSON
        json.loads(stdout)


class TestImportanceContracts:
    """Contract tests for importance command."""

    def test_importance_stdout_csv_schema(self, model_bundle: Path):
        """Verify importance outputs CSV with feature,importance columns."""
        result = runner.invoke(
            app,
            [
                "importance",
                "--model", str(model_bundle),
                "--output", "-",
                "--quiet",
            ],
        )

        assert result.exit_code == EXIT_SUCCESS

        # Verify CSV schema
        reader = csv.DictReader(io.StringIO(result.stdout))
        rows = list(reader)

        assert len(rows) > 0
        assert reader.fieldnames == ["feature", "importance"]

        # Verify all rows have valid data
        for row in rows:
            assert row["feature"]
            float(row["importance"])  # Should be parseable as float

    def test_importance_output_jsonl(self, model_bundle: Path):
        """Verify importance can output JSONL."""
        result = runner.invoke(
            app,
            [
                "importance",
                "--model", str(model_bundle),
                "--output", "-",
                "--output-format", "jsonl",
                "--quiet",
            ],
        )

        assert result.exit_code == EXIT_SUCCESS

        # Verify each line is valid JSON with expected fields
        lines = result.stdout.strip().split("\n")
        assert len(lines) > 0

        for line in lines:
            obj = json.loads(line)
            assert "feature" in obj
            assert "importance" in obj

    def test_importance_top_n_filter(self, model_bundle: Path):
        """Verify --top flag limits output."""
        result = runner.invoke(
            app,
            [
                "importance",
                "--model", str(model_bundle),
                "--output", "-",
                "--top", "2",
                "--quiet",
            ],
        )

        assert result.exit_code == EXIT_SUCCESS

        reader = csv.DictReader(io.StringIO(result.stdout))
        rows = list(reader)
        assert len(rows) == 2


class TestTrainContracts:
    """Contract tests for train command."""

    def test_train_rejects_stdin(self, tmp_path: Path):
        """Verify train rejects stdin as source."""
        result = runner.invoke(
            app,
            [
                "train",
                "--source", "-",
                "--target", "label",
                "--output", str(tmp_path / "model.dyxgb"),
            ],
        )

        assert result.exit_code == EXIT_USAGE_ERROR
        assert "stdin" in result.stderr.lower()

    def test_train_creates_bundle(self, train_csv: Path, tmp_path: Path):
        """Verify train creates .dyxgb bundle file."""
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

        assert result.exit_code == EXIT_SUCCESS
        assert bundle_path.exists()

        # Verify it's a valid zip file
        import zipfile
        assert zipfile.is_zipfile(bundle_path)

        # Verify contents
        with zipfile.ZipFile(bundle_path, "r") as zf:
            names = zf.namelist()
            assert "metadata.json" in names
            assert "model.json" in names

    def test_train_logs_to_stderr(self, train_csv: Path, tmp_path: Path):
        """Verify train logs go to stderr not stdout."""
        bundle_path = tmp_path / "model.dyxgb"

        result = runner.invoke(
            app,
            [
                "train",
                "--source", str(train_csv),
                "--target", "label",
                "--output", str(bundle_path),
                # No --quiet flag
            ],
        )

        assert result.exit_code == EXIT_SUCCESS

        # stdout should be empty (or minimal)
        assert result.stdout == "" or result.stdout.strip() == ""

        # stderr should contain progress messages
        assert "Loading" in result.stderr or "Training" in result.stderr


class TestExitCodes:
    """Test exit code conventions."""

    def test_exit_code_success(self, model_bundle: Path, predict_csv: Path):
        """Verify exit code 0 on success."""
        result = runner.invoke(
            app,
            [
                "predict",
                "--source", str(predict_csv),
                "--model", str(model_bundle),
                "--output", "-",
                "--quiet",
            ],
        )
        assert result.exit_code == EXIT_SUCCESS

    def test_exit_code_usage_error(self, tmp_path: Path):
        """Verify exit code 2 on usage errors."""
        # Missing required source
        result = runner.invoke(
            app,
            [
                "train",
                "--target", "label",
                "--output", str(tmp_path / "model.dyxgb"),
            ],
        )
        assert result.exit_code == EXIT_USAGE_ERROR

    def test_exit_code_runtime_error(self, tmp_path: Path):
        """Verify exit code 1 on runtime errors."""
        # Non-existent file
        result = runner.invoke(
            app,
            [
                "predict",
                "--source", str(tmp_path / "nonexistent.csv"),
                "--model", str(tmp_path / "nonexistent.dyxgb"),
                "--output", "-",
            ],
        )
        assert result.exit_code == EXIT_RUNTIME_ERROR


class TestLegacyCompatibility:
    """Test backward compatibility with legacy format."""

    def test_predict_with_legacy_model(
        self,
        legacy_model: tuple[Path, Path],
        predict_csv: Path,
    ):
        """Verify predict works with legacy model.json + encoder.joblib."""
        model_path, encoder_path = legacy_model

        result = runner.invoke(
            app,
            [
                "predict",
                "--source", str(predict_csv),
                "--model", str(model_path),
                "--encoder", str(encoder_path),
                "--output", "-",
                "--quiet",
            ],
        )

        assert result.exit_code == EXIT_SUCCESS

        # Verify output
        reader = csv.DictReader(io.StringIO(result.stdout))
        rows = list(reader)
        assert len(rows) == 3
