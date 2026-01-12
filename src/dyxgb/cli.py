"""Command-line interface for dyxgb.

Unix Philosophy:
- stdout: data only (CSV, JSONL, JSON)
- stderr: logs, progress, human-readable output
- Exit codes: 0 (success), 1 (runtime error), 2 (usage error)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from dyxgb import __version__
from dyxgb.io import (
    EXIT_RUNTIME_ERROR,
    EXIT_USAGE_ERROR,
    RuntimeIOError,
    UsageError,
    is_stdin_source,
    is_stdout_dest,
    is_tty,
    read_table,
    stderr_print,
    write_json,
    write_table,
)

app = typer.Typer(
    name="dyxgb",
    help="Dynamic XGBoost - A flexible CLI tool for XGBoost training and prediction.",
    add_completion=False,
)


def _stderr_console():
    """Get Rich console for stderr output."""
    try:
        from rich.console import Console

        return Console(stderr=True, force_terminal=is_tty(sys.stderr))
    except ImportError:
        return None


console = _stderr_console()


def _print(msg: str, style: str | None = None) -> None:
    """Print to stderr with optional Rich styling."""
    if console:
        if style:
            console.print(f"[{style}]{msg}[/{style}]")
        else:
            console.print(msg)
    else:
        stderr_print(msg)


def _error(msg: str) -> None:
    """Print error to stderr."""
    _print(f"Error: {msg}", "red")


def _info(msg: str) -> None:
    """Print info to stderr."""
    _print(msg, "cyan")


def _success(msg: str) -> None:
    """Print success to stderr."""
    _print(msg, "green")


def _warning(msg: str) -> None:
    """Print warning to stderr."""
    _print(msg, "yellow")


def version_callback(value: bool) -> None:
    if value:
        stderr_print(f"dyxgb version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", "-v", callback=version_callback, is_eager=True),
    ] = False,
) -> None:
    """Dynamic XGBoost CLI."""
    pass


@app.command()
def interactive() -> None:
    """Run in interactive mode with prompts."""
    try:
        from dyxgb.interactive import run_interactive
    except ImportError:
        _error("Interactive mode requires 'inquirerpy'. Install: pip install dyxgb[interactive]")
        raise typer.Exit(EXIT_USAGE_ERROR)

    run_interactive()


@app.command()
def train(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to YAML/TOML config file"),
    ] = None,
    source: Annotated[
        str | None,
        typer.Option("--source", "-s", help="Data source (file path or database URI)"),
    ] = None,
    query: Annotated[
        str | None,
        typer.Option("--query", "-q", help="SQL query for database sources"),
    ] = None,
    table: Annotated[
        str | None,
        typer.Option("--table", "-t", help="Table name for database sources"),
    ] = None,
    target: Annotated[
        str | None,
        typer.Option("--target", help="Target column name"),
    ] = None,
    features: Annotated[
        str | None,
        typer.Option("--features", "-f", help="Comma-separated feature column names"),
    ] = None,
    task: Annotated[
        str,
        typer.Option("--task", help="Task type: classification or regression"),
    ] = "classification",
    tune: Annotated[
        bool,
        typer.Option("--tune", help="Run hyperparameter tuning with Optuna"),
    ] = False,
    tune_trials: Annotated[
        int,
        typer.Option("--tune-trials", help="Number of Optuna trials"),
    ] = 50,
    output: Annotated[
        str,
        typer.Option("--output", "-o", help="Output path for model bundle"),
    ] = "model.dyxgb",
    quiet: Annotated[
        bool,
        typer.Option("--quiet", help="Minimize stderr output"),
    ] = False,
) -> None:
    """Train an XGBoost model.

    Training ONLY supports file or database sources (not stdin).
    Output is a .dyxgb bundle file containing model, encoder, and metadata.

    Examples:

        # From file source
        dyxgb train --source data.csv --target label --output model.dyxgb

        # From database
        dyxgb train --source "postgres://..." --query "SELECT * FROM train" --target label

        # With hyperparameter tuning
        dyxgb train --source data.parquet --target y --tune --tune-trials 100
    """
    from dyxgb.api import train_model, tune_model
    from dyxgb.bundle import save_bundle
    from dyxgb.config import Config, DataSourceConfig, load_config
    from dyxgb.model.trainer import TaskType
    from dyxgb.transforms import TransformPipeline

    try:
        # Reject stdin for training
        if source == "-":
            _error("Training from stdin is not supported. Use a file path or database URI.")
            raise typer.Exit(EXIT_USAGE_ERROR)

        # Load config if provided
        cfg = Config()
        if config:
            if not quiet:
                _info(f"Loading config from {config}")
            cfg = load_config(config)

        # Override with CLI arguments
        if source:
            if Path(source).exists() or not source.startswith(
                ("sqlite:", "duckdb:", "postgres:", "postgresql:")
            ):
                cfg.data["train"] = DataSourceConfig(type="file", path=source)
            else:
                cfg.data["train"] = DataSourceConfig(
                    type="database", uri=source, query=query, table=table
                )

        if target:
            cfg.model.target = target
        if features:
            cfg.model.features = [f.strip() for f in features.split(",")]

        # Validate required fields
        if "train" not in cfg.data:
            _error("No training data source specified. Use --source or --config")
            raise typer.Exit(EXIT_USAGE_ERROR)
        if not cfg.model.target:
            _error("No target column specified. Use --target or config file")
            raise typer.Exit(EXIT_USAGE_ERROR)

        # Load training data
        if not quiet:
            _info("Loading training data...")

        train_source = cfg.data["train"]
        data_source = train_source.path or train_source.uri
        if not data_source:
            _error("Invalid data source configuration")
            raise typer.Exit(EXIT_USAGE_ERROR)

        df = read_table(data_source, query=train_source.query, table=train_source.table)

        if not quiet:
            _success(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        # Apply transforms if configured
        pipeline: TransformPipeline | None = None
        if cfg.transforms:
            if not quiet:
                _info("Building transform pipeline...")
            pipeline = TransformPipeline.from_config(cfg.transforms.to_pipeline_config())
            if not quiet:
                _info(f"Fitting and applying {len(pipeline)} transforms...")
            df = pipeline.fit_transform(df, target_column=cfg.model.target)
            if not quiet:
                _success(f"Transformed data: {len(df)} rows, {len(df.columns)} columns")

        # Determine features
        feature_cols = cfg.model.features
        if not feature_cols:
            feature_cols = [c for c in df.columns if c != cfg.model.target]
            if not quiet:
                _warning(f"Using all columns as features: {feature_cols}")

        task_type = TaskType(task)

        # Hyperparameter tuning
        hyperparameters = None
        if tune:
            if not quiet:
                _info(f"Running {tune_trials} hyperparameter optimization trials...")
            hyperparameters = tune_model(
                df,
                cfg.model.target,
                feature_cols,
                task_type=task_type,
                n_trials=tune_trials,
                metric=cfg.tuning.metric,
            )
            if not quiet:
                _success("Tuning complete!")

        # Train model
        if not quiet:
            _info("Training XGBoost model...")

        result = train_model(
            df,
            cfg.model.target,
            feature_cols,
            task_type=task_type,
            hyperparameters=hyperparameters or cfg.model.get_hyperparameters(),
            validation_split=cfg.model.validation_split,
            early_stopping_rounds=cfg.model.early_stopping_rounds,
        )

        if not quiet:
            _success(f"Training score: {result.train_score:.4f}")
            _success(f"Validation score: {result.val_score:.4f}")

        # Save bundle
        output_path = Path(output)

        # Safety: refuse to write binary to TTY stdout
        if str(output) == "-":
            if is_tty(sys.stdout):
                _error("Cannot write binary bundle to terminal. Use --output FILE or redirect.")
                raise typer.Exit(EXIT_USAGE_ERROR)
            # Writing to stdout pipe is allowed but unusual for training
            _warning("Writing bundle to stdout...")

        save_bundle(
            output_path,
            result.model,
            task_type,
            feature_cols,
            cfg.model.target,
            label_encoder=result.label_encoder,
            pipeline=pipeline,
            train_score=result.train_score,
            val_score=result.val_score,
        )

        if not quiet:
            _success(f"Saved model bundle -> {output}")

    except typer.Exit:
        raise  # Re-raise Exit exceptions to preserve exit code
    except (UsageError, RuntimeIOError) as e:
        _error(str(e))
        raise typer.Exit(e.exit_code)
    except Exception as e:
        _error(f"Training failed: {e}")
        raise typer.Exit(EXIT_RUNTIME_ERROR)


@app.command()
def predict(
    source: Annotated[
        str,
        typer.Option("--source", "-s", help="Data source (file, database URI, or - for stdin)"),
    ] = "-",
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Path to model bundle or model.json"),
    ] = "model.dyxgb",
    encoder: Annotated[
        str | None,
        typer.Option("--encoder", "-e", help="Path to label encoder (legacy mode)"),
    ] = None,
    pipeline_input: Annotated[
        str | None,
        typer.Option("--pipeline", "-p", help="Path to transform pipeline (legacy mode)"),
    ] = None,
    features: Annotated[
        str | None,
        typer.Option("--features", "-f", help="Comma-separated feature column names"),
    ] = None,
    task: Annotated[
        str,
        typer.Option("--task", help="Task type (legacy mode): classification or regression"),
    ] = "classification",
    output: Annotated[
        str,
        typer.Option("--output", "-o", help="Output path (- for stdout)"),
    ] = "-",
    input_format: Annotated[
        str | None,
        typer.Option("--input-format", help="Input format for stdin: csv or jsonl"),
    ] = None,
    output_format: Annotated[
        str | None,
        typer.Option("--output-format", help="Output format: csv, jsonl, parquet"),
    ] = None,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", help="Minimize stderr output"),
    ] = False,
) -> None:
    """Make predictions with a trained model.

    Reads from stdin by default, writes CSV to stdout by default.
    Perfect for Unix pipelines.

    Examples:

        # Pipe from stdin to stdout (default)
        dyxgb predict --model model.dyxgb < new.csv > preds.csv

        # Explicit stdin/stdout
        dyxgb predict --source - --output - --model model.dyxgb

        # JSONL format
        dyxgb predict --source - --input-format jsonl --output-format jsonl < data.jsonl

        # From file to file
        dyxgb predict --source data.csv --model model.dyxgb --output predictions.parquet
    """
    from dyxgb.api import predict_df
    from dyxgb.bundle import load_model_or_bundle
    from dyxgb.model.trainer import TaskType

    try:
        # Load model
        if not quiet and not is_stdin_source(source):
            _info("Loading model...")

        bundle = load_model_or_bundle(
            model,
            encoder_path=encoder,
            pipeline_path=pipeline_input,
            task_type=TaskType(task),
        )

        # Load data
        if not quiet and not is_stdin_source(source):
            _info("Loading prediction data...")

        df = read_table(source, input_format=input_format)

        if not quiet and not is_stdout_dest(output):
            _success(f"Loaded {len(df)} rows")

        # Parse features
        feature_cols = None
        if features:
            feature_cols = [f.strip() for f in features.split(",")]

        # Make predictions
        if not quiet and not is_stdout_dest(output):
            _info("Making predictions...")

        result = predict_df(
            df,
            bundle,
            feature_columns=feature_cols,
            include_probabilities=True,
        )

        # Determine output columns (predictions only for clean output)
        if bundle.task_type == TaskType.CLASSIFICATION:
            output_cols = ["predicted_label", "confidence"]
            # Add probability columns
            output_cols.extend([c for c in result.predictions.columns if c.startswith("prob_")])
        else:
            output_cols = ["predicted_value"]

        # Write output
        write_table(
            result.predictions.select(output_cols),
            output,
            output_format=output_format,
        )

        if not quiet and not is_stdout_dest(output):
            _success(f"Saved {len(result.predictions)} predictions -> {output}")

    except typer.Exit:
        raise
    except (UsageError, RuntimeIOError) as e:
        _error(str(e))
        raise typer.Exit(e.exit_code)
    except Exception as e:
        _error(f"Prediction failed: {e}")
        raise typer.Exit(EXIT_RUNTIME_ERROR)


@app.command()
def evaluate(
    source: Annotated[
        str,
        typer.Option("--source", "-s", help="Path to test data with true labels"),
    ],
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Path to model bundle or model.json"),
    ] = "model.dyxgb",
    encoder: Annotated[
        str | None,
        typer.Option("--encoder", "-e", help="Path to label encoder (legacy mode)"),
    ] = None,
    target: Annotated[
        str,
        typer.Option("--target", help="Target column name in test data"),
    ] = "label",
    features: Annotated[
        str | None,
        typer.Option("--features", "-f", help="Comma-separated feature column names"),
    ] = None,
    task: Annotated[
        str,
        typer.Option("--task", help="Task type (legacy mode): classification or regression"),
    ] = "classification",
    output: Annotated[
        str,
        typer.Option("--output", "-o", help="Output path for metrics JSON (- for stdout)"),
    ] = "-",
    quiet: Annotated[
        bool,
        typer.Option("--quiet", help="Minimize stderr output"),
    ] = False,
) -> None:
    """Evaluate model performance on test data.

    Outputs a JSON object to stdout by default.

    Examples:

        # Evaluate and get JSON metrics on stdout
        dyxgb evaluate --source test.csv --model model.dyxgb --target label > metrics.json

        # Parse with jq
        dyxgb evaluate --source test.csv --model model.dyxgb --target y | jq '.accuracy'
    """
    from dyxgb.api import evaluate_df
    from dyxgb.bundle import load_model_or_bundle
    from dyxgb.model.trainer import TaskType

    try:
        # Load model
        if not quiet:
            _info("Loading model...")

        bundle = load_model_or_bundle(
            model,
            encoder_path=encoder,
            task_type=TaskType(task),
        )

        # Load test data
        if not quiet:
            _info("Loading test data...")

        df = read_table(source)

        if not quiet:
            _success(f"Loaded {len(df)} rows")

        # Parse features
        feature_cols = None
        if features:
            feature_cols = [f.strip() for f in features.split(",")]

        # Evaluate
        if not quiet:
            _info("Evaluating model...")

        result = evaluate_df(df, bundle, target, feature_columns=feature_cols)

        # Output JSON
        write_json(result.metrics, output)

        if not quiet and not is_stdout_dest(output):
            _success(f"Saved metrics -> {output}")

    except typer.Exit:
        raise
    except (UsageError, RuntimeIOError) as e:
        _error(str(e))
        raise typer.Exit(e.exit_code)
    except Exception as e:
        _error(f"Evaluation failed: {e}")
        raise typer.Exit(EXIT_RUNTIME_ERROR)


@app.command()
def importance(
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Path to model bundle or model.json"),
    ] = "model.dyxgb",
    output: Annotated[
        str,
        typer.Option("--output", "-o", help="Output path (- for stdout)"),
    ] = "-",
    output_format: Annotated[
        str | None,
        typer.Option("--output-format", help="Output format: csv, jsonl, parquet"),
    ] = None,
    importance_type: Annotated[
        str,
        typer.Option("--type", help="Importance type: weight, gain, cover"),
    ] = "gain",
    top_n: Annotated[
        int | None,
        typer.Option("--top", "-n", help="Limit to top N features"),
    ] = None,
    task: Annotated[
        str,
        typer.Option("--task", help="Task type (legacy mode): classification or regression"),
    ] = "classification",
    quiet: Annotated[
        bool,
        typer.Option("--quiet", help="Minimize stderr output"),
    ] = False,
) -> None:
    """Show or export feature importance.

    Outputs CSV to stdout by default with columns: feature,importance

    Examples:

        # Get importance as CSV
        dyxgb importance --model model.dyxgb > importance.csv

        # Get top 10 as JSONL
        dyxgb importance --model model.dyxgb --top 10 --output-format jsonl

        # Save to parquet file
        dyxgb importance --model model.dyxgb --output importance.parquet
    """
    from dyxgb.api import get_importance
    from dyxgb.bundle import load_model_or_bundle
    from dyxgb.model.trainer import TaskType

    try:
        # Load model
        if not quiet:
            _info("Loading model...")

        bundle = load_model_or_bundle(model, task_type=TaskType(task))

        # Get importance
        result = get_importance(bundle, importance_type=importance_type)

        # Convert to DataFrame
        importance_df = result.to_dataframe()

        # Apply top_n filter
        if top_n is not None and top_n > 0:
            importance_df = importance_df.head(top_n)

        # Write output (CSV columns: feature,importance)
        write_table(
            importance_df.select(["feature", "importance"]),
            output,
            output_format=output_format,
        )

        if not quiet and not is_stdout_dest(output):
            _success(f"Saved importance -> {output}")

    except typer.Exit:
        raise
    except (UsageError, RuntimeIOError) as e:
        _error(str(e))
        raise typer.Exit(e.exit_code)
    except Exception as e:
        _error(f"Importance extraction failed: {e}")
        raise typer.Exit(EXIT_RUNTIME_ERROR)


if __name__ == "__main__":
    app()
