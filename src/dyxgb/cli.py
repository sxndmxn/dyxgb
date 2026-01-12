"""Command-line interface for dyxgb."""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from dyxgb import __version__

app = typer.Typer(
    name="dyxgb",
    help="Dynamic XGBoost - A flexible CLI tool for XGBoost training and prediction.",
    add_completion=False,
)
console = Console()


def version_callback(value: bool) -> None:
    if value:
        console.print(f"dyxgb version {__version__}")
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
    from dyxgb.interactive import run_interactive

    run_interactive()


@app.command()
def train(
    config: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Path to YAML/TOML config file"),
    ] = None,
    source: Annotated[
        Optional[str],
        typer.Option("--source", "-s", help="Data source (file path or database URI)"),
    ] = None,
    query: Annotated[
        Optional[str],
        typer.Option("--query", "-q", help="SQL query for database sources"),
    ] = None,
    table: Annotated[
        Optional[str],
        typer.Option("--table", "-t", help="Table name for database sources"),
    ] = None,
    target: Annotated[
        Optional[str],
        typer.Option("--target", help="Target column name"),
    ] = None,
    features: Annotated[
        Optional[str],
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
        typer.Option("--output", "-o", help="Output path for model"),
    ] = "model.json",
    encoder_output: Annotated[
        Optional[str],
        typer.Option("--encoder-output", help="Output path for label encoder"),
    ] = None,
) -> None:
    """Train an XGBoost model.

    Examples:

        # From config file
        dyxgb train --config config.yaml

        # From file source
        dyxgb train --source data.csv --target label --features "f1,f2,f3"

        # From database
        dyxgb train --source "postgres://user:pass@host/db" --query "SELECT * FROM train" --target label

        # With hyperparameter tuning
        dyxgb train --config config.yaml --tune --tune-trials 100
    """
    import polars as pl
    from dyxgb.config import Config, load_config
    from dyxgb.data.file import FileLoader
    from dyxgb.data.database import load_from_uri
    from dyxgb.model.trainer import Trainer, TaskType, save_model
    from dyxgb.model.tuning import tune_hyperparameters
    from dyxgb.evaluation.importance import get_feature_importance, print_importance

    # Load config if provided
    cfg = Config()
    if config:
        console.print(f"[cyan]Loading config from {config}[/cyan]")
        cfg = load_config(config)

    # Override with CLI arguments
    if source:
        # Detect if file or database URI
        if Path(source).exists() or not source.startswith(
            ("sqlite:", "duckdb:", "postgres:", "postgresql:")
        ):
            cfg.data["train"] = type(
                "DataSourceConfig",
                (),
                {
                    "type": "file",
                    "path": source,
                    "uri": None,
                    "query": None,
                    "table": None,
                },
            )()
        else:
            cfg.data["train"] = type(
                "DataSourceConfig",
                (),
                {
                    "type": "database",
                    "path": None,
                    "uri": source,
                    "query": query,
                    "table": table,
                },
            )()

    if target:
        cfg.model.target = target
    if features:
        cfg.model.features = [f.strip() for f in features.split(",")]
    if task:
        cfg.model.task = task
    if tune:
        cfg.tuning.enabled = True
        cfg.tuning.n_trials = tune_trials

    # Validate required fields
    if "train" not in cfg.data:
        console.print(
            "[red]Error: No training data source specified. Use --source or --config[/red]"
        )
        raise typer.Exit(1)
    if not cfg.model.target:
        console.print(
            "[red]Error: No target column specified. Use --target or config file[/red]"
        )
        raise typer.Exit(1)

    # Load training data
    console.print("[cyan]Loading training data...[/cyan]")
    train_source = cfg.data["train"]

    if train_source.type == "file":
        df = FileLoader(train_source.path).load()
    else:
        df = load_from_uri(
            train_source.uri, query=train_source.query, table=train_source.table
        )

    console.print(f"[green]Loaded {len(df)} rows, {len(df.columns)} columns[/green]")

    # Determine features
    feature_cols = cfg.model.features
    if not feature_cols:
        feature_cols = [c for c in df.columns if c != cfg.model.target]
        console.print(f"[yellow]Using all columns as features: {feature_cols}[/yellow]")

    task_type = TaskType(cfg.model.task)

    # Hyperparameter tuning
    hyperparameters = None
    if cfg.tuning.enabled:
        console.print(
            f"[cyan]Running {cfg.tuning.n_trials} hyperparameter optimization trials...[/cyan]"
        )
        hyperparameters = tune_hyperparameters(
            df,
            cfg.model.target,
            feature_cols,
            task_type=task_type,
            n_trials=cfg.tuning.n_trials,
            metric=cfg.tuning.metric,
        )
        console.print("[green]Tuning complete![/green]")

    # Train model
    console.print("[cyan]Training XGBoost model...[/cyan]")
    trainer = Trainer(
        task_type=task_type,
        hyperparameters=hyperparameters or cfg.model.get_hyperparameters(),
        validation_split=cfg.model.validation_split,
        early_stopping_rounds=cfg.model.early_stopping_rounds,
    )
    result = trainer.train(df, cfg.model.target, feature_cols)

    console.print(f"[green]Training score: {result.train_score:.4f}[/green]")
    console.print(f"[green]Validation score: {result.val_score:.4f}[/green]")

    # Show feature importance
    importance = get_feature_importance(result.model, importance_type="gain")
    print_importance(importance, top_n=min(20, len(feature_cols)))

    # Save model
    model_path = output or cfg.output.model_path
    enc_path = encoder_output or (
        cfg.output.encoder_path if task_type == TaskType.CLASSIFICATION else None
    )

    save_model(result, model_path, enc_path)
    console.print(f"[green]Saved model -> {model_path}[/green]")
    if enc_path:
        console.print(f"[green]Saved encoder -> {enc_path}[/green]")


@app.command()
def predict(
    config: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Path to YAML/TOML config file"),
    ] = None,
    source: Annotated[
        Optional[str],
        typer.Option("--source", "-s", help="Data source (file path or database URI)"),
    ] = None,
    query: Annotated[
        Optional[str],
        typer.Option("--query", "-q", help="SQL query for database sources"),
    ] = None,
    table: Annotated[
        Optional[str],
        typer.Option("--table", "-t", help="Table name for database sources"),
    ] = None,
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Path to trained model"),
    ] = "model.json",
    encoder: Annotated[
        Optional[str],
        typer.Option(
            "--encoder", "-e", help="Path to label encoder (for classification)"
        ),
    ] = None,
    features: Annotated[
        Optional[str],
        typer.Option("--features", "-f", help="Comma-separated feature column names"),
    ] = None,
    task: Annotated[
        str,
        typer.Option("--task", help="Task type: classification or regression"),
    ] = "classification",
    output: Annotated[
        str,
        typer.Option("--output", "-o", help="Output path for predictions"),
    ] = "predictions.parquet",
) -> None:
    """Make predictions with a trained model.

    Examples:

        # Basic prediction
        dyxgb predict --source data.csv --model model.json --output predictions.parquet

        # With classification encoder
        dyxgb predict --source data.csv --model model.json --encoder encoder.joblib

        # From database
        dyxgb predict --source "postgres://..." --query "SELECT * FROM new_data" --model model.json
    """
    from pathlib import Path as P
    from dyxgb.config import Config, load_config
    from dyxgb.data.file import FileLoader
    from dyxgb.data.database import load_from_uri
    from dyxgb.model.predictor import Predictor
    from dyxgb.model.trainer import TaskType

    # Load config if provided
    cfg = Config()
    if config:
        console.print(f"[cyan]Loading config from {config}[/cyan]")
        cfg = load_config(config)

    # Determine source
    data_source = source
    if not data_source and "predict" in cfg.data:
        predict_cfg = cfg.data["predict"]
        data_source = predict_cfg.path or predict_cfg.uri

    if not data_source:
        console.print(
            "[red]Error: No data source specified. Use --source or --config[/red]"
        )
        raise typer.Exit(1)

    # Load data
    console.print("[cyan]Loading prediction data...[/cyan]")
    if P(data_source).exists() or not data_source.startswith(
        ("sqlite:", "duckdb:", "postgres:", "postgresql:")
    ):
        df = FileLoader(data_source).load()
    else:
        df = load_from_uri(data_source, query=query, table=table)

    console.print(f"[green]Loaded {len(df)} rows[/green]")

    # Parse features
    feature_cols = None
    if features:
        feature_cols = [f.strip() for f in features.split(",")]
    elif cfg.model.features:
        feature_cols = cfg.model.features

    # Load model and make predictions
    task_type = TaskType(task)
    predictor = Predictor.from_files(model, encoder, task_type, feature_cols)

    console.print("[cyan]Making predictions...[/cyan]")
    predictions = predictor.predict(df)

    # Save predictions
    output_path = P(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output.lower().endswith(".csv"):
        predictions.write_csv(output_path)
    else:
        predictions.write_parquet(output_path)

    console.print(f"[green]Saved {len(predictions)} predictions -> {output}[/green]")

    # Show sample
    console.print("\n[bold]Sample predictions:[/bold]")
    console.print(predictions.head(5))


@app.command()
def evaluate(
    source: Annotated[
        str,
        typer.Option("--source", "-s", help="Path to test data with true labels"),
    ],
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Path to trained model"),
    ] = "model.json",
    encoder: Annotated[
        Optional[str],
        typer.Option("--encoder", "-e", help="Path to label encoder"),
    ] = None,
    target: Annotated[
        str,
        typer.Option("--target", help="Target column name in test data"),
    ] = "label",
    features: Annotated[
        Optional[str],
        typer.Option("--features", "-f", help="Comma-separated feature column names"),
    ] = None,
    task: Annotated[
        str,
        typer.Option("--task", help="Task type: classification or regression"),
    ] = "classification",
) -> None:
    """Evaluate model performance on test data.

    Examples:

        dyxgb evaluate --source test.csv --model model.json --target label
    """
    import numpy as np
    from dyxgb.data.file import FileLoader
    from dyxgb.model.predictor import Predictor
    from dyxgb.model.trainer import TaskType
    from dyxgb.evaluation.metrics import (
        evaluate_classification,
        evaluate_regression,
        print_metrics,
    )

    # Load test data
    console.print("[cyan]Loading test data...[/cyan]")
    df = FileLoader(source).load()
    console.print(f"[green]Loaded {len(df)} rows[/green]")

    # Parse features
    feature_cols = None
    if features:
        feature_cols = [f.strip() for f in features.split(",")]

    # Load model and predict
    task_type = TaskType(task)
    predictor = Predictor.from_files(model, encoder, task_type, feature_cols)

    console.print("[cyan]Making predictions...[/cyan]")
    predictions = predictor.predict(df)

    # Get true and predicted values
    y_true = df[target].to_numpy()

    if task_type == TaskType.CLASSIFICATION:
        y_pred = predictions["predicted_label"].to_numpy()
        y_proba = None
        if "confidence" in predictions.columns:
            # For multi-class, we'd need all probabilities
            y_proba = predictions.select(
                [c for c in predictions.columns if c.startswith("prob_")]
            ).to_numpy()
            if y_proba.size == 0:
                y_proba = None

        metrics = evaluate_classification(y_true, y_pred, y_proba)
    else:
        y_pred = predictions["predicted_value"].to_numpy()
        metrics = evaluate_regression(y_true, y_pred)

    print_metrics(metrics, task_type)


@app.command()
def importance(
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Path to trained model"),
    ] = "model.json",
    output: Annotated[
        Optional[str],
        typer.Option("--output", "-o", help="Output path (JSON, CSV, or Parquet)"),
    ] = None,
    importance_type: Annotated[
        str,
        typer.Option("--type", help="Importance type: weight, gain, cover"),
    ] = "gain",
    top_n: Annotated[
        int,
        typer.Option("--top", "-n", help="Number of top features to show"),
    ] = 20,
    task: Annotated[
        str,
        typer.Option("--task", help="Task type: classification or regression"),
    ] = "classification",
) -> None:
    """Show or export feature importance.

    Examples:

        # Show top 20 features
        dyxgb importance --model model.json --top 20

        # Export to JSON
        dyxgb importance --model model.json --output importance.json
    """
    from dyxgb.model.trainer import TaskType, load_model
    from dyxgb.evaluation.importance import (
        get_feature_importance,
        export_importance,
        print_importance,
    )

    task_type = TaskType(task)
    model_obj, _ = load_model(model, task_type=task_type)

    imp = get_feature_importance(model_obj, importance_type=importance_type)

    if output:
        export_importance(imp, output)
        console.print(f"[green]Exported feature importance -> {output}[/green]")
    else:
        print_importance(imp, top_n=top_n)


if __name__ == "__main__":
    app()
