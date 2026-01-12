"""Interactive mode using InquirerPy prompts."""

from pathlib import Path

import polars as pl
from InquirerPy import inquirer

from dyxgb.data.database import load_from_uri
from dyxgb.data.file import FileLoader
from dyxgb.evaluation.importance import get_feature_importance, print_importance
from dyxgb.model.predictor import Predictor
from dyxgb.model.trainer import TaskType, Trainer, save_model


def select_data_source() -> tuple[str, dict]:
    """Interactively select data source type and configuration."""
    source_type = inquirer.select(
        message="Select data source type:",
        choices=["file", "sqlite", "duckdb", "postgres"],
    ).execute()

    config = {"type": source_type}

    if source_type == "file":
        path = inquirer.filepath(
            message="Enter path to data file:",
            validate=lambda x: Path(x).exists() or "File not found",
        ).execute()
        config["path"] = path
    else:
        uri = inquirer.text(
            message=f"Enter {source_type} connection URI:",
            default=_get_uri_example(source_type),
        ).execute()
        config["uri"] = uri

        use_query = inquirer.confirm(
            message="Use custom SQL query? (No = specify table name)",
            default=False,
        ).execute()

        if use_query:
            query = inquirer.text(
                message="Enter SQL query:",
                default="SELECT * FROM ",
            ).execute()
            config["query"] = query
        else:
            table = inquirer.text(
                message="Enter table name:",
            ).execute()
            config["table"] = table

    return source_type, config


def _get_uri_example(source_type: str) -> str:
    """Get example URI for data source type."""
    examples = {
        "sqlite": "sqlite:///path/to/database.db",
        "duckdb": "duckdb:///path/to/database.duckdb",
        "postgres": "postgres://user:password@localhost:5432/dbname",
    }
    return examples.get(source_type, "")


def load_data_interactive(config: dict) -> pl.DataFrame:
    """Load data based on interactive configuration."""
    source_type = config["type"]

    if source_type == "file":
        return FileLoader(config["path"]).load()
    else:
        return load_from_uri(
            config["uri"],
            query=config.get("query"),
            table=config.get("table"),
        )


def select_target_and_features(df: pl.DataFrame) -> tuple[str, list[str]]:
    """Interactively select target column and features."""
    cols = df.columns

    feature_cols = inquirer.checkbox(
        message="Select feature columns for training (SPACE to toggle, ENTER to confirm):",
        choices=cols,
        validate=lambda res: len(res) > 0 or "Select at least one feature.",
    ).execute()

    remaining = [c for c in cols if c not in feature_cols]
    target_col = inquirer.select(
        message="Select target column:",
        choices=remaining,
    ).execute()

    return target_col, list(feature_cols)


def select_task_type() -> TaskType:
    """Interactively select task type."""
    task = inquirer.select(
        message="Select task type:",
        choices=[
            {"name": "Classification (predict categories)", "value": "classification"},
            {"name": "Regression (predict numeric values)", "value": "regression"},
        ],
    ).execute()
    return TaskType(task)


def align_unknown_columns(
    unknown_df: pl.DataFrame,
    selected_features: list[str],
) -> tuple[pl.DataFrame, list[str]]:
    """Interactively align unknown data columns with training features."""
    unknown_cols = set(unknown_df.columns)
    rename_map: dict[str, str] = {}

    for feat in selected_features:
        if feat not in unknown_cols:
            choices = list(unknown_df.columns) + ["<skip>"]
            mapped = inquirer.select(
                message=f"Missing feature '{feat}'. Map a column or <skip>:",
                choices=choices,
            ).execute()
            if mapped != "<skip>":
                rename_map[mapped] = feat

    if rename_map:
        unknown_df = unknown_df.rename(rename_map)
        unknown_cols = set(unknown_df.columns)

    missing_after = [c for c in selected_features if c not in unknown_cols]
    if missing_after:
        print("Dropping features not present in unknown_data:", missing_after)
        selected_features = [c for c in selected_features if c in unknown_cols]

    return unknown_df, selected_features


def save_artifacts_interactive(result, task_type: TaskType) -> tuple[str, str | None]:
    """Interactively save model artifacts."""
    model_path = inquirer.text(
        message="Path to save XGBoost model (.json):",
        default="xgb_model.json",
    ).execute()

    encoder_path = None
    if task_type == TaskType.CLASSIFICATION:
        encoder_path = inquirer.text(
            message="Path to save LabelEncoder (.joblib):",
            default="label_encoder.joblib",
        ).execute()

    save_model(result, model_path, encoder_path)
    print(f"Saved model -> {model_path}")
    if encoder_path:
        print(f"Saved label encoder -> {encoder_path}")

    return model_path, encoder_path


def predict_unknowns_interactive(
    result,
    unknown_df: pl.DataFrame,
    feature_cols: list[str],
) -> None:
    """Interactively run predictions on unknown data."""
    predictor = Predictor(
        model=result.model,
        label_encoder=result.label_encoder,
        task_type=result.task_type,
        feature_columns=feature_cols,
    )

    try:
        predictions_df = predictor.predict(unknown_df)
    except ValueError as e:
        print(f"Cannot predict: {e}")
        return

    # Show sample predictions
    print("\nSample predictions:")
    print(predictions_df.head(5))

    if inquirer.confirm(
        message="Save predictions?",
        default=True,
    ).execute():
        out_path = inquirer.text(
            message="Path to save predictions (.parquet / .csv):",
            default="predictions.parquet",
        ).execute()

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if out_path.lower().endswith(".csv"):
            predictions_df.write_csv(out_path)
        else:
            predictions_df.write_parquet(out_path)
        print(f"Saved predictions -> {out_path}")


def run_interactive() -> None:
    """Run the full interactive workflow."""
    print("=== Dynamic XGBoost - Interactive Mode ===\n")

    # Load training data
    print("Configure TRAINING data source:")
    _, train_config = select_data_source()
    print("\nLoading training data...")
    train_df = load_data_interactive(train_config)
    print(f"Loaded {len(train_df)} rows, {len(train_df.columns)} columns")

    # Select task, target, features
    task_type = select_task_type()
    target_col, feature_cols = select_target_and_features(train_df)

    # Optionally tune hyperparameters
    do_tune = inquirer.confirm(
        message="Run hyperparameter tuning with Optuna?",
        default=False,
    ).execute()

    hyperparameters = None
    if do_tune:
        n_trials = inquirer.number(
            message="Number of tuning trials:",
            default=50,
            min_allowed=10,
            max_allowed=1000,
        ).execute()

        print(f"\nRunning {n_trials} optimization trials...")
        from dyxgb.model.tuning import tune_hyperparameters

        hyperparameters = tune_hyperparameters(
            train_df,
            target_col,
            feature_cols,
            task_type=task_type,
            n_trials=int(n_trials),
        )
        print("Tuning complete!")

    # Train model
    print("\nTraining XGBoost model...")
    trainer = Trainer(
        task_type=task_type,
        hyperparameters=hyperparameters,
    )
    result = trainer.train(train_df, target_col, feature_cols)

    print(f"Training score: {result.train_score:.4f}")
    print(f"Validation score: {result.val_score:.4f}")

    # Show feature importance
    if inquirer.confirm(
        message="Show feature importance?",
        default=True,
    ).execute():
        importance = get_feature_importance(result.model, importance_type="gain")
        print_importance(importance, top_n=min(20, len(feature_cols)))

    # Save model
    model_path, encoder_path = save_artifacts_interactive(result, task_type)

    # Load unknown data for prediction
    if inquirer.confirm(
        message="Load data for prediction?",
        default=True,
    ).execute():
        print("\nConfigure PREDICTION data source:")
        _, predict_config = select_data_source()
        print("\nLoading prediction data...")
        unknown_df = load_data_interactive(predict_config)
        print(f"Loaded {len(unknown_df)} rows")

        # Align columns
        unknown_df, feature_cols = align_unknown_columns(unknown_df, feature_cols)

        # Run predictions
        predict_unknowns_interactive(result, unknown_df, feature_cols)

    print("\nDone!")
