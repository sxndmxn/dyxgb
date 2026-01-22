"""Interactive mode using InquirerPy prompts."""

from pathlib import Path
from typing import Any

import polars as pl
from InquirerPy import inquirer

from dyxgb.data.database import load_from_uri
from dyxgb.data.file import FileLoader
from dyxgb.evaluation.importance import get_feature_importance, print_importance
from dyxgb.model.predictor import Predictor
from dyxgb.model.trainer import TaskType, Trainer, save_model
from dyxgb.transforms.features import FeatureTransform
from dyxgb.transforms.registry import get_function, list_functions


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


def _parse_scalar(value: str) -> Any:
    """Parse a scalar value from text input."""
    raw = value.strip()
    if raw == "":
        return ""
    lower = raw.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _prompt_optional_float(message: str) -> float | None:
    """Prompt for an optional float; blank returns None."""
    while True:
        raw = inquirer.text(message=message, default="").execute()
        raw = raw.strip()
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            print("Enter a numeric value or leave blank.")


def _prompt_float_list(message: str) -> list[float]:
    """Prompt for a comma-separated list of floats."""
    while True:
        raw = inquirer.text(message=message).execute()
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if not parts:
            print("Enter at least one value.")
            continue
        try:
            return [float(p) for p in parts]
        except ValueError:
            print("Values must be numbers separated by commas.")


def _prompt_labels(expected: int) -> list[str]:
    """Prompt for comma-separated labels with exact count."""
    while True:
        raw = inquirer.text(
            message=f"Labels (comma-separated, {expected} total):",
        ).execute()
        labels = [p.strip() for p in raw.split(",") if p.strip()]
        if len(labels) != expected:
            print(f"Expected {expected} labels.")
            continue
        return labels


def _prompt_fill_value() -> Any:
    """Prompt for a fill value with basic parsing."""
    while True:
        raw = inquirer.text(message="Fill value (string or number):").execute()
        if raw.strip():
            return _parse_scalar(raw)
        print("Fill value cannot be empty.")


def _feature_required_columns(feature_config: dict[str, Any]) -> set[str]:
    """Get required input columns for a feature config."""
    if "columns" in feature_config:
        return set(feature_config["columns"])
    if "column" in feature_config:
        return {feature_config["column"]}
    return set()


def _collect_raw_dependencies(feature_configs: list[dict[str, Any]]) -> set[str]:
    """Collect base columns needed for feature configs."""
    derived = {cfg.get("name") for cfg in feature_configs}
    raw: set[str] = set()
    for cfg in feature_configs:
        for col in _feature_required_columns(cfg):
            if col not in derived:
                raw.add(col)
    return raw


def _filter_feature_configs(
    feature_configs: list[dict[str, Any]],
    available_columns: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Drop feature configs whose inputs are missing."""
    available = set(available_columns)
    filtered: list[dict[str, Any]] = []
    dropped: list[str] = []
    for cfg in feature_configs:
        required = _feature_required_columns(cfg)
        if required <= available:
            filtered.append(cfg)
            if "name" in cfg:
                available.add(cfg["name"])
        else:
            dropped.append(cfg.get("name", "<unknown>"))
    return filtered, dropped


def select_feature_engineering(df: pl.DataFrame) -> list[dict[str, Any]]:
    """Interactively build feature engineering configs."""
    if not inquirer.confirm(
        message="Add feature engineering?",
        default=False,
    ).execute():
        return []

    specs = sorted(list_functions(), key=lambda f: f.name)
    choices = [
        {
            "name": f"{spec.name} ({spec.category}) - {spec.description}",
            "value": spec.name,
        }
        for spec in specs
    ]

    feature_configs: list[dict[str, Any]] = []
    available_columns = list(df.columns)

    while True:
        while True:
            feature_name = inquirer.text(
                message="New feature name:",
                validate=lambda x: bool(x.strip()) or "Name is required.",
            ).execute().strip()
            if feature_name in available_columns:
                overwrite = inquirer.confirm(
                    message=f"Column '{feature_name}' exists. Overwrite?",
                    default=False,
                ).execute()
                if overwrite:
                    break
                continue
            break

        func_name = inquirer.select(
            message="Select function:",
            choices=choices,
        ).execute()
        spec = get_function(func_name)

        feature_config: dict[str, Any] = {
            "name": feature_name,
            "function": spec.name,
        }

        if spec.name in {"ratio", "difference", "product"}:
            columns = inquirer.checkbox(
                message="Select 2 columns:",
                choices=available_columns,
                validate=lambda res: len(res) == 2 or "Select exactly 2 columns.",
            ).execute()
            feature_config["columns"] = list(columns)
        else:
            column = inquirer.select(
                message="Select column:",
                choices=available_columns,
            ).execute()
            feature_config["column"] = column

        if spec.name == "clip":
            min_val = _prompt_optional_float("Minimum value (blank for none):")
            max_val = _prompt_optional_float("Maximum value (blank for none):")
            if min_val is not None:
                feature_config["min_val"] = min_val
            if max_val is not None:
                feature_config["max_val"] = max_val
        elif spec.name == "threshold":
            value = inquirer.number(
                message="Threshold value:",
                default=0,
            ).execute()
            feature_config["value"] = float(value)
        elif spec.name == "bin":
            bins = _prompt_float_list("Bin edges (comma-separated):")
            feature_config["bins"] = bins
            if inquirer.confirm(
                message="Provide custom labels?",
                default=False,
            ).execute():
                feature_config["labels"] = _prompt_labels(len(bins) + 1)
        elif spec.name == "contains":
            pattern = inquirer.text(message="Pattern to search for:").execute()
            feature_config["pattern"] = pattern
        elif spec.name == "days_since":
            reference_date = inquirer.text(
                message="Reference date (YYYY-MM-DD, blank for today):",
                default="",
            ).execute()
            if reference_date.strip():
                feature_config["reference_date"] = reference_date.strip()
        elif spec.name == "fillna":
            feature_config["value"] = _prompt_fill_value()

        feature_configs.append(feature_config)
        if feature_name not in available_columns:
            available_columns.append(feature_name)

        if not inquirer.confirm(
            message="Add another feature?",
            default=False,
        ).execute():
            break

    return feature_configs


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
                message=f"Missing column '{feat}'. Map a column or <skip>:",
                choices=choices,
            ).execute()
            if mapped != "<skip>":
                rename_map[mapped] = feat

    if rename_map:
        unknown_df = unknown_df.rename(rename_map)
        unknown_cols = set(unknown_df.columns)

    missing_after = [c for c in selected_features if c not in unknown_cols]
    if missing_after:
        print("Dropping columns not present in unknown data:", missing_after)
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

    # Optional feature engineering
    feature_configs = select_feature_engineering(train_df)
    derived_feature_names: set[str] = set()
    raw_dependencies: set[str] = set()
    if feature_configs:
        feature_transform = FeatureTransform(features=feature_configs)
        train_df = feature_transform.transform(train_df)
        derived_feature_names = {cfg["name"] for cfg in feature_configs}
        raw_dependencies = _collect_raw_dependencies(feature_configs)
    else:
        feature_transform = None

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
        if feature_transform:
            raw_feature_cols = [c for c in feature_cols if c not in derived_feature_names]
            required_raw_cols = sorted(set(raw_feature_cols) | raw_dependencies)
            unknown_df, aligned_raw_cols = align_unknown_columns(unknown_df, required_raw_cols)

            aligned_raw_set = set(aligned_raw_cols)
            dropped_raw = [c for c in raw_feature_cols if c not in aligned_raw_set]
            if dropped_raw:
                print("Dropping raw features not present in unknown data:", dropped_raw)
                feature_cols = [c for c in feature_cols if c not in dropped_raw]

            filtered_configs, dropped_derived = _filter_feature_configs(
                feature_configs,
                list(unknown_df.columns),
            )
            if dropped_derived:
                print("Dropping derived features due to missing columns:", dropped_derived)
                feature_cols = [c for c in feature_cols if c not in dropped_derived]

            if filtered_configs:
                unknown_df = FeatureTransform(features=filtered_configs).transform(unknown_df)
        else:
            unknown_df, feature_cols = align_unknown_columns(unknown_df, feature_cols)

        # Run predictions
        predict_unknowns_interactive(result, unknown_df, feature_cols)

    print("\nDone!")
