"""Core API layer for dyxgb.

This module provides pure functions that can be called from:
- CLI commands
- Python library usage
- Tests

All functions here:
- Take DataFrames and config objects as input
- Return data structures (not formatted strings)
- Do NOT perform I/O (file reading/writing, stdin/stdout)
- Do NOT print anything
- Raise exceptions for errors (let caller handle)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import polars as pl

from dyxgb.model.trainer import TaskType, TrainResult, Trainer, HyperParameters

if TYPE_CHECKING:
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier, XGBRegressor
    from dyxgb.transforms import TransformPipeline
    from dyxgb.bundle import Bundle


@dataclass
class PredictResult:
    """Result of prediction operation."""

    predictions: pl.DataFrame
    task_type: TaskType
    feature_columns: list[str]


@dataclass
class EvaluateResult:
    """Result of evaluation operation."""

    metrics: dict[str, Any]
    task_type: TaskType
    predictions: pl.DataFrame | None = None


@dataclass
class ImportanceResult:
    """Result of importance extraction."""

    features: list[str]
    importance_values: list[float]
    importance_type: str

    def to_dataframe(self) -> pl.DataFrame:
        """Convert to DataFrame with feature,importance columns."""
        return pl.DataFrame({
            "feature": self.features,
            "importance": self.importance_values,
        }).sort("importance", descending=True)

    def to_dict(self) -> dict[str, float]:
        """Convert to dict mapping feature -> importance."""
        return dict(zip(self.features, self.importance_values))


def train_model(
    df: pl.DataFrame,
    target_column: str,
    feature_columns: list[str] | None = None,
    task_type: TaskType | str = TaskType.CLASSIFICATION,
    hyperparameters: HyperParameters | dict[str, Any] | None = None,
    validation_split: float = 0.2,
    early_stopping_rounds: int | None = 50,
    pipeline: "TransformPipeline | None" = None,
) -> TrainResult:
    """Train an XGBoost model.

    Args:
        df: Training data.
        target_column: Name of target column.
        feature_columns: Feature columns. If None, uses all except target.
        task_type: Classification or regression.
        hyperparameters: Model hyperparameters.
        validation_split: Fraction for validation.
        early_stopping_rounds: Early stopping configuration.
        pipeline: Optional transform pipeline to fit and apply.

    Returns:
        TrainResult with model and metadata.
    """
    if isinstance(task_type, str):
        task_type = TaskType(task_type)

    # Apply pipeline if provided
    if pipeline is not None:
        df = pipeline.fit_transform(df, target_column=target_column)

    trainer = Trainer(
        task_type=task_type,
        hyperparameters=hyperparameters,
        validation_split=validation_split,
        early_stopping_rounds=early_stopping_rounds,
    )

    return trainer.train(df, target_column, feature_columns)


def predict_df(
    df: pl.DataFrame,
    bundle: "Bundle",
    feature_columns: list[str] | None = None,
    include_probabilities: bool = True,
    output_columns: list[str] | None = None,
) -> PredictResult:
    """Make predictions on a DataFrame.

    Args:
        df: Data to predict on.
        bundle: Loaded model bundle.
        feature_columns: Features to use. If None, uses bundle's features.
        include_probabilities: Include probability columns for classification.
        output_columns: Columns to include in output. If None, includes only predictions.

    Returns:
        PredictResult with predictions DataFrame.
    """
    from dyxgb.model.predictor import Predictor

    # Apply pipeline if bundle has one
    if bundle.pipeline is not None:
        df = bundle.pipeline.transform(df)

    # Determine feature columns
    features = feature_columns or bundle.feature_columns
    if not features:
        # Try to get from model
        try:
            features = list(bundle.model.get_booster().feature_names or [])
        except (AttributeError, TypeError):
            raise ValueError(
                "Feature columns must be specified - not available in model or bundle"
            )

    # Create predictor and predict
    predictor = Predictor(
        model=bundle.model,
        label_encoder=bundle.label_encoder,
        task_type=bundle.task_type,
        feature_columns=features,
    )

    predictions_df = predictor.predict(df, include_probabilities=include_probabilities)

    # Filter to output columns if specified
    if output_columns is not None:
        # Always include prediction columns
        pred_cols = _get_prediction_columns(predictions_df, bundle.task_type)
        all_cols = list(dict.fromkeys(output_columns + pred_cols))
        available = [c for c in all_cols if c in predictions_df.columns]
        predictions_df = predictions_df.select(available)

    return PredictResult(
        predictions=predictions_df,
        task_type=bundle.task_type,
        feature_columns=features,
    )


def _get_prediction_columns(df: pl.DataFrame, task_type: TaskType) -> list[str]:
    """Get prediction-related columns from DataFrame."""
    if task_type == TaskType.CLASSIFICATION:
        cols = ["predicted_label", "confidence"]
        # Add probability columns
        cols.extend([c for c in df.columns if c.startswith("prob_")])
    else:
        cols = ["predicted_value"]
    return [c for c in cols if c in df.columns]


def evaluate_df(
    df: pl.DataFrame,
    bundle: "Bundle",
    target_column: str,
    feature_columns: list[str] | None = None,
) -> EvaluateResult:
    """Evaluate model on test data.

    Args:
        df: Test data with true labels.
        bundle: Loaded model bundle.
        target_column: Column with true labels.
        feature_columns: Feature columns. If None, uses bundle's features.

    Returns:
        EvaluateResult with metrics dict.
    """
    import numpy as np
    from dyxgb.evaluation.metrics import evaluate_classification, evaluate_regression

    # Get predictions
    pred_result = predict_df(
        df,
        bundle,
        feature_columns=feature_columns,
        include_probabilities=True,
    )

    # Extract true and predicted values
    y_true = df[target_column].to_numpy()

    if bundle.task_type == TaskType.CLASSIFICATION:
        y_pred = pred_result.predictions["predicted_label"].to_numpy()

        # Get probabilities if available
        y_proba = None
        prob_cols = [c for c in pred_result.predictions.columns if c.startswith("prob_")]
        if prob_cols:
            y_proba = pred_result.predictions.select(prob_cols).to_numpy()

        metrics_obj = evaluate_classification(y_true, y_pred, y_proba)
    else:
        y_pred = pred_result.predictions["predicted_value"].to_numpy()
        metrics_obj = evaluate_regression(y_true, y_pred)

    return EvaluateResult(
        metrics=metrics_obj.to_dict(),
        task_type=bundle.task_type,
        predictions=pred_result.predictions,
    )


def get_importance(
    bundle: "Bundle",
    importance_type: str = "gain",
) -> ImportanceResult:
    """Extract feature importance from model.

    Args:
        bundle: Loaded model bundle.
        importance_type: Type of importance (weight, gain, cover).

    Returns:
        ImportanceResult with feature importance data.
    """
    from dyxgb.evaluation.importance import get_feature_importance

    importance = get_feature_importance(
        bundle.model,
        importance_type=importance_type,
        feature_names=bundle.feature_columns or None,
    )

    return ImportanceResult(
        features=importance.feature_names,
        importance_values=importance.importance_values,
        importance_type=importance.importance_type,
    )


def tune_model(
    df: pl.DataFrame,
    target_column: str,
    feature_columns: list[str],
    task_type: TaskType | str = TaskType.CLASSIFICATION,
    n_trials: int = 50,
    metric: str | None = None,
    timeout: int | None = None,
    cv_folds: int = 5,
) -> HyperParameters:
    """Run hyperparameter tuning with Optuna.

    Args:
        df: Training data.
        target_column: Target column name.
        feature_columns: Feature columns.
        task_type: Classification or regression.
        n_trials: Number of Optuna trials.
        metric: Metric to optimize.
        timeout: Timeout in seconds.
        cv_folds: Cross-validation folds.

    Returns:
        Best hyperparameters found.
    """
    from dyxgb.model.tuning import tune_hyperparameters

    if isinstance(task_type, str):
        task_type = TaskType(task_type)

    return tune_hyperparameters(
        df,
        target_column,
        feature_columns,
        task_type=task_type,
        n_trials=n_trials,
        metric=metric,
    )
