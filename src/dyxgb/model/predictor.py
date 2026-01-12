"""Prediction module for trained XGBoost models."""

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from dyxgb.model.trainer import TaskType, load_model


class Predictor:
    """Make predictions with trained XGBoost models."""

    def __init__(
        self,
        model: XGBClassifier | XGBRegressor,
        label_encoder: LabelEncoder | None = None,
        task_type: TaskType = TaskType.CLASSIFICATION,
        feature_columns: list[str] | None = None,
    ):
        self.model = model
        self.label_encoder = label_encoder
        self.task_type = task_type
        self.feature_columns = feature_columns

    @classmethod
    def from_files(
        cls,
        model_path: str | Path,
        encoder_path: str | Path | None = None,
        task_type: TaskType | str = TaskType.CLASSIFICATION,
        feature_columns: list[str] | None = None,
    ) -> "Predictor":
        """Load predictor from saved model files."""
        if isinstance(task_type, str):
            task_type = TaskType(task_type)

        model, label_encoder = load_model(model_path, encoder_path, task_type)
        return cls(model, label_encoder, task_type, feature_columns)

    def predict(
        self,
        df: pl.DataFrame,
        feature_columns: list[str] | None = None,
        include_probabilities: bool = True,
    ) -> pl.DataFrame:
        """Make predictions on new data.

        Args:
            df: Data to predict on
            feature_columns: Feature columns to use. If None, uses columns from init
                           or tries to use model's feature names.
            include_probabilities: For classification, include prediction probabilities

        Returns:
            Original DataFrame with prediction columns added
        """
        # Determine feature columns
        features = feature_columns or self.feature_columns
        if features is None:
            # Try to get from model
            try:
                features = list(self.model.get_booster().feature_names)
            except (AttributeError, TypeError):
                raise ValueError(
                    "Feature columns must be specified either in constructor, "
                    "predict() call, or be available in the model"
                )

        # Validate columns exist
        missing = set(features) - set(df.columns)
        if missing:
            raise ValueError(f"Missing feature columns in data: {missing}")

        # Prepare data
        X = df.select(features).to_pandas()

        if self.task_type == TaskType.CLASSIFICATION:
            return self._predict_classification(df, X, include_probabilities)
        else:
            return self._predict_regression(df, X)

    def _predict_classification(
        self,
        df: pl.DataFrame,
        X: Any,
        include_probabilities: bool,
    ) -> pl.DataFrame:
        """Make classification predictions."""
        # Get probabilities
        proba = self.model.predict_proba(X)

        # Get predicted class indices
        pred_indices = proba.argmax(axis=1)

        # Convert back to original labels if encoder exists
        if self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(pred_indices)
        else:
            predictions = pred_indices

        # Get confidence (max probability)
        confidences = proba.max(axis=1)

        # Build result DataFrame
        result = df.with_columns(
            pl.Series("predicted_label", predictions),
            pl.Series("confidence", confidences),
        )

        # Optionally add per-class probabilities
        if include_probabilities and self.label_encoder is not None:
            for i, class_name in enumerate(self.label_encoder.classes_):
                result = result.with_columns(
                    pl.Series(f"prob_{class_name}", proba[:, i])
                )

        return result

    def _predict_regression(self, df: pl.DataFrame, X: Any) -> pl.DataFrame:
        """Make regression predictions."""
        predictions = self.model.predict(X)

        return df.with_columns(pl.Series("predicted_value", predictions))


def predict_from_model(
    df: pl.DataFrame,
    model_path: str | Path,
    encoder_path: str | Path | None = None,
    task_type: TaskType | str = TaskType.CLASSIFICATION,
    feature_columns: list[str] | None = None,
    include_probabilities: bool = True,
) -> pl.DataFrame:
    """Convenience function to make predictions from saved model files.

    Args:
        df: Data to predict on
        model_path: Path to saved model
        encoder_path: Path to saved label encoder (for classification)
        task_type: Type of task
        feature_columns: Feature columns to use
        include_probabilities: Include class probabilities (classification only)

    Returns:
        DataFrame with predictions added
    """
    predictor = Predictor.from_files(
        model_path, encoder_path, task_type, feature_columns
    )
    return predictor.predict(df, include_probabilities=include_probabilities)
