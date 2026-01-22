"""XGBoost model training for classification and regression."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import joblib
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor


class TaskType(str, Enum):
    """Machine learning task type."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


@dataclass
class HyperParameters:
    """XGBoost hyperparameters."""

    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 1
    gamma: float = 0.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    random_state: int = 42
    n_jobs: int = -1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for XGBoost."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }


@dataclass
class TrainResult:
    """Result of model training."""

    model: XGBClassifier | XGBRegressor
    label_encoder: LabelEncoder | None
    task_type: TaskType
    feature_columns: list[str]
    target_column: str
    train_score: float | None = None
    val_score: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


class Trainer:
    """Train XGBoost models for classification or regression."""

    def __init__(
        self,
        task_type: TaskType | str = TaskType.CLASSIFICATION,
        hyperparameters: HyperParameters | dict[str, Any] | None = None,
        validation_split: float = 0.2,
        early_stopping_rounds: int | None = 50,
    ):
        if isinstance(task_type, str):
            task_type = TaskType(task_type)
        self.task_type = task_type

        if hyperparameters is None:
            self.hyperparameters = HyperParameters()
        elif isinstance(hyperparameters, dict):
            self.hyperparameters = HyperParameters(**hyperparameters)
        else:
            self.hyperparameters = hyperparameters

        self.validation_split = validation_split
        self.early_stopping_rounds = early_stopping_rounds

    def train(
        self,
        df: pl.DataFrame,
        target_column: str,
        feature_columns: list[str] | None = None,
    ) -> TrainResult:
        """Train the model on the provided data.

        Args:
            df: Training data as Polars DataFrame
            target_column: Name of the target column
            feature_columns: List of feature column names. If None, uses all columns
                           except target.

        Returns:
            TrainResult containing the trained model and metadata
        """
        # Determine feature columns
        if feature_columns is None:
            feature_columns = [c for c in df.columns if c != target_column]

        # Validate columns exist
        missing = set(feature_columns + [target_column]) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in data: {missing}")

        # Convert to pandas for sklearn/xgboost
        df_pd = df.select(feature_columns + [target_column]).to_pandas()
        X = df_pd[feature_columns]
        y = df_pd[target_column]

        # Handle label encoding for classification
        label_encoder = None
        if self.task_type == TaskType.CLASSIFICATION:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.validation_split,
            random_state=self.hyperparameters.random_state,
        )

        # Create and train model
        model = self._create_model(label_encoder)

        # Fit with early stopping if validation set available
        eval_set = [(X_val, y_val)] if self.early_stopping_rounds else None

        fit_params: dict[str, Any] = {}
        if eval_set and self.early_stopping_rounds:
            fit_params["eval_set"] = eval_set
            fit_params["verbose"] = False

        model.fit(X_train, y_train, **fit_params)

        # Calculate scores
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)

        return TrainResult(
            model=model,
            label_encoder=label_encoder,
            task_type=self.task_type,
            feature_columns=feature_columns,
            target_column=target_column,
            train_score=train_score,
            val_score=val_score,
        )

    def _create_model(
        self, label_encoder: LabelEncoder | None = None
    ) -> XGBClassifier | XGBRegressor:
        """Create the appropriate XGBoost model."""
        params = self.hyperparameters.to_dict()

        if self.task_type == TaskType.CLASSIFICATION:
            # Determine objective based on number of classes
            if label_encoder is not None and len(label_encoder.classes_) > 2:
                params["objective"] = "multi:softprob"
            else:
                params["objective"] = "binary:logistic"

            if self.early_stopping_rounds:
                params["early_stopping_rounds"] = self.early_stopping_rounds

            return XGBClassifier(**params)
        else:
            params["objective"] = "reg:squarederror"

            if self.early_stopping_rounds:
                params["early_stopping_rounds"] = self.early_stopping_rounds

            return XGBRegressor(**params)


def save_model(
    result: TrainResult,
    model_path: str | Path,
    encoder_path: str | Path | None = None,
) -> None:
    """Save trained model and label encoder to disk.

    Args:
        result: TrainResult from training
        model_path: Path to save model (JSON format)
        encoder_path: Path to save label encoder (joblib format).
                     Only needed for classification.
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    result.model.save_model(str(model_path))

    if result.label_encoder is not None and encoder_path:
        encoder_path = Path(encoder_path)
        encoder_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(result.label_encoder, encoder_path)


def load_model(
    model_path: str | Path,
    encoder_path: str | Path | None = None,
    task_type: TaskType | str = TaskType.CLASSIFICATION,
) -> tuple[XGBClassifier | XGBRegressor, LabelEncoder | None]:
    """Load model and label encoder from disk.

    Args:
        model_path: Path to saved model
        encoder_path: Path to saved label encoder
        task_type: Type of task (classification or regression)

    Returns:
        Tuple of (model, label_encoder). label_encoder is None for regression.
    """
    if isinstance(task_type, str):
        task_type = TaskType(task_type)

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if task_type == TaskType.CLASSIFICATION:
        model = XGBClassifier()
    else:
        model = XGBRegressor()

    model.load_model(str(model_path))

    label_encoder = None
    if encoder_path:
        encoder_path = Path(encoder_path)
        if encoder_path.exists():
            label_encoder = joblib.load(encoder_path)

    return model, label_encoder
