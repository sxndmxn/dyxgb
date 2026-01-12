"""Hyperparameter tuning with Optuna."""

from dataclasses import dataclass
from typing import Any, Callable

import polars as pl
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from dyxgb.model.trainer import TaskType, HyperParameters


@dataclass
class TuningResult:
    """Result of hyperparameter tuning."""

    best_params: dict[str, Any]
    best_score: float
    n_trials: int
    study: Any  # optuna.Study


class OptunaOptimizer:
    """Hyperparameter optimization using Optuna."""

    def __init__(
        self,
        task_type: TaskType | str = TaskType.CLASSIFICATION,
        n_trials: int = 50,
        cv_folds: int = 5,
        metric: str | None = None,
        random_state: int = 42,
        n_jobs: int = -1,
        timeout: int | None = None,
    ):
        if isinstance(task_type, str):
            task_type = TaskType(task_type)

        self.task_type = task_type
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.timeout = timeout

        # Set default metric based on task type
        if metric is None:
            self.metric = (
                "f1_weighted"
                if task_type == TaskType.CLASSIFICATION
                else "neg_root_mean_squared_error"
            )
        else:
            self.metric = metric

    def optimize(
        self,
        df: pl.DataFrame,
        target_column: str,
        feature_columns: list[str] | None = None,
        param_space: dict[str, Any] | None = None,
    ) -> TuningResult:
        """Run hyperparameter optimization.

        Args:
            df: Training data
            target_column: Name of target column
            feature_columns: Feature column names (None = all except target)
            param_space: Custom parameter space. If None, uses defaults.

        Returns:
            TuningResult with best parameters and score
        """
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            raise ImportError(
                "Optuna is required for hyperparameter tuning. "
                "Install with: uv add optuna"
            )

        # Suppress optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Prepare data
        if feature_columns is None:
            feature_columns = [c for c in df.columns if c != target_column]

        df_pd = df.select(feature_columns + [target_column]).to_pandas()
        X = df_pd[feature_columns].values
        y = df_pd[target_column].values

        # Encode labels for classification
        label_encoder = None
        if self.task_type == TaskType.CLASSIFICATION:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

        # Create objective function
        objective = self._create_objective(X, y, label_encoder, param_space)

        # Run optimization
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
        )

        return TuningResult(
            best_params=study.best_params,
            best_score=study.best_value,
            n_trials=len(study.trials),
            study=study,
        )

    def _create_objective(
        self,
        X: Any,
        y: Any,
        label_encoder: LabelEncoder | None,
        param_space: dict[str, Any] | None,
    ) -> Callable:
        """Create Optuna objective function."""

        def objective(trial: Any) -> float:
            # Define parameter space
            params = self._get_trial_params(trial, param_space)

            # Create model
            if self.task_type == TaskType.CLASSIFICATION:
                if label_encoder is not None and len(label_encoder.classes_) > 2:
                    params["objective"] = "multi:softprob"
                else:
                    params["objective"] = "binary:logistic"
                model = XGBClassifier(
                    **params, n_jobs=self.n_jobs, random_state=self.random_state
                )
            else:
                params["objective"] = "reg:squarederror"
                model = XGBRegressor(
                    **params, n_jobs=self.n_jobs, random_state=self.random_state
                )

            # Cross-validation
            scores = cross_val_score(
                model, X, y, cv=self.cv_folds, scoring=self.metric, n_jobs=1
            )

            return scores.mean()

        return objective

    def _get_trial_params(
        self, trial: Any, param_space: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Get parameters for a single trial."""
        if param_space:
            return {
                name: self._sample_param(trial, name, config)
                for name, config in param_space.items()
            }

        # Default parameter space
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    def _sample_param(self, trial: Any, name: str, config: dict[str, Any]) -> Any:
        """Sample a parameter based on configuration."""
        param_type = config.get("type", "float")

        if param_type == "int":
            return trial.suggest_int(
                name, config["low"], config["high"], step=config.get("step", 1)
            )
        elif param_type == "float":
            return trial.suggest_float(
                name, config["low"], config["high"], log=config.get("log", False)
            )
        elif param_type == "categorical":
            return trial.suggest_categorical(name, config["choices"])
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")


def tune_hyperparameters(
    df: pl.DataFrame,
    target_column: str,
    feature_columns: list[str] | None = None,
    task_type: TaskType | str = TaskType.CLASSIFICATION,
    n_trials: int = 50,
    metric: str | None = None,
) -> HyperParameters:
    """Convenience function to tune hyperparameters and return as HyperParameters.

    Args:
        df: Training data
        target_column: Target column name
        feature_columns: Feature column names
        task_type: Classification or regression
        n_trials: Number of optimization trials
        metric: Scoring metric

    Returns:
        HyperParameters with optimized values
    """
    optimizer = OptunaOptimizer(
        task_type=task_type,
        n_trials=n_trials,
        metric=metric,
    )

    result = optimizer.optimize(df, target_column, feature_columns)

    # Convert best params to HyperParameters
    return HyperParameters(**result.best_params)
