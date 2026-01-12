"""Evaluation metrics for classification and regression models."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from dyxgb.model.trainer import TaskType


@dataclass
class ClassificationMetrics:
    """Classification evaluation metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None = None
    confusion_matrix: np.ndarray | None = None
    classification_report: str | None = None
    per_class_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }
        if self.roc_auc is not None:
            result["roc_auc"] = self.roc_auc
        if self.confusion_matrix is not None:
            result["confusion_matrix"] = self.confusion_matrix.tolist()
        if self.per_class_metrics:
            result["per_class"] = self.per_class_metrics
        return result


@dataclass
class RegressionMetrics:
    """Regression evaluation metrics."""

    mse: float
    rmse: float
    mae: float
    r2: float
    mape: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
        }
        if self.mape is not None:
            result["mape"] = self.mape
        return result


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    labels: list[str] | None = None,
    average: str = "weighted",
) -> ClassificationMetrics:
    """Evaluate classification model performance.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for ROC-AUC)
        labels: Class labels for confusion matrix
        average: Averaging strategy for multi-class metrics

    Returns:
        ClassificationMetrics with computed values
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    # ROC-AUC (requires probabilities)
    roc_auc = None
    if y_proba is not None:
        try:
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                # Binary classification - use probability of positive class
                roc_auc = roc_auc_score(y_true, y_proba[:, 1])
            elif y_proba.ndim == 2 and y_proba.shape[1] > 2:
                # Multi-class - use one-vs-rest
                roc_auc = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average=average
                )
        except ValueError:
            # ROC-AUC may fail for certain edge cases
            pass

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Classification report
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)

    return ClassificationMetrics(
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=roc_auc,
        confusion_matrix=cm,
        classification_report=report,
    )


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> RegressionMetrics:
    """Evaluate regression model performance.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        RegressionMetrics with computed values
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE (handle division by zero)
    mape = None
    nonzero_mask = y_true != 0
    if nonzero_mask.any():
        mape = (
            np.mean(
                np.abs(
                    (y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask]
                )
            )
            * 100
        )

    return RegressionMetrics(
        mse=mse,
        rmse=rmse,
        mae=mae,
        r2=r2,
        mape=mape,
    )


def print_metrics(
    metrics: ClassificationMetrics | RegressionMetrics,
    task_type: TaskType | None = None,
) -> None:
    """Print metrics in a formatted way.

    Args:
        metrics: Metrics object to print
        task_type: Optional task type for header
    """
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        _print_metrics_rich(console, metrics, task_type)
    except ImportError:
        _print_metrics_plain(metrics, task_type)


def _print_metrics_rich(
    console: Any,
    metrics: ClassificationMetrics | RegressionMetrics,
    task_type: TaskType | None,
) -> None:
    """Print metrics using rich formatting."""
    from rich.table import Table

    if isinstance(metrics, ClassificationMetrics):
        title = (
            "Classification Metrics"
            if task_type is None
            else f"{task_type.value.title()} Metrics"
        )
        table = Table(title=title)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Accuracy", f"{metrics.accuracy:.4f}")
        table.add_row("Precision", f"{metrics.precision:.4f}")
        table.add_row("Recall", f"{metrics.recall:.4f}")
        table.add_row("F1 Score", f"{metrics.f1:.4f}")
        if metrics.roc_auc is not None:
            table.add_row("ROC-AUC", f"{metrics.roc_auc:.4f}")

        console.print(table)

        if metrics.confusion_matrix is not None:
            console.print("\n[bold]Confusion Matrix:[/bold]")
            console.print(metrics.confusion_matrix)

    else:  # RegressionMetrics
        title = (
            "Regression Metrics"
            if task_type is None
            else f"{task_type.value.title()} Metrics"
        )
        table = Table(title=title)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("MSE", f"{metrics.mse:.4f}")
        table.add_row("RMSE", f"{metrics.rmse:.4f}")
        table.add_row("MAE", f"{metrics.mae:.4f}")
        table.add_row("R²", f"{metrics.r2:.4f}")
        if metrics.mape is not None:
            table.add_row("MAPE", f"{metrics.mape:.2f}%")

        console.print(table)


def _print_metrics_plain(
    metrics: ClassificationMetrics | RegressionMetrics,
    task_type: TaskType | None,
) -> None:
    """Print metrics in plain text."""
    if isinstance(metrics, ClassificationMetrics):
        print("\n=== Classification Metrics ===")
        print(f"Accuracy:  {metrics.accuracy:.4f}")
        print(f"Precision: {metrics.precision:.4f}")
        print(f"Recall:    {metrics.recall:.4f}")
        print(f"F1 Score:  {metrics.f1:.4f}")
        if metrics.roc_auc is not None:
            print(f"ROC-AUC:   {metrics.roc_auc:.4f}")
        if metrics.confusion_matrix is not None:
            print("\nConfusion Matrix:")
            print(metrics.confusion_matrix)
    else:
        print("\n=== Regression Metrics ===")
        print(f"MSE:  {metrics.mse:.4f}")
        print(f"RMSE: {metrics.rmse:.4f}")
        print(f"MAE:  {metrics.mae:.4f}")
        print(f"R²:   {metrics.r2:.4f}")
        if metrics.mape is not None:
            print(f"MAPE: {metrics.mape:.2f}%")
