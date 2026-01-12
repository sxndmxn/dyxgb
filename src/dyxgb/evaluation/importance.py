"""Feature importance extraction and export."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from xgboost import XGBClassifier, XGBRegressor


@dataclass
class FeatureImportance:
    """Feature importance data."""

    feature_names: list[str]
    importance_values: list[float]
    importance_type: str

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary mapping feature names to importance."""
        return dict(zip(self.feature_names, self.importance_values))

    def to_dataframe(self) -> pl.DataFrame:
        """Convert to Polars DataFrame sorted by importance."""
        return pl.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.importance_values,
                "importance_type": [self.importance_type] * len(self.feature_names),
            }
        ).sort("importance", descending=True)

    def top_n(self, n: int = 10) -> "FeatureImportance":
        """Get top N most important features."""
        sorted_pairs = sorted(
            zip(self.feature_names, self.importance_values),
            key=lambda x: x[1],
            reverse=True,
        )[:n]
        names, values = zip(*sorted_pairs) if sorted_pairs else ([], [])
        return FeatureImportance(
            feature_names=list(names),
            importance_values=list(values),
            importance_type=self.importance_type,
        )


def get_feature_importance(
    model: XGBClassifier | XGBRegressor,
    importance_type: str = "weight",
    feature_names: list[str] | None = None,
) -> FeatureImportance:
    """Extract feature importance from trained model.

    Args:
        model: Trained XGBoost model
        importance_type: Type of importance to extract:
            - "weight": Number of times feature is used in splits
            - "gain": Average gain of splits using the feature
            - "cover": Average coverage of splits using the feature
            - "total_gain": Total gain of splits using the feature
            - "total_cover": Total coverage of splits using the feature
        feature_names: Optional list of feature names (uses model's if not provided)

    Returns:
        FeatureImportance object
    """
    booster = model.get_booster()

    # Get importance scores
    importance_dict = booster.get_score(importance_type=importance_type)

    # Get feature names from model if not provided
    if feature_names is None:
        try:
            feature_names = booster.feature_names
            if feature_names is None:
                # Fall back to generic names
                feature_names = [f"f{i}" for i in range(model.n_features_in_)]
        except AttributeError:
            feature_names = [f"f{i}" for i in range(model.n_features_in_)]

    # Map importance scores to feature names
    # XGBoost uses f0, f1, etc. internally if no feature names set
    names = []
    values = []

    for fname in feature_names:
        # Try feature name directly
        if fname in importance_dict:
            names.append(fname)
            values.append(importance_dict[fname])
        else:
            # Try generic format (f0, f1, etc.)
            idx = feature_names.index(fname)
            generic_name = f"f{idx}"
            if generic_name in importance_dict:
                names.append(fname)
                values.append(importance_dict[generic_name])
            else:
                # Feature has zero importance
                names.append(fname)
                values.append(0.0)

    return FeatureImportance(
        feature_names=names,
        importance_values=values,
        importance_type=importance_type,
    )


def export_importance(
    importance: FeatureImportance,
    path: str | Path,
    format: str = "auto",
) -> None:
    """Export feature importance to file.

    Args:
        importance: FeatureImportance object to export
        path: Output path
        format: Output format ("json", "csv", "parquet", or "auto" to detect from extension)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Auto-detect format from extension
    if format == "auto":
        suffix = path.suffix.lower()
        format_map = {
            ".json": "json",
            ".csv": "csv",
            ".parquet": "parquet",
            ".pq": "parquet",
        }
        format = format_map.get(suffix, "json")

    if format == "json":
        data = {
            "importance_type": importance.importance_type,
            "features": importance.to_dict(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    elif format == "csv":
        importance.to_dataframe().write_csv(path)

    elif format == "parquet":
        importance.to_dataframe().write_parquet(path)

    else:
        raise ValueError(f"Unsupported export format: {format}")


def print_importance(
    importance: FeatureImportance,
    top_n: int = 20,
) -> None:
    """Print feature importance in formatted table.

    Args:
        importance: FeatureImportance object
        top_n: Number of top features to show
    """
    top = importance.top_n(top_n)

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title=f"Feature Importance ({importance.importance_type})")
        table.add_column("Rank", style="dim")
        table.add_column("Feature", style="cyan")
        table.add_column("Importance", style="green", justify="right")

        for i, (name, value) in enumerate(
            zip(top.feature_names, top.importance_values), 1
        ):
            table.add_row(str(i), name, f"{value:.4f}")

        console.print(table)

    except ImportError:
        # Fallback to plain text
        print(f"\n=== Feature Importance ({importance.importance_type}) ===")
        print(f"{'Rank':<6}{'Feature':<30}{'Importance':>12}")
        print("-" * 48)
        for i, (name, value) in enumerate(
            zip(top.feature_names, top.importance_values), 1
        ):
            print(f"{i:<6}{name:<30}{value:>12.4f}")
