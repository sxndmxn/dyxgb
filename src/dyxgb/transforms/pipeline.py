"""Transform pipeline orchestrator."""

from pathlib import Path
from typing import Any

import joblib
import polars as pl

from dyxgb.transforms.base import BaseTransform
from dyxgb.transforms.rename import RenameTransform
from dyxgb.transforms.cast import CastTransform
from dyxgb.transforms.missing import MissingTransform
from dyxgb.transforms.features import FeatureTransform
from dyxgb.transforms.encode import EncodeTransform
from dyxgb.transforms.scale import ScaleTransform


class TransformPipeline:
    """Orchestrates data transforms in a specific order.

    Transform order (fixed):
        1. rename - Map column names to canonical names
        2. cast - Convert column types
        3. missing - Handle missing values
        4. features - Create derived features
        5. encode - Encode categorical columns
        6. scale - Scale numeric columns

    Usage:
        # Create from config dict
        pipeline = TransformPipeline.from_config(config_dict)

        # Fit on training data (learns parameters for stateful transforms)
        pipeline.fit(train_df, target_column="label")

        # Transform training data
        train_transformed = pipeline.transform(train_df)

        # Save fitted pipeline
        pipeline.save("pipeline.joblib")

        # Later, load and apply to prediction data
        pipeline = TransformPipeline.load("pipeline.joblib")
        pred_transformed = pipeline.transform(pred_df)
    """

    # Order in which transforms are applied
    TRANSFORM_ORDER = ["rename", "cast", "missing", "features", "encode", "scale"]

    def __init__(self) -> None:
        """Initialize empty pipeline."""
        self.transforms: dict[str, BaseTransform] = {}
        self.is_fitted = False
        self._target_column: str | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TransformPipeline":
        """Create pipeline from config dictionary.

        Args:
            config: Transform configuration dict with keys like 'rename', 'cast', etc.

        Returns:
            Configured TransformPipeline
        """
        pipeline = cls()

        # Rename transform
        if "rename" in config and config["rename"]:
            pipeline.transforms["rename"] = RenameTransform(mapping=config["rename"])

        # Cast transform
        if "cast" in config and config["cast"]:
            pipeline.transforms["cast"] = CastTransform(types=config["cast"])

        # Missing value transform
        if "missing" in config and config["missing"]:
            missing_config = config["missing"]
            pipeline.transforms["missing"] = MissingTransform(
                strategy=missing_config.get("strategy", "median"),
                columns=missing_config.get("columns"),
                constant_value=missing_config.get("constant_value"),
            )

        # Feature engineering transform
        if "features" in config and config["features"]:
            pipeline.transforms["features"] = FeatureTransform(features=config["features"])

        # Encode transform
        if "encode" in config and config["encode"]:
            encode_config = config["encode"]
            columns = encode_config.get("columns", [])
            if columns:
                pipeline.transforms["encode"] = EncodeTransform(
                    columns=columns,
                    unknown_value=encode_config.get("unknown_value", -1),
                )

        # Scale transform
        if "scale" in config and config["scale"]:
            scale_config = config["scale"]
            pipeline.transforms["scale"] = ScaleTransform(
                method=scale_config.get("method", "standard"),
                columns=scale_config.get("columns"),
            )

        return pipeline

    def fit(self, df: pl.DataFrame, target_column: str | None = None) -> "TransformPipeline":
        """Fit all transforms on training data.

        Transforms are fitted in order, with each transform receiving the
        output of the previous one. This ensures that stateful transforms
        (like ScaleTransform) learn from properly preprocessed data.

        Args:
            df: Training DataFrame
            target_column: Target column name (excluded from some transforms)

        Returns:
            self
        """
        self._target_column = target_column
        current_df = df

        for name in self.TRANSFORM_ORDER:
            if name not in self.transforms:
                continue

            transform = self.transforms[name]

            # Fit the transform
            transform.fit(current_df, target_column=target_column)

            # Apply transform to get data for next step
            current_df = transform.transform(current_df)

        self.is_fitted = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply all transforms to data.

        Args:
            df: DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted and self._has_stateful_transforms():
            raise RuntimeError(
                "Pipeline has stateful transforms that must be fitted first. "
                "Call pipeline.fit(train_df) before transform()."
            )

        current_df = df

        for name in self.TRANSFORM_ORDER:
            if name not in self.transforms:
                continue

            transform = self.transforms[name]
            current_df = transform.transform(current_df)

        return current_df

    def fit_transform(self, df: pl.DataFrame, target_column: str | None = None) -> pl.DataFrame:
        """Fit and transform in one step.

        Args:
            df: Training DataFrame
            target_column: Target column name

        Returns:
            Transformed DataFrame
        """
        return self.fit(df, target_column).transform(df)

    def _has_stateful_transforms(self) -> bool:
        """Check if pipeline has any stateful transforms."""
        stateful = {"missing", "encode", "scale"}
        return bool(stateful & set(self.transforms.keys()))

    def save(self, path: str | Path) -> None:
        """Save fitted pipeline to disk.

        Args:
            path: Path to save pipeline (typically .joblib)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize pipeline state
        state = {
            "is_fitted": self.is_fitted,
            "target_column": self._target_column,
            "transforms": {},
        }

        for name, transform in self.transforms.items():
            state["transforms"][name] = {
                "class": transform.__class__.__name__,
                "params": transform.get_params(),
            }

        joblib.dump(state, path)

    @classmethod
    def load(cls, path: str | Path) -> "TransformPipeline":
        """Load fitted pipeline from disk.

        Args:
            path: Path to saved pipeline

        Returns:
            Loaded TransformPipeline
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {path}")

        state = joblib.load(path)

        pipeline = cls()
        pipeline.is_fitted = state["is_fitted"]
        pipeline._target_column = state["target_column"]

        # Reconstruct transforms
        transform_classes = {
            "RenameTransform": RenameTransform,
            "CastTransform": CastTransform,
            "MissingTransform": MissingTransform,
            "FeatureTransform": FeatureTransform,
            "EncodeTransform": EncodeTransform,
            "ScaleTransform": ScaleTransform,
        }

        for name, transform_state in state["transforms"].items():
            cls_name = transform_state["class"]
            params = transform_state["params"]

            if cls_name not in transform_classes:
                raise ValueError(f"Unknown transform class: {cls_name}")

            # Create transform instance and restore params
            transform_cls = transform_classes[cls_name]

            # Handle different constructors
            if cls_name == "RenameTransform":
                transform = RenameTransform(mapping=params.get("mapping", {}))
            elif cls_name == "CastTransform":
                transform = CastTransform(types=params.get("types", {}))
            elif cls_name == "MissingTransform":
                transform = MissingTransform(
                    strategy=params.get("strategy", "median"),
                    columns=params.get("columns"),
                    constant_value=params.get("constant_value"),
                )
                transform.set_params(params)
            elif cls_name == "FeatureTransform":
                transform = FeatureTransform(features=params.get("features", []))
            elif cls_name == "EncodeTransform":
                transform = EncodeTransform(
                    columns=params.get("columns", []),
                    unknown_value=params.get("unknown_value", -1),
                )
                transform.set_params(params)
            elif cls_name == "ScaleTransform":
                transform = ScaleTransform(
                    method=params.get("method", "standard"),
                    columns=params.get("columns"),
                )
                transform.set_params(params)
            else:
                continue

            pipeline.transforms[name] = transform

        return pipeline

    def get_transform(self, name: str) -> BaseTransform | None:
        """Get a specific transform by name.

        Args:
            name: Transform name (e.g., 'encode', 'scale')

        Returns:
            Transform instance or None if not configured
        """
        return self.transforms.get(name)

    def __repr__(self) -> str:
        transform_names = list(self.transforms.keys())
        return f"TransformPipeline({transform_names}, fitted={self.is_fitted})"

    def __len__(self) -> int:
        return len(self.transforms)

    def __bool__(self) -> bool:
        return len(self.transforms) > 0
