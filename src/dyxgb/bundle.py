"""Model bundle format for single-file artifact storage.

A .dyxgb bundle is a ZIP archive containing:
- model.json: XGBoost serialized model
- encoder.joblib: Label encoder (classification only)
- pipeline.joblib: Transform pipeline (optional)
- metadata.json: Bundle metadata (version, task type, features, etc.)
"""

from __future__ import annotations

import io
import json
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from dyxgb import __version__
from dyxgb.model.trainer import TaskType

if TYPE_CHECKING:
    from dyxgb.transforms import TransformPipeline


BUNDLE_EXTENSION = ".dyxgb"
METADATA_FILENAME = "metadata.json"
MODEL_FILENAME = "model.json"
ENCODER_FILENAME = "encoder.joblib"
PIPELINE_FILENAME = "pipeline.joblib"


@dataclass
class BundleMetadata:
    """Metadata stored in the bundle."""

    dyxgb_version: str
    task_type: str
    feature_columns: list[str]
    target_column: str
    created_at: str
    has_encoder: bool = False
    has_pipeline: bool = False
    train_score: float | None = None
    val_score: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dyxgb_version": self.dyxgb_version,
            "task_type": self.task_type,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "created_at": self.created_at,
            "has_encoder": self.has_encoder,
            "has_pipeline": self.has_pipeline,
            "train_score": self.train_score,
            "val_score": self.val_score,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BundleMetadata":
        """Create from dictionary."""
        return cls(
            dyxgb_version=data["dyxgb_version"],
            task_type=data["task_type"],
            feature_columns=data["feature_columns"],
            target_column=data["target_column"],
            created_at=data["created_at"],
            has_encoder=data.get("has_encoder", False),
            has_pipeline=data.get("has_pipeline", False),
            train_score=data.get("train_score"),
            val_score=data.get("val_score"),
            extra=data.get("extra", {}),
        )


@dataclass
class Bundle:
    """Loaded model bundle with all artifacts."""

    model: XGBClassifier | XGBRegressor
    metadata: BundleMetadata
    label_encoder: LabelEncoder | None = None
    pipeline: Any | None = None  # TransformPipeline

    @property
    def task_type(self) -> TaskType:
        """Get task type from metadata."""
        return TaskType(self.metadata.task_type)

    @property
    def feature_columns(self) -> list[str]:
        """Get feature columns from metadata."""
        return self.metadata.feature_columns


def save_bundle(
    path: str | Path,
    model: XGBClassifier | XGBRegressor,
    task_type: TaskType | str,
    feature_columns: list[str],
    target_column: str,
    label_encoder: LabelEncoder | None = None,
    pipeline: Any | None = None,
    train_score: float | None = None,
    val_score: float | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """Save model and artifacts to a bundle file.

    Args:
        path: Output path for bundle file.
        model: Trained XGBoost model.
        task_type: Task type (classification or regression).
        feature_columns: List of feature column names.
        target_column: Target column name.
        label_encoder: Label encoder for classification (optional).
        pipeline: Transform pipeline (optional).
        train_score: Training score (optional).
        val_score: Validation score (optional).
        extra_metadata: Additional metadata to store (optional).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(task_type, TaskType):
        task_type_str = task_type.value
    else:
        task_type_str = task_type

    # Create metadata
    metadata = BundleMetadata(
        dyxgb_version=__version__,
        task_type=task_type_str,
        feature_columns=feature_columns,
        target_column=target_column,
        created_at=datetime.now(timezone.utc).isoformat(),
        has_encoder=label_encoder is not None,
        has_pipeline=pipeline is not None,
        train_score=train_score,
        val_score=val_score,
        extra=extra_metadata or {},
    )

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Save metadata
        metadata_json = json.dumps(metadata.to_dict(), indent=2)
        zf.writestr(METADATA_FILENAME, metadata_json)

        # Save model (XGBoost needs a file path)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            model.save_model(tmp_path)
            zf.write(tmp_path, MODEL_FILENAME)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # Save encoder if present
        if label_encoder is not None:
            encoder_bytes = io.BytesIO()
            joblib.dump(label_encoder, encoder_bytes)
            zf.writestr(ENCODER_FILENAME, encoder_bytes.getvalue())

        # Save pipeline if present
        if pipeline is not None:
            pipeline_bytes = io.BytesIO()
            joblib.dump(pipeline, pipeline_bytes)
            zf.writestr(PIPELINE_FILENAME, pipeline_bytes.getvalue())


def load_bundle(path: str | Path) -> Bundle:
    """Load model bundle from file.

    Args:
        path: Path to bundle file.

    Returns:
        Bundle containing model and all artifacts.

    Raises:
        FileNotFoundError: If bundle file doesn't exist.
        ValueError: If bundle is invalid or corrupted.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Bundle not found: {path}")

    try:
        with zipfile.ZipFile(path, "r") as zf:
            # Load metadata
            try:
                metadata_json = zf.read(METADATA_FILENAME).decode("utf-8")
                metadata = BundleMetadata.from_dict(json.loads(metadata_json))
            except KeyError:
                raise ValueError(f"Invalid bundle: missing {METADATA_FILENAME}")

            # Load model
            try:
                model_data = zf.read(MODEL_FILENAME)
            except KeyError:
                raise ValueError(f"Invalid bundle: missing {MODEL_FILENAME}")

            # Create appropriate model type
            task_type = TaskType(metadata.task_type)
            if task_type == TaskType.CLASSIFICATION:
                model = XGBClassifier()
            else:
                model = XGBRegressor()

            # XGBoost needs a file to load from
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                tmp.write(model_data)
                tmp_path = tmp.name
            try:
                model.load_model(tmp_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

            # Load encoder if present
            label_encoder = None
            if metadata.has_encoder:
                try:
                    encoder_data = zf.read(ENCODER_FILENAME)
                    label_encoder = joblib.load(io.BytesIO(encoder_data))
                except KeyError:
                    pass  # Encoder flag set but file missing - continue without it

            # Load pipeline if present
            pipeline = None
            if metadata.has_pipeline:
                try:
                    pipeline_data = zf.read(PIPELINE_FILENAME)
                    pipeline = joblib.load(io.BytesIO(pipeline_data))
                except KeyError:
                    pass  # Pipeline flag set but file missing - continue without it

            return Bundle(
                model=model,
                metadata=metadata,
                label_encoder=label_encoder,
                pipeline=pipeline,
            )

    except zipfile.BadZipFile:
        raise ValueError(f"Invalid bundle file (not a valid ZIP): {path}")


def is_bundle_path(path: str | Path) -> bool:
    """Check if path looks like a bundle file."""
    return str(path).lower().endswith(BUNDLE_EXTENSION)


def load_model_or_bundle(
    model_path: str | Path,
    encoder_path: str | Path | None = None,
    pipeline_path: str | Path | None = None,
    task_type: TaskType | str = TaskType.CLASSIFICATION,
) -> Bundle:
    """Load model from bundle or legacy separate files.

    This provides backward compatibility with the old multi-file format.

    Args:
        model_path: Path to bundle file or model.json.
        encoder_path: Path to encoder (legacy mode only).
        pipeline_path: Path to pipeline (legacy mode only).
        task_type: Task type (legacy mode only).

    Returns:
        Bundle with loaded artifacts.
    """
    model_path = Path(model_path)

    # Try bundle format first
    if is_bundle_path(model_path):
        return load_bundle(model_path)

    # Try loading as bundle anyway (user might have named it differently)
    if model_path.exists():
        try:
            with zipfile.ZipFile(model_path, "r") as zf:
                if METADATA_FILENAME in zf.namelist():
                    return load_bundle(model_path)
        except zipfile.BadZipFile:
            pass

    # Fall back to legacy format
    from dyxgb.model.trainer import load_model

    if isinstance(task_type, str):
        task_type = TaskType(task_type)

    model, label_encoder = load_model(model_path, encoder_path, task_type)

    # Load pipeline if specified
    pipeline = None
    if pipeline_path and Path(pipeline_path).exists():
        pipeline = joblib.load(pipeline_path)

    # Try to get feature names from model
    try:
        feature_columns = list(model.get_booster().feature_names or [])
    except (AttributeError, TypeError):
        feature_columns = []

    metadata = BundleMetadata(
        dyxgb_version=__version__,
        task_type=task_type.value,
        feature_columns=feature_columns,
        target_column="",  # Unknown in legacy format
        created_at=datetime.now(timezone.utc).isoformat(),
        has_encoder=label_encoder is not None,
        has_pipeline=pipeline is not None,
    )

    return Bundle(
        model=model,
        metadata=metadata,
        label_encoder=label_encoder,
        pipeline=pipeline,
    )
