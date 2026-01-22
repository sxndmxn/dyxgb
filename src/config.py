"""Configuration file loading and validation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from dyxgb.data.base import DataSource
from dyxgb.model.trainer import HyperParameters, TaskType


@dataclass
class DataSourceConfig:
    """Configuration for a data source."""

    type: str  # file, sqlite, duckdb, postgres
    path: str | None = None
    uri: str | None = None
    query: str | None = None
    table: str | None = None

    def __post_init__(self) -> None:
        if self.type == "file":
            if not self.path:
                raise ValueError("File data source requires 'path'")
        else:
            if not self.uri:
                raise ValueError(f"{self.type} data source requires 'uri'")
            if not self.query and not self.table:
                raise ValueError(f"{self.type} data source requires 'query' or 'table'")

    @property
    def source_type(self) -> DataSource:
        """Convert string type to DataSource enum."""
        return DataSource(self.type)


@dataclass
class ModelConfig:
    """Model training configuration."""

    task: str = "classification"
    target: str = ""
    features: list[str] = field(default_factory=list)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    validation_split: float = 0.2
    early_stopping_rounds: int | None = 50

    @property
    def task_type(self) -> TaskType:
        """Convert string task to TaskType enum."""
        return TaskType(self.task)

    def get_hyperparameters(self) -> HyperParameters:
        """Convert hyperparameters dict to HyperParameters object."""
        if self.hyperparameters:
            return HyperParameters(**self.hyperparameters)
        return HyperParameters()


@dataclass
class TuningConfig:
    """Hyperparameter tuning configuration."""

    enabled: bool = False
    n_trials: int = 50
    metric: str | None = None
    timeout: int | None = None
    cv_folds: int = 5


@dataclass
class TransformsConfig:
    """Data transformation pipeline configuration."""

    rename: dict[str, str] = field(default_factory=dict)
    cast: dict[str, str] = field(default_factory=dict)
    missing: dict[str, Any] = field(default_factory=dict)
    features: list[dict[str, str]] = field(default_factory=list)
    encode: dict[str, Any] = field(default_factory=dict)
    scale: dict[str, Any] = field(default_factory=dict)

    def to_pipeline_config(self) -> dict[str, Any]:
        """Convert to format expected by TransformPipeline.from_config()."""
        config: dict[str, Any] = {}

        if self.rename:
            config["rename"] = self.rename
        if self.cast:
            config["cast"] = self.cast
        if self.missing:
            config["missing"] = self.missing
        if self.features:
            config["features"] = self.features
        if self.encode:
            config["encode"] = self.encode
        if self.scale:
            config["scale"] = self.scale

        return config

    def __bool__(self) -> bool:
        """Return True if any transforms are configured."""
        return bool(
            self.rename or self.cast or self.missing or self.features or self.encode or self.scale
        )


@dataclass
class OutputConfig:
    """Output paths configuration."""

    model_path: str = "model.json"
    encoder_path: str = "label_encoder.joblib"
    pipeline_path: str = "pipeline.joblib"
    predictions_path: str = "predictions.parquet"
    importance_path: str = "feature_importance.json"
    metrics_path: str | None = None


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    show_metrics: bool = True
    metrics: list[str] = field(default_factory=lambda: ["accuracy", "f1", "precision", "recall"])
    show_importance: bool = True
    importance_type: str = "gain"
    top_n_features: int = 20


@dataclass
class Config:
    """Main configuration container."""

    data: dict[str, DataSourceConfig] = field(default_factory=dict)
    transforms: TransformsConfig = field(default_factory=TransformsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        config = cls()

        # Parse data sources
        if "data" in data:
            for name, source_data in data["data"].items():
                config.data[name] = DataSourceConfig(**source_data)

        # Parse transforms config
        if "transforms" in data:
            transforms_data = data["transforms"]
            config.transforms = TransformsConfig(
                rename=transforms_data.get("rename", {}),
                cast=transforms_data.get("cast", {}),
                missing=transforms_data.get("missing", {}),
                features=transforms_data.get("features", []),
                encode=transforms_data.get("encode", {}),
                scale=transforms_data.get("scale", {}),
            )

        # Parse model config
        if "model" in data:
            config.model = ModelConfig(**data["model"])

        # Parse tuning config
        if "tuning" in data:
            config.tuning = TuningConfig(**data["tuning"])

        # Parse output config
        if "output" in data:
            config.output = OutputConfig(**data["output"])

        # Parse evaluation config
        if "evaluation" in data:
            config.evaluation = EvaluationConfig(**data["evaluation"])

        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data or {})

    @classmethod
    def from_toml(cls, path: str | Path) -> "Config":
        """Load configuration from TOML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore

        with open(path, "rb") as f:
            data = tomllib.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        """Load configuration from file (auto-detect format)."""
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix in (".yaml", ".yml"):
            return cls.from_yaml(path)
        elif suffix == ".toml":
            return cls.from_toml(path)
        else:
            raise ValueError(f"Unsupported config format: {suffix}. Use .yaml or .toml")

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        result: dict[str, Any] = {
            "data": {
                name: {
                    "type": src.type,
                    "path": src.path,
                    "uri": src.uri,
                    "query": src.query,
                    "table": src.table,
                }
                for name, src in self.data.items()
            },
            "model": {
                "task": self.model.task,
                "target": self.model.target,
                "features": self.model.features,
                "hyperparameters": self.model.hyperparameters,
                "validation_split": self.model.validation_split,
                "early_stopping_rounds": self.model.early_stopping_rounds,
            },
            "tuning": {
                "enabled": self.tuning.enabled,
                "n_trials": self.tuning.n_trials,
                "metric": self.tuning.metric,
                "timeout": self.tuning.timeout,
                "cv_folds": self.tuning.cv_folds,
            },
            "output": {
                "model_path": self.output.model_path,
                "encoder_path": self.output.encoder_path,
                "pipeline_path": self.output.pipeline_path,
                "predictions_path": self.output.predictions_path,
                "importance_path": self.output.importance_path,
                "metrics_path": self.output.metrics_path,
            },
            "evaluation": {
                "show_metrics": self.evaluation.show_metrics,
                "metrics": self.evaluation.metrics,
                "show_importance": self.evaluation.show_importance,
                "importance_type": self.evaluation.importance_type,
                "top_n_features": self.evaluation.top_n_features,
            },
        }

        # Only include transforms if configured
        if self.transforms:
            result["transforms"] = {
                "rename": self.transforms.rename,
                "cast": self.transforms.cast,
                "missing": self.transforms.missing,
                "features": self.transforms.features,
                "encode": self.transforms.encode,
                "scale": self.transforms.scale,
            }

        return result

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def load_config(path: str | Path) -> Config:
    """Convenience function to load config from file."""
    return Config.from_file(path)
