"""Dynamic XGBoost - A flexible CLI tool for XGBoost training and prediction.

Unix Philosophy:
- stdout: data only (CSV, JSONL, JSON)
- stderr: logs, progress, human-readable output
- Exit codes: 0 (success), 1 (runtime error), 2 (usage error)
"""

__version__ = "0.3.0"

# Public API exports
from dyxgb.api import (
    EvaluateResult,
    ImportanceResult,
    PredictResult,
    evaluate_df,
    get_importance,
    predict_df,
    train_model,
    tune_model,
)
from dyxgb.bundle import (
    Bundle,
    BundleMetadata,
    load_bundle,
    load_model_or_bundle,
    save_bundle,
)
from dyxgb.io import (
    EXIT_RUNTIME_ERROR,
    EXIT_SUCCESS,
    EXIT_USAGE_ERROR,
    InputFormat,
    OutputFormat,
    read_table,
    write_json,
    write_table,
)

__all__ = [
    # Version
    "__version__",
    # API
    "train_model",
    "predict_df",
    "evaluate_df",
    "get_importance",
    "tune_model",
    "PredictResult",
    "EvaluateResult",
    "ImportanceResult",
    # Bundle
    "Bundle",
    "BundleMetadata",
    "load_bundle",
    "save_bundle",
    "load_model_or_bundle",
    # I/O
    "read_table",
    "write_table",
    "write_json",
    "InputFormat",
    "OutputFormat",
    "EXIT_SUCCESS",
    "EXIT_RUNTIME_ERROR",
    "EXIT_USAGE_ERROR",
]
