"""Unix-style I/O adapters for stdin/stdout/file operations.

This module provides standardized read/write helpers for Unix-like behavior:
- stdin/stdout support for pipe-friendly workflows
- Format auto-detection from file extensions
- Clear separation of data (stdout) and logs (stderr)
"""

from __future__ import annotations

import json
import sys
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, TextIO, BinaryIO

import polars as pl

if TYPE_CHECKING:
    from typing import Any


class InputFormat(str, Enum):
    """Supported input formats."""

    CSV = "csv"
    JSONL = "jsonl"
    PARQUET = "parquet"
    JSON = "json"


class OutputFormat(str, Enum):
    """Supported output formats."""

    CSV = "csv"
    JSONL = "jsonl"
    PARQUET = "parquet"
    JSON = "json"


# Exit codes following Unix convention
EXIT_SUCCESS = 0
EXIT_RUNTIME_ERROR = 1
EXIT_USAGE_ERROR = 2


class IOError(Exception):
    """Base exception for I/O errors."""

    def __init__(self, message: str, exit_code: int = EXIT_RUNTIME_ERROR):
        super().__init__(message)
        self.exit_code = exit_code


class UsageError(IOError):
    """Usage/configuration error (exit code 2)."""

    def __init__(self, message: str):
        super().__init__(message, EXIT_USAGE_ERROR)


class RuntimeIOError(IOError):
    """Runtime I/O error (exit code 1)."""

    def __init__(self, message: str):
        super().__init__(message, EXIT_RUNTIME_ERROR)


def _detect_input_format(path: str | Path) -> InputFormat:
    """Detect input format from file extension."""
    suffix = Path(path).suffix.lower()
    format_map = {
        ".csv": InputFormat.CSV,
        ".jsonl": InputFormat.JSONL,
        ".ndjson": InputFormat.JSONL,
        ".parquet": InputFormat.PARQUET,
        ".pq": InputFormat.PARQUET,
        ".json": InputFormat.JSON,
    }
    fmt = format_map.get(suffix)
    if fmt is None:
        supported = ", ".join(format_map.keys())
        raise UsageError(f"Unsupported file format: {suffix}. Supported: {supported}")
    return fmt


def _detect_output_format(path: str | Path) -> OutputFormat:
    """Detect output format from file extension."""
    suffix = Path(path).suffix.lower()
    format_map = {
        ".csv": OutputFormat.CSV,
        ".jsonl": OutputFormat.JSONL,
        ".ndjson": OutputFormat.JSONL,
        ".parquet": OutputFormat.PARQUET,
        ".pq": OutputFormat.PARQUET,
        ".json": OutputFormat.JSON,
    }
    fmt = format_map.get(suffix)
    if fmt is None:
        supported = ", ".join(format_map.keys())
        raise UsageError(f"Unsupported output format: {suffix}. Supported: {supported}")
    return fmt


def read_table(
    source: str,
    *,
    input_format: str | InputFormat | None = None,
    query: str | None = None,
    table: str | None = None,
) -> pl.DataFrame:
    """Read tabular data from stdin, file, or database.

    Args:
        source: Data source. Use "-" for stdin, file path, or database URI.
        input_format: Format hint (required for stdin). One of: csv, jsonl.
        query: SQL query for database sources.
        table: Table name for database sources.

    Returns:
        Polars DataFrame with loaded data.

    Raises:
        UsageError: For invalid format/source combinations.
        RuntimeIOError: For I/O failures.
    """
    # Handle stdin
    if source == "-":
        return _read_stdin(input_format)

    # Handle database URIs
    if source.startswith(("sqlite:", "duckdb:", "postgres:", "postgresql:")):
        return _read_database(source, query=query, table=table)

    # Handle file paths
    return _read_file(source, input_format)


def _read_stdin(input_format: str | InputFormat | None) -> pl.DataFrame:
    """Read from stdin."""
    # Determine format
    if input_format is None:
        fmt = InputFormat.CSV  # Default to CSV
    elif isinstance(input_format, str):
        try:
            fmt = InputFormat(input_format.lower())
        except ValueError:
            raise UsageError(
                f"Invalid input format: {input_format}. "
                f"Stdin only supports: csv, jsonl"
            )
    else:
        fmt = input_format

    # Validate stdin-compatible formats
    if fmt not in (InputFormat.CSV, InputFormat.JSONL):
        raise UsageError(
            f"Format '{fmt.value}' is not supported for stdin. "
            f"Use --input-format csv or --input-format jsonl"
        )

    try:
        data = sys.stdin.buffer.read()
        if fmt == InputFormat.CSV:
            return pl.read_csv(data)
        else:  # JSONL
            return pl.read_ndjson(data)
    except Exception as e:
        raise RuntimeIOError(f"Failed to read from stdin: {e}")


def _read_file(
    path: str | Path,
    input_format: str | InputFormat | None,
) -> pl.DataFrame:
    """Read from file."""
    path = Path(path)

    if not path.exists():
        raise RuntimeIOError(f"File not found: {path}")

    # Determine format
    if input_format is None:
        fmt = _detect_input_format(path)
    elif isinstance(input_format, str):
        try:
            fmt = InputFormat(input_format.lower())
        except ValueError:
            raise UsageError(f"Invalid input format: {input_format}")
    else:
        fmt = input_format

    try:
        if fmt == InputFormat.CSV:
            return pl.read_csv(path)
        elif fmt == InputFormat.JSONL:
            return pl.read_ndjson(path)
        elif fmt == InputFormat.PARQUET:
            return pl.read_parquet(path)
        elif fmt == InputFormat.JSON:
            return pl.read_json(path)
        else:
            raise UsageError(f"Unsupported format: {fmt}")
    except Exception as e:
        raise RuntimeIOError(f"Failed to read {path}: {e}")


def _read_database(
    uri: str,
    query: str | None = None,
    table: str | None = None,
) -> pl.DataFrame:
    """Read from database."""
    from dyxgb.data.database import load_from_uri

    try:
        return load_from_uri(uri, query=query, table=table)
    except Exception as e:
        raise RuntimeIOError(f"Database read failed: {e}")


def write_table(
    df: pl.DataFrame,
    dest: str,
    *,
    output_format: str | OutputFormat | None = None,
    include_columns: list[str] | None = None,
) -> None:
    """Write tabular data to stdout or file.

    Args:
        df: DataFrame to write.
        dest: Destination. Use "-" for stdout, or file path.
        output_format: Format hint. Defaults to csv for stdout, auto-detect for files.
        include_columns: Optional list of columns to include in output.

    Raises:
        UsageError: For invalid format/destination combinations.
        RuntimeIOError: For I/O failures.
    """
    # Filter columns if specified
    if include_columns:
        missing = set(include_columns) - set(df.columns)
        if missing:
            raise UsageError(f"Missing columns for output: {missing}")
        df = df.select(include_columns)

    # Handle stdout
    if dest == "-":
        _write_stdout(df, output_format)
        return

    # Handle file paths
    _write_file(df, dest, output_format)


def _write_stdout(
    df: pl.DataFrame,
    output_format: str | OutputFormat | None,
) -> None:
    """Write to stdout."""
    # Determine format
    if output_format is None:
        fmt = OutputFormat.CSV  # Default to CSV for stdout
    elif isinstance(output_format, str):
        try:
            fmt = OutputFormat(output_format.lower())
        except ValueError:
            raise UsageError(f"Invalid output format: {output_format}")
    else:
        fmt = output_format

    # Validate stdout-compatible formats
    if fmt == OutputFormat.PARQUET:
        raise UsageError(
            "Parquet format is not supported for stdout. "
            "Use --output-format csv or --output-format jsonl"
        )

    try:
        if fmt == OutputFormat.CSV:
            csv_str = df.write_csv()
            sys.stdout.write(csv_str)
        elif fmt == OutputFormat.JSONL:
            # Write JSONL line by line
            for row in df.iter_rows(named=True):
                sys.stdout.write(json.dumps(row) + "\n")
        elif fmt == OutputFormat.JSON:
            # Write as JSON array
            rows = df.to_dicts()
            sys.stdout.write(json.dumps(rows, indent=2) + "\n")
        sys.stdout.flush()
    except Exception as e:
        raise RuntimeIOError(f"Failed to write to stdout: {e}")


def _write_file(
    df: pl.DataFrame,
    path: str | Path,
    output_format: str | OutputFormat | None,
) -> None:
    """Write to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Determine format
    if output_format is None:
        fmt = _detect_output_format(path)
    elif isinstance(output_format, str):
        try:
            fmt = OutputFormat(output_format.lower())
        except ValueError:
            raise UsageError(f"Invalid output format: {output_format}")
    else:
        fmt = output_format

    try:
        if fmt == OutputFormat.CSV:
            df.write_csv(path)
        elif fmt == OutputFormat.JSONL:
            df.write_ndjson(path)
        elif fmt == OutputFormat.PARQUET:
            df.write_parquet(path)
        elif fmt == OutputFormat.JSON:
            rows = df.to_dicts()
            with open(path, "w") as f:
                json.dump(rows, f, indent=2)
    except Exception as e:
        raise RuntimeIOError(f"Failed to write to {path}: {e}")


def write_json(
    data: dict[str, Any] | list[Any],
    dest: str,
    *,
    indent: int = 2,
) -> None:
    """Write JSON data to stdout or file.

    Args:
        data: JSON-serializable data.
        dest: Destination. Use "-" for stdout, or file path.
        indent: Indentation level for pretty printing.

    Raises:
        RuntimeIOError: For I/O failures.
    """
    try:
        json_str = json.dumps(data, indent=indent, default=_json_serializer)

        if dest == "-":
            sys.stdout.write(json_str + "\n")
            sys.stdout.flush()
        else:
            path = Path(dest)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                f.write(json_str + "\n")
    except Exception as e:
        raise RuntimeIOError(f"Failed to write JSON: {e}")


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for numpy/polars types."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def is_stdin_source(source: str) -> bool:
    """Check if source refers to stdin."""
    return source == "-"


def is_stdout_dest(dest: str) -> bool:
    """Check if destination refers to stdout."""
    return dest == "-"


def is_tty(stream: TextIO | BinaryIO | None = None) -> bool:
    """Check if stream is a TTY (terminal).

    Args:
        stream: Stream to check. Defaults to stdout.

    Returns:
        True if stream is a TTY.
    """
    if stream is None:
        stream = sys.stdout
    try:
        return stream.isatty()
    except AttributeError:
        return False


def stderr_print(*args: Any, **kwargs: Any) -> None:
    """Print to stderr (for logs, progress, human-readable output).

    This should be used for all non-data output to keep stdout clean.
    """
    print(*args, file=sys.stderr, **kwargs)
