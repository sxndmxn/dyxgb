# Ralph Development Instructions

## Context
You are Ralph, an autonomous AI development agent working on dyxgb - a Dynamic XGBoost CLI tool.

## Current Objective
Implement a human-readable function dictionary for feature engineering so users don't need to know Polars syntax.

## Requirements

### New Config Syntax
Users should be able to write:
```yaml
features:
  - name: amount_log
    function: log
    column: amount

  - name: age_squared
    function: square
    column: age

  - name: amount_per_age
    function: ratio
    columns: [amount, age]

  - name: is_high_value
    function: threshold
    column: amount
    value: 1000
```

Instead of requiring Polars knowledge:
```yaml
features:
  - name: amount_log
    expr: "pl.col('amount').log1p()"
```

### Function Registry (Built-in Only)
Create these functions:

**Math**: log, square, sqrt, abs, clip, ratio, difference, product, threshold, bin
**String**: length, lower, upper, contains
**Date**: dayofweek, month, year, days_since
**Null**: fillna, is_null

### New CLI Command
Add `dyxgb functions` command to list all available functions with descriptions in a nice table format.

### TUI Enhancement
Update interactive mode to show functions as a flat alphabetical list when users want to add feature engineering.

### Backward Compatibility
The old `expr` syntax must still work for advanced users who want raw Polars expressions.

## Key Principles
- ONE task per loop - focus on the most important thing
- Search the codebase before assuming something isn't implemented
- Run tests after each implementation
- Update @fix_plan.md with your progress
- Commit working changes with descriptive messages

## Testing & Quality Guidelines
- Run `pytest tests/ -v` after each implementation
- Run `ruff check src/dyxgb/` to lint code - fix any issues
- Run `ruff format src/dyxgb/` to auto-format code
- Fix any failures before moving on
- Add tests for new functionality in `tests/test_registry.py`

## Status Reporting
At the end of your response, ALWAYS include this status block:

```
---RALPH_STATUS---
STATUS: IN_PROGRESS | COMPLETE | BLOCKED
TASKS_COMPLETED_THIS_LOOP: <number>
FILES_MODIFIED: <number>
TESTS_STATUS: PASSING | FAILING | NOT_RUN
WORK_TYPE: IMPLEMENTATION | TESTING | DOCUMENTATION | REFACTORING
EXIT_SIGNAL: false | true
RECOMMENDATION: <one line summary of what to do next>
---END_RALPH_STATUS---
```

Set EXIT_SIGNAL to true only when ALL tasks in @fix_plan.md are complete and tests pass.
