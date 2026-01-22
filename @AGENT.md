# Agent Build Instructions

## Project: dyxgb (Dynamic XGBoost)

## Setup
```bash
cd ~/code/dyxgb
pip install -e ".[dev]"
```

## Running Tests
```bash
pytest tests/ -v
```

## Type Checking
```bash
mypy src/dyxgb/
```

## Linting
```bash
ruff check src/dyxgb/
```

## Key Files for This Feature
- `src/dyxgb/transforms/features.py` - Current feature transform (uses eval for Polars expressions)
- `src/dyxgb/config.py` - Config dataclasses including FeatureConfig
- `src/dyxgb/cli.py` - Typer CLI app (add `functions` command here)
- `src/dyxgb/interactive.py` - InquirerPy TUI (add function selection here)
- `config.example.yaml` - Example config file (add function examples)

## Quality Standards
- All tests must pass (`pytest tests/ -v`)
- Follow existing code patterns and style
- Use Polars for all data operations
- Update config.example.yaml with working examples

## Git Workflow
```bash
git add <specific files>
git commit -m "feat(transforms): descriptive message"
```
