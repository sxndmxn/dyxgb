# Fix Plan - Feature Engineering Dictionary

## Priority Tasks

### Phase 1: Core Registry
- [ ] Create `src/dyxgb/transforms/registry.py` with FUNCTION_REGISTRY dict
- [ ] Implement math functions (log, square, sqrt, abs, clip)
- [ ] Implement two-column functions (ratio, difference, product)
- [ ] Implement threshold and bin functions
- [ ] Implement string functions (length, lower, upper, contains)
- [ ] Implement date functions (dayofweek, month, year, days_since)
- [ ] Implement null functions (fillna, is_null)

### Phase 2: Config Integration
- [ ] Update `src/dyxgb/config.py` FeatureConfig dataclass to support function + params
- [ ] Update `src/dyxgb/transforms/features.py` to resolve functions from registry
- [ ] Update `config.example.yaml` with human-readable function examples

### Phase 3: CLI & TUI
- [ ] Add `dyxgb functions` command to `src/dyxgb/cli.py`
- [ ] Update `src/dyxgb/interactive.py` with function selection list

### Phase 4: Testing & Verification
- [ ] Create `tests/test_registry.py` with tests for each function
- [ ] Run full test suite and fix any failures
- [ ] Test end-to-end with sample config using new function syntax

## Completed
(Items move here when done)
