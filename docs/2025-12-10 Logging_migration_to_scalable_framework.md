# Logging Migration to Scalable Framework

**Date:** 2025-12-10
**Component:** `brio_ext/metrics/logger.py`

## Summary

Updated the MetricsLogger to use platform-appropriate default paths instead of hardcoded `~/.briodocs/metrics.jsonl`, aligning with BrioDocs' path consolidation strategy.

## Changes Made

### 1. Added Platform Detection
Added `import platform` to detect the operating system.

### 2. New Helper Function: `_get_default_metrics_path()`
A new function that returns the appropriate metrics log path based on:

- **Platform:**
  - macOS: `~/Library/Application Support/BrioDocs/`
  - Windows: `%APPDATA%/BrioDocs/`
  - Linux: `~/.config/briodocs/`

- **Environment:**
  - Production (default): `logs/metrics.jsonl`
  - Development (`BRIODOCS_ENV=dev`): `logs-dev/metrics.jsonl`

### 3. Updated `MetricsLogger.__init__`
Changed the default path resolution to use the new `_get_default_metrics_path()` function while preserving backward compatibility via the `BRIODOCS_METRICS_PATH` environment variable.

## Behavior Matrix

| Scenario | BRIODOCS_ENV | Metrics Path (macOS) |
|----------|--------------|----------------------|
| Electron (prod) | not set (default) | `~/Library/Application Support/BrioDocs/logs/metrics.jsonl` |
| Dev script | `dev` | `~/Library/Application Support/BrioDocs/logs-dev/metrics.jsonl` |
| Override | any | Uses `BRIODOCS_METRICS_PATH` env var if set |

## Benefits

1. **Platform Consistency:** Uses OS-appropriate data directories
2. **Dev/Prod Isolation:** Separate log paths prevent mixing development and production data
3. **Cross-Platform Support:** Works correctly on macOS, Windows, and Linux
4. **Backward Compatible:** Existing `BRIODOCS_METRICS_PATH` override still works

## Testing

```bash
# Test prod mode (default)
python -c "from brio_ext.metrics.logger import MetricsLogger; m = MetricsLogger(); print(m.log_path)"
# Expected: .../BrioDocs/logs/metrics.jsonl

# Test dev mode
BRIODOCS_ENV=dev python -c "from brio_ext.metrics.logger import MetricsLogger; m = MetricsLogger(); print(m.log_path)"
# Expected: .../BrioDocs/logs-dev/metrics.jsonl
```

## Related Work

This change is part of the BrioDocs path consolidation effort to centralize all application data in platform-appropriate locations.
